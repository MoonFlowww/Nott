#ifndef Nott_BATCH_ITERATOR_HPP
#define Nott_BATCH_ITERATOR_HPP
/**
 * @file batch_iterator.hpp
 * @brief Three batch-iteration strategies collapsed from the old 12-path
 *        combinatorics down to 3 runtime-dispatched functions.
 *
 * Strategy selection:
 *   - prefetch_available  -> iterate_batches_prefetch()  (CUDA double-buffer)
 *   - use_buffer()        -> iterate_batches_buffered()  (CPU-side deque)
 *   - otherwise           -> iterate_batches_simple()    (straight for-loop)
 *
 * All three share the same fetch_batch() + process_batch() pattern.
 * The policy struct provides all flags; no template booleans leak into
 * the call site.
 */

#include <cstddef>
#include <deque>
#include <utility>
#include <stdexcept>
#include <torch/torch.h>
#include "training_policy.hpp"
#include "dataset_pipeline.hpp"

#ifdef TORCH_CUDA_AVAILABLE
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAStream.h>
#endif

namespace Nott::Training {

/**
 * @brief Build a batch slice from the dataset.
 *
 * Returns {inputs, targets} for batch @p batch_index.  Empty tensors
 * signal end-of-data.
 *
 * @tparam TensorDataset  Any struct with .inputs and .targets members.
 * @tparam FetchBatchFn   Callable type (used for ADL only, not invoked).
 * @param dataset              Source dataset.
 * @param batch_index          Zero-based batch number.
 * @param batch_size           Number of samples per batch.
 * @param total_samples        Total sample count in the dataset.
 * @param epoch_indices        Shuffle permutation (empty if no shuffle).
 * @param channels_last_inputs Whether to enforce ChannelsLast on inputs.
 * @return Pair of {inputs, targets} tensors, or empty tensors on EOF.
 */
template<typename TensorDataset, typename FetchBatchFn>
auto fetch_batch_impl(
    const TensorDataset&  dataset,
    std::size_t           batch_index,
    std::size_t           batch_size,
    std::int64_t          total_samples,
    const torch::Tensor&  epoch_indices,
    bool                  channels_last_inputs,
    FetchBatchFn&&        /*unused -- for ADL only*/)
    -> std::pair<torch::Tensor, torch::Tensor>
{
    const auto offset = static_cast<std::int64_t>(batch_index) * static_cast<std::int64_t>(batch_size);
    const auto remaining = total_samples - offset;
    const auto current_batch = std::min<std::int64_t>(static_cast<std::int64_t>(batch_size), remaining);
    if (current_batch <= 0) return {};

    torch::Tensor batch_inputs, batch_targets;

    if (epoch_indices.defined()) {
        /// Shuffle path
        auto batch_indices = epoch_indices.narrow(0, offset, current_batch);
        if (!batch_indices.device().is_cpu())
            batch_indices = batch_indices.to(torch::kCPU);
        batch_inputs  = dataset.inputs.index_select(0, batch_indices);
        batch_targets = dataset.targets.index_select(0, batch_indices);
        batch_inputs  = ensure_memory_format(std::move(batch_inputs),
            channels_last_inputs ? torch::MemoryFormat::ChannelsLast : torch::MemoryFormat::Contiguous);
        batch_targets = ensure_memory_format(std::move(batch_targets),
            torch::MemoryFormat::Contiguous);
        batch_inputs  = ensure_pinned(std::move(batch_inputs));
        batch_targets = ensure_pinned(std::move(batch_targets));
    } else {
        /// Sequential (narrow) path, zero-copy view
        batch_inputs  = dataset.inputs.narrow(0, offset, current_batch);
        batch_targets = dataset.targets.narrow(0, offset, current_batch);
    }
    return {std::move(batch_inputs), std::move(batch_targets)};
}

/**
 * @brief Strategy 1: simple linear iteration.
 *
 * No buffering, no prefetch.  Suitable for CPU training or small datasets.
 *
 * @tparam TensorDataset   Dataset type.
 * @tparam ProcessBatchFn  Callable void(Tensor inputs, Tensor targets).
 */
template<typename TensorDataset, typename ProcessBatchFn>
void iterate_batches_simple(
    TensorDataset&        dataset,
    const TrainingPolicy& policy,
    std::int64_t          total_samples,
    const torch::Tensor&  epoch_indices,
    bool                  channels_last_inputs,
    ProcessBatchFn&&      process_batch)
{
    const auto total_batches = static_cast<std::size_t>(
        (total_samples + static_cast<std::int64_t>(policy.batch_size) - 1)
        / static_cast<std::int64_t>(policy.batch_size));

    for (std::size_t i = 0; i < total_batches; ++i) {
        auto [inputs, targets] = fetch_batch_impl(
            dataset, i, policy.batch_size, total_samples,
            epoch_indices, channels_last_inputs, process_batch);
        if (!inputs.defined() || !targets.defined()) break;
        process_batch(std::move(inputs), std::move(targets));
    }
}

/**
 * @brief Strategy 2: CPU-side deque buffering.
 *
 * Maintains a look-ahead window of `buffer_vram + 1` batches on CPU.
 *
 * @tparam TensorDataset   Dataset type.
 * @tparam ProcessBatchFn  Callable void(Tensor inputs, Tensor targets).
 */
template<typename TensorDataset, typename ProcessBatchFn>
void iterate_batches_buffered(
    TensorDataset&        dataset,
    const TrainingPolicy& policy,
    std::int64_t          total_samples,
    const torch::Tensor&  epoch_indices,
    bool                  channels_last_inputs,
    ProcessBatchFn&&      process_batch)
{
    const auto total_batches = static_cast<std::size_t>(
        (total_samples + static_cast<std::int64_t>(policy.batch_size) - 1)
        / static_cast<std::int64_t>(policy.batch_size));
    const std::size_t max_batches = total_batches == 0 ? 1 : total_batches;
    const std::size_t buffer_limit = std::max<std::size_t>(1,
        std::min<std::size_t>(policy.buffer_vram + 1, max_batches));

    std::deque<std::pair<torch::Tensor, torch::Tensor>> buffered;
    std::size_t next_to_load = 0;

    auto maintain_buffer = [&](std::size_t current) {
        const std::size_t desired = std::min(buffer_limit, total_batches - current);
        while (buffered.size() < desired && next_to_load < total_batches) {
            auto batch = fetch_batch_impl(
                dataset, next_to_load, policy.batch_size, total_samples,
                epoch_indices, channels_last_inputs, process_batch);
            buffered.push_back(std::move(batch));
            ++next_to_load;
        }
    };

    for (std::size_t i = 0; i < total_batches; ++i) {
        maintain_buffer(i);
        if (buffered.empty()) break;
        auto [inputs, targets] = std::move(buffered.front());
        buffered.pop_front();
        process_batch(std::move(inputs), std::move(targets));
        maintain_buffer(i + 1);
    }
}

/**
 * @brief Strategy 3: CUDA prefetch (double-buffered).
 *
 * Overlaps CPU-to-GPU transfers with GPU computation using two CUDA streams.
 */

#ifdef TORCH_CUDA_AVAILABLE

/**
 * @struct PrefetchState
 * @brief Double-buffer state for CUDA prefetch iteration.
 */
struct alignas(64) PrefetchState {
    /** @brief Construct with a CUDA stream from the default pool. */
    explicit PrefetchState(int device_index)
        : stream(c10::cuda::getStreamFromPool(/*high=*/false, device_index)) {}

    /// Hot path -- read on every batch iteration, packed into one cache line
    std::array<bool, 2> pending{false, false};
    std::array<bool, 2> input_stable{false, false};
    std::array<bool, 2> target_stable{false, false};
    std::array<bool, 2> consumed_valid{false, false};

    /// Cold path -- accessed once per transfer, separate cache lines
    c10::cuda::CUDAStream stream;
    std::array<torch::Tensor, 2> inputs{};
    std::array<torch::Tensor, 2> targets{};
    std::array<at::cuda::CUDAEvent, 2> events{};    // copy-done, compute waits before reading
    std::array<at::cuda::CUDAEvent, 2> consumed{};  // compute-done, prefetch waits before reusing the slot
};

/**
 * @brief Run prefetch-based batch iteration.
 *
 * @tparam TensorDataset   Dataset type.
 * @tparam FetchBatchFn    Callable returning {inputs, targets}.
 * @tparam ProcessBatchFn  Callable void(Tensor inputs, Tensor targets).
 */
template<typename TensorDataset, typename FetchBatchFn, typename ProcessBatchFn>
void iterate_batches_prefetch(
    TensorDataset&        dataset,
    const TrainingPolicy& policy,
    std::int64_t          total_samples,
    const torch::Tensor&  epoch_indices,
    bool                  channels_last_inputs,
    PrefetchState&        pstate,
    const torch::Device&  device,
    FetchBatchFn&&        fetch_batch,
    ProcessBatchFn&&      process_batch)
{
    const auto total_batches = static_cast<std::size_t>(
        (total_samples + static_cast<std::int64_t>(policy.batch_size) - 1)
        / static_cast<std::int64_t>(policy.batch_size));

    auto schedule = [&](std::size_t batch_index, std::size_t slot) -> bool {
        pstate.pending[slot] = false;
        auto [inputs, targets] = fetch_batch_impl(
            dataset, batch_index, policy.batch_size, total_samples,
            epoch_indices, channels_last_inputs, fetch_batch);
        if (!inputs.defined() || !targets.defined()) return false;

        // WAR: don't overwrite this slot's buffers until the compute that last read them is done
        if (pstate.consumed_valid[slot])
            pstate.consumed[slot].block(pstate.stream);

        // copies go on the prefetch stream (that is what overlaps them with compute)
        pstate.inputs[slot] = stage_to_device(std::move(inputs),
            pstate.inputs[slot], pstate.input_stable[slot], device,
            channels_last_inputs, /*force_non_blocking=*/true, &pstate.stream);
        pstate.targets[slot] = stage_to_device(std::move(targets),
            pstate.targets[slot], pstate.target_stable[slot], device,
            /*apply_channels_last=*/false, /*force_non_blocking=*/true, &pstate.stream);

        pstate.events[slot].record(pstate.stream);
        pstate.pending[slot] = true;
        return true;
    };

    auto wait = [&](std::size_t slot) {
        if (pstate.pending[slot]) {
            // make the compute stream wait for this slot's copy to finish before reading it
            auto cur = at::cuda::getCurrentCUDAStream(device.index());
            pstate.events[slot].block(cur);
            pstate.pending[slot] = false;
        }
    };

    std::size_t slot = 0;
    bool has_batch = schedule(0, slot);
    for (std::size_t i = 0; i < total_batches && has_batch; ++i) {
        wait(slot);
        process_batch(pstate.inputs[slot], pstate.targets[slot]);
        // mark compute done reading this slot so the next copy into it can safely wait
        pstate.consumed[slot].record(at::cuda::getCurrentCUDAStream(device.index()));
        pstate.consumed_valid[slot] = true;
        std::size_t next = slot ^ 1U;
        has_batch = schedule(i + 1, next);
        slot = next;
    }
}

#endif // TORCH_CUDA_AVAILABLE

/**
 * @brief Unified batch-iteration dispatch.
 *
 * Picks the right strategy based on @p policy and calls @p process_batch
 * for every batch.
 *
 * @tparam TensorDataset   Dataset type.
 * @tparam ProcessBatchFn  void(Tensor inputs, Tensor targets).
 * @tparam FetchBatchFn    (optional) custom fetch function.
 * @param dataset              Training dataset.
 * @param policy               Resolved training policy.
 * @param total_samples        Number of samples.
 * @param epoch_indices        Shuffle permutation (empty if none).
 * @param channels_last_inputs Whether to enforce ChannelsLast.
 * @param process_batch        Called for each batch.
 * @param fetch_batch          Custom fetch (defaults to fetch_batch_impl).
 * @param prefetch_state       CUDA prefetch state (CUDA-only).
 * @return Total samples processed.
 */
template<typename TensorDataset, typename ProcessBatchFn, typename FetchBatchFn = ProcessBatchFn>
std::int64_t iterate_batches(
    TensorDataset&        dataset,
    const TrainingPolicy& policy,
    std::int64_t          total_samples,
    const torch::Tensor&  epoch_indices,
    bool                  channels_last_inputs,
    ProcessBatchFn&&      process_batch,
    FetchBatchFn&&        fetch_batch
#ifdef TORCH_CUDA_AVAILABLE
    , PrefetchState*      prefetch_state = nullptr
#endif
    )
{
    (void) fetch_batch; // used by prefetch path via ADL / overload

#ifdef TORCH_CUDA_AVAILABLE
    if (policy.use_prefetch() && prefetch_state) {
        const auto device = dataset.inputs.defined()
            ? dataset.inputs.device() : torch::Device(torch::kCPU);
        iterate_batches_prefetch(
            dataset, policy, total_samples, epoch_indices,
            channels_last_inputs, *prefetch_state, device,
            std::forward<FetchBatchFn>(fetch_batch),
            std::forward<ProcessBatchFn>(process_batch));
        return static_cast<std::int64_t>(total_samples);
    }
#endif

    if (policy.use_buffer()) {
        iterate_batches_buffered(
            dataset, policy, total_samples, epoch_indices,
            channels_last_inputs, std::forward<ProcessBatchFn>(process_batch));
    } else {
        iterate_batches_simple(
            dataset, policy, total_samples, epoch_indices,
            channels_last_inputs, std::forward<ProcessBatchFn>(process_batch));
    }
    return static_cast<std::int64_t>(total_samples);
}

} // namespace Nott::Training

#endif // Nott_BATCH_ITERATOR_HPP
