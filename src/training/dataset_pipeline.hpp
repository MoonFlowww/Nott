#ifndef Nott_DATASET_PIPELINE_HPP
#define Nott_DATASET_PIPELINE_HPP
/**
 * @file dataset_pipeline.hpp
 * @brief Single point of truth for tensor preparation.
 *
 * Replaces the duplicated ensure_layout / ensure_contiguous / ensure_cpu /
 * pin_memory logic that was scattered across train(), run_epochs(),
 * compute_dataset_loss(), and prepare_tensor_dataset().
 *
 * Every helper is a pure function: same inputs produce same outputs, no
 * hidden state, trivially testable.
 *
 * The TensorDataset type is defined in core.hpp
 * (TrainingDetails::TensorDataset).  Functions accepting/returning it are
 * templates to avoid circular includes.
 */

#include <cstddef>
#include <utility>
#include <torch/torch.h>

#ifdef TORCH_CUDA_AVAILABLE
#include <ATen/cuda/CUDAStream.h>
#endif

namespace Nott::Training {

/**
 * @brief Ensure a tensor is contiguous in the requested memory format.
 *
 * Returns the input unchanged if already satisfied -- avoids redundant
 * GPU-to-GPU copies that were silently triggered by the old code path.
 *
 * @param tensor  Input tensor (may be undefined).
 * @param fmt     Desired memory format (default: Contiguous).
 * @return        Tensor guaranteed to be contiguous in @p fmt.
 */
[[nodiscard]] inline torch::Tensor ensure_memory_format(
    torch::Tensor tensor,
    torch::MemoryFormat fmt = torch::MemoryFormat::Contiguous)
{
    if (!tensor.defined()) return tensor;

    if (fmt == torch::MemoryFormat::ChannelsLast && tensor.dim() >= 4) {
        if (!tensor.is_contiguous(torch::MemoryFormat::ChannelsLast))
            tensor = tensor.contiguous(torch::MemoryFormat::ChannelsLast);
    } else {
        if (!tensor.is_contiguous())
            tensor = tensor.contiguous();
    }
    return tensor;
}

/**
 * @brief Pin memory once at the CPU-to-GPU boundary.
 *
 * Old code called pin_memory() in 4 different places.  Pin exactly once,
 * right before training begins.  Already-pinned tensors are no-ops
 * (checked internally by libtorch).
 *
 * @param tensor  Input tensor.
 * @return        Tensor with pinned memory if it was on CPU.
 */
[[nodiscard]] inline torch::Tensor ensure_pinned(torch::Tensor tensor) {
#ifdef TORCH_CUDA_AVAILABLE
    // ifdef: kernel compiled in. is_available: GPU actually present. Both needed.
    if (torch::cuda::is_available() && tensor.defined() && tensor.device().is_cpu() && !tensor.is_pinned())
        tensor = tensor.pin_memory();
#endif
    return tensor;
}

/**
 * @brief Prepare a full TensorDataset in one call.
 *
 * Combines ensure_memory_format() + ensure_pinned() in the correct order:
 *   1. Make contiguous in the desired format.
 *   2. Pin (if CPU).
 *
 * @tparam TensorDataset  Any struct with .inputs and .targets fields.
 * @param inputs   Input tensor.
 * @param targets  Target tensor.
 * @param fmt      Memory format for inputs (targets always Contiguous).
 * @return         Prepared dataset.
 */
template<typename TensorDataset>
[[nodiscard]] TensorDataset prepare_dataset(
    torch::Tensor inputs,
    torch::Tensor targets,
    torch::MemoryFormat fmt = torch::MemoryFormat::Contiguous)
{
    inputs  = ensure_memory_format(std::move(inputs), fmt);
    targets = ensure_memory_format(std::move(targets), torch::MemoryFormat::Contiguous);

    inputs  = ensure_pinned(std::move(inputs));
    targets = ensure_pinned(std::move(targets));

    return {std::move(inputs), std::move(targets)};
}

/**
 * @brief Move a dataset to CPU (for VRAM-buffered training paths).
 *
 * @tparam TensorDataset  Any struct with .inputs and .targets.
 * @param ds   Dataset to transfer.
 * @param fmt  Memory format to preserve during the transfer.
 * @return     Dataset on CPU with pinned memory.
 */
template<typename TensorDataset>
[[nodiscard]] TensorDataset ensure_cpu(
    TensorDataset ds,
    torch::MemoryFormat fmt = torch::MemoryFormat::Contiguous)
{
    if (ds.inputs.defined() && !ds.inputs.device().is_cpu()) {
        if (fmt == torch::MemoryFormat::ChannelsLast && ds.inputs.dim() >= 4) {
            auto opts = ds.inputs.options().device(torch::kCPU);
            ds.inputs = ds.inputs.to(opts, /*non_blocking=*/false, /*copy=*/false,
                                      torch::MemoryFormat::ChannelsLast);
        } else {
            ds.inputs = ds.inputs.to(torch::kCPU);
        }
    }
    if (ds.targets.defined() && !ds.targets.device().is_cpu()) {
        ds.targets = ds.targets.to(torch::kCPU);
    }
    ds.inputs  = ensure_pinned(std::move(ds.inputs));
    ds.targets = ensure_pinned(std::move(ds.targets));
    return ds;
}

/**
 * @brief Stage a tensor to a device with buffer reuse.
 *
 * Transfers @p tensor to @p device, reusing @p buffer when sizes match.
 * Replaces the 36-line `stage_to_device` lambda from run_epochs().
 *
 * @param tensor              Input tensor.
 * @param buffer              Reusable device buffer (mutated).
 * @param buffer_stable       Whether buffer can be reused (mutated).
 * @param device              Target device.
 * @param apply_channels_last Whether to enforce ChannelsLast layout.
 * @param force_non_blocking  Force non-blocking copy even if not pinned.
 * @param prefetch_stream     Optional CUDA stream for prefetch copies.
 * @return                    Tensor on @p device (may alias @p buffer).
 */
[[nodiscard]] inline torch::Tensor stage_to_device(
    torch::Tensor tensor,
    torch::Tensor& buffer,
    bool&          buffer_stable,
    const torch::Device& device,
    bool           apply_channels_last,
    bool           force_non_blocking = false
#ifdef TORCH_CUDA_AVAILABLE
    , torch::cuda::CUDAStream* prefetch_stream = nullptr
#endif
    )
{
    tensor = ensure_memory_format(std::move(tensor),
        apply_channels_last ? torch::MemoryFormat::ChannelsLast : torch::MemoryFormat::Contiguous);

    if (!tensor.defined() || tensor.device() == device)
        return tensor;

    auto options = tensor.options().device(device);
    const bool non_blocking = force_non_blocking || tensor.is_pinned();
    const bool needs_cl = apply_channels_last && tensor.dim() >= 4;

    if (buffer_stable && buffer.defined() && !buffer.sizes().equals(tensor.sizes()))
        buffer_stable = false;

    if (!buffer_stable) {
        if (!buffer.defined() || buffer.device() != device
            || buffer.scalar_type() != tensor.scalar_type()
            || !buffer.sizes().equals(tensor.sizes())
            || (needs_cl ? !buffer.is_contiguous(torch::MemoryFormat::ChannelsLast)
                         : !buffer.is_contiguous())) {
            auto fmt = needs_cl ? torch::MemoryFormat::ChannelsLast : torch::MemoryFormat::Contiguous;
            buffer = torch::empty(tensor.sizes(), options, fmt);
        } else {
            buffer_stable = true;
        }
    }

#ifdef TORCH_CUDA_AVAILABLE
    if (prefetch_stream) {
        torch::cuda::CUDAStreamGuard guard(*prefetch_stream);
        buffer.copy_(tensor, non_blocking);
    } else {
        buffer.copy_(tensor, non_blocking);
    }
#else
    buffer.copy_(tensor, non_blocking);
#endif
    return buffer;
}

} // namespace Nott::Training

#endif // Nott_DATASET_PIPELINE_HPP
