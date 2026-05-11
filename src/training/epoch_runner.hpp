#ifndef Nott_EPOCH_RUNNER_HPP
#define Nott_EPOCH_RUNNER_HPP
/**
 * @file epoch_runner.hpp
 * @brief Epoch loop wired to batch_iterator.hpp and training_policy.hpp.
 *
 * Responsibilities:
 *   1. Per-epoch index shuffle.
 *   2. Batch dispatch via iterate_batches() (prefetch / buffered / simple).
 *   3. Loss accumulation on device (no per-batch CPU sync).
 *   4. Test evaluation and best-state tracking.
 *   5. Epoch-end callback carrying all telemetry data.
 *
 * Template parameters decouple this header from core.hpp.
 */

#include <chrono>
#include <cstddef>
#include <optional>
#include <utility>
#include <vector>
#include <torch/torch.h>
#include "training_policy.hpp"
#include "dataset_pipeline.hpp"
#include "batch_iterator.hpp"

namespace Nott::Training {

/**
 * @struct EpochLogEntry
 * @brief Data produced at the end of each epoch.
 *
 * Passed to the on_epoch_end callback; caller is responsible for
 * telemetry recording and console output.
 */
struct EpochLogEntry {
    std::size_t  epoch_index{};
    std::size_t  total_epochs{};
    double       train_loss{0.0};
    std::optional<double> test_loss{};
    std::optional<double> delta{};
    bool         improved{false};
    double       duration_seconds{0.0};
    std::size_t  processed_steps{0};
    std::chrono::system_clock::time_point timestamp{};
};

/**
 * @brief Run the full epoch loop.
 *
 * @tparam TensorDataset   Struct with .inputs and .targets tensor members.
 * @tparam ModelT          Exposes forward(), compute_loss(), zero_grad(),
 *                         step_optimizers(), step_scheduler(), device(),
 *                         eval(), parameters(), buffers(),
 *                         update_regularization_states().
 * @tparam ComputeLossFn   std::optional<double>(ModelT&, Dataset&, Policy&)
 * @tparam OnEpochEndFn    void(const EpochLogEntry&)
 * @tparam StepFn          torch::Tensor(ModelT&, Tensor inputs, Tensor targets)
 */
template<
    typename TensorDataset,
    typename ModelT,
    typename ComputeLossFn,
    typename OnEpochEndFn,
    typename StepFn>
void run_epochs(
    ModelT&                              model,
    TensorDataset&                       train_dataset,
    const std::optional<TensorDataset>&  test_dataset,
    const TrainingPolicy&                policy,
    ComputeLossFn&&                      compute_test_loss,
    OnEpochEndFn&&                       on_epoch_end,
    StepFn&&                             training_step,
    std::size_t&                         global_step_index
#ifdef TORCH_CUDA_AVAILABLE
    , PrefetchState*                     prefetch_state = nullptr
#endif
)
{
    const auto device        = model.device();
    const auto total_samples = train_dataset.inputs.size(0);
    const bool channels_last = policy.memory_format == torch::MemoryFormat::ChannelsLast;

    std::optional<double>          best_test{};
    std::vector<torch::Tensor>     best_parameters;
    std::vector<torch::Tensor>     best_buffers;
    bool                           best_state_captured = false;

    torch::TensorOptions index_opts = torch::TensorOptions().dtype(torch::kLong);

    for (std::size_t epoch = 0; epoch < policy.epochs; ++epoch) {
        const auto epoch_start = std::chrono::steady_clock::now();

        /* shuffle indices */
        torch::Tensor epoch_indices;
        if (policy.shuffle) {
            epoch_indices = (total_samples > 1)
                ? torch::randperm(total_samples, index_opts)
                : torch::arange(total_samples, index_opts);
        }

        /* accumulate on device — no per-batch CPU sync */
        torch::Tensor accumulation = torch::zeros({},
            torch::TensorOptions().dtype(torch::kFloat64).device(device));
        std::int64_t total_weight  = 0;
        std::size_t  processed     = 0;

        auto process_one = [&](torch::Tensor inputs, torch::Tensor targets) {
            if (!inputs.defined() || !targets.defined()) return;
            const auto n = targets.size(0);
            if (n <= 0) return;

            auto loss = training_step(model, std::move(inputs), std::move(targets));

            model.step_scheduler();
            if (policy.regularization_active)
                model.update_regularization_states(global_step_index, true);
            ++global_step_index;
            ++processed;

            accumulation += loss.detach().to(torch::kFloat64) * static_cast<double>(n);
            total_weight += n;
        };

        iterate_batches(
            train_dataset, policy, total_samples, epoch_indices,
            channels_last, process_one, process_one
#ifdef TORCH_CUDA_AVAILABLE
            , prefetch_state
#endif
        );

        /* materialise train loss (one GPU sync per epoch) */
        double train_loss_val = 0.0;
        if (total_weight > 0)
            train_loss_val = (accumulation / static_cast<double>(total_weight))
                             .item<double>();

        /* test evaluation */
        std::optional<double> test_loss{};
        if (test_dataset)
            test_loss = compute_test_loss(model, *test_dataset, policy);

        /* best-state tracking */
        bool improved = false;
        std::optional<double> delta{};
        if (test_loss) {
            if (!best_test) {
                improved  = true;
                best_test = test_loss;
            } else {
                delta = *test_loss - *best_test;
                if (*test_loss < *best_test) {
                    improved  = true;
                    best_test = test_loss;
                }
            }
        }

        if (improved && policy.restore_best_state) {
            best_parameters.clear();
            best_buffers.clear();
            for (auto& p : model.parameters())
                best_parameters.push_back(p.defined()
                    ? p.detach().clone(torch::MemoryFormat::Preserve)
                    : torch::Tensor{});
            for (auto& b : model.buffers())
                best_buffers.push_back(b.defined()
                    ? b.detach().clone(torch::MemoryFormat::Preserve)
                    : torch::Tensor{});
            best_state_captured = true;
        }

        /* epoch-end callback */
        const double duration = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - epoch_start).count();

        on_epoch_end(EpochLogEntry{
            epoch + 1,
            policy.epochs,
            train_loss_val,
            test_loss,
            delta,
            improved,
            duration,
            processed,
            std::chrono::system_clock::now()
        });
    }

    /* restore best state */
    if (policy.restore_best_state && best_state_captured) {
        torch::NoGradGuard ng;
        auto params = model.parameters();
        for (std::size_t i = 0; i < std::min(params.size(), best_parameters.size()); ++i)
            if (params[i].defined() && best_parameters[i].defined())
                params[i].copy_(best_parameters[i]);
        auto bufs = model.buffers();
        for (std::size_t i = 0; i < std::min(bufs.size(), best_buffers.size()); ++i)
            if (bufs[i].defined() && best_buffers[i].defined())
                bufs[i].copy_(best_buffers[i]);
    }
}

} // namespace Nott::Training
#endif // Nott_EPOCH_RUNNER_HPP
