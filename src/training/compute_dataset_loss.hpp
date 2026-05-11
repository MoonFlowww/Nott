#ifndef Nott_COMPUTE_DATASET_LOSS_HPP
#define Nott_COMPUTE_DATASET_LOSS_HPP

#include <chrono>
#include <cstddef>
#include <optional>
#include <stdexcept>

#include <torch/torch.h>

#include "dataset_pipeline.hpp"
#include "deferred_scalar.hpp"
#include "../common/streaming.hpp"

namespace Nott::Training {

/**
 * @brief Evaluate weighted mean loss over a full dataset without grad.
 *
 * Templated on ModelT and DatasetT to avoid including core.hpp.
 *
 * ModelT must expose (publicly):
 *   device(), has_regularization(), is_training(), train(), eval(),
 *   stream_forward(), compute_loss(), compute_regularization_penalty(),
 *   collect_learning_rates(), record_dataset_loss_telemetry()
 *
 * DatasetT must have .inputs and .targets tensor fields.
 *
 * @return Weighted mean loss as a scalar double, or nullopt if the dataset
 *         is empty or undefined.
 */
template<typename ModelT, typename DatasetT>
std::optional<double> compute_dataset_loss(
    ModelT&            model,
    const DatasetT&    dataset,
    std::size_t        batch_size,
    bool               use_buffer,
    std::size_t        buffer_vram,
    torch::MemoryFormat memory_format = torch::MemoryFormat::Contiguous)
{
    if (!dataset.inputs.defined() || !dataset.targets.defined())
        return std::nullopt;
    if (dataset.inputs.size(0) == 0)
        return std::nullopt;
    if (batch_size == 0)
        throw std::invalid_argument(
            "Batch size must be greater than zero when computing dataset loss.");

    const auto device              = model.device();
    const auto total_samples       = dataset.inputs.size(0);
    const bool regularization_active = model.has_regularization();
    const bool channels_last       = memory_format == torch::MemoryFormat::ChannelsLast;

    torch::NoGradGuard no_grad;
    const bool was_training = model.is_training();
    model.eval();

    torch::Tensor ds_inputs  = dataset.inputs;
    torch::Tensor ds_targets = dataset.targets;

    if (use_buffer) {
        if (!device.is_cuda())
            throw std::runtime_error(
                "VRAM buffering for dataset loss requires the model to be on a CUDA device.");
        if (ds_inputs.defined()  && !ds_inputs.device().is_cpu())
            ds_inputs  = ds_inputs.to(torch::kCPU);
        if (ds_targets.defined() && !ds_targets.device().is_cpu())
            ds_targets = ds_targets.to(torch::kCPU);
    }

    ds_inputs  = ensure_pinned(std::move(ds_inputs));
    ds_targets = ensure_pinned(std::move(ds_targets));

    const auto batch_extent    = static_cast<std::int64_t>(batch_size);
    const std::size_t total_batches = total_samples > 0
        ? static_cast<std::size_t>((total_samples + batch_extent - 1) / batch_extent)
        : 0;

    double       accumulation_val = 0.0;
    std::int64_t weight           = 0;

    torch::Tensor input_buf, target_buf;
    bool          input_buf_stable = false, target_buf_stable = false;

    auto prepare_batch = [&](torch::Tensor batch_inputs, torch::Tensor batch_targets)
        -> std::optional<Nott::StreamingBatch>
    {
        if (!batch_inputs.defined() || !batch_targets.defined())
            return std::nullopt;

        auto staged_inputs  = stage_to_device(std::move(batch_inputs),
                                               input_buf, input_buf_stable,
                                               device, channels_last);
        auto staged_targets = stage_to_device(std::move(batch_targets),
                                               target_buf, target_buf_stable,
                                               device, false);

        if (!staged_inputs.defined() || !staged_targets.defined())
            return std::nullopt;

        Nott::StreamingBatch batch{};
        batch.inputs  = std::move(staged_inputs);
        batch.targets = std::move(staged_targets);
        return batch;
    };

    auto consume_batch = [&](torch::Tensor prediction, Nott::StreamingBatch batch) {
        if (!prediction.defined() || !batch.targets.defined())
            return;

        auto targets = std::move(batch.targets);
        const auto current_batch = targets.size(0);
        if (current_batch <= 0)
            return;

        if (!prediction.sizes().equals(targets.sizes()) &&
            targets.numel() == prediction.numel())
            targets = targets.reshape_as(prediction);

        auto loss = model.compute_loss(prediction, targets);
        if (loss.dim() != 0)
            loss = loss.mean();

        if (regularization_active) {
            auto penalty = model.compute_regularization_penalty();
            if (penalty.defined()) {
                if (penalty.device() != loss.device())
                    penalty = penalty.to(loss.device());
                if (penalty.scalar_type() != loss.scalar_type())
                    penalty = penalty.to(loss.scalar_type());
                loss = loss + penalty;
            }
        }

        accumulation_val += loss.detach().template item<double>() * static_cast<double>(current_batch);
        weight           += current_batch;
    };

    Nott::StreamingOptions streaming_options{};
    streaming_options.batch_size = batch_size;
    if (use_buffer) {
        const std::size_t max_batches = total_batches == 0 ? 1 : total_batches;
        streaming_options.buffer_batches = std::max<std::size_t>(
            1, std::min<std::size_t>(buffer_vram + 1, max_batches));
    }

    model.stream_forward(std::move(ds_inputs), std::move(ds_targets),
                         streaming_options, prepare_batch, consume_batch);

    if (was_training)
        model.train();
    else
        model.eval();

    if (weight == 0)
        return std::nullopt;

    const double avg_loss = accumulation_val / static_cast<double>(weight);

    auto loss_tensor  = torch::tensor(avg_loss,
                            torch::TensorOptions().dtype(torch::kFloat64));
    const torch::Device cpu_device{torch::kCPU};
    auto loss_scalar  = DeferredScalar::from_tensor(std::move(loss_tensor), cpu_device);

    auto learning_rates = model.collect_learning_rates();
    const auto timestamp = std::chrono::system_clock::now();

    using DatasetLossSnapshot = typename ModelT::TrainingTelemetry::DatasetLossSnapshot;
    model.record_dataset_loss_telemetry(DatasetLossSnapshot{
        loss_scalar,
        static_cast<std::size_t>(dataset.inputs.size(0)),
        std::move(learning_rates),
        timestamp
    });

    return loss_scalar.materialize();
}

} // namespace Nott::Training

#endif // Nott_COMPUTE_DATASET_LOSS_HPP
