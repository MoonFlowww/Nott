#ifndef Nott_TRAINING_PREFLIGHT_HPP
#define Nott_TRAINING_PREFLIGHT_HPP
/**
 * @file training_preflight.hpp
 * @brief All validation and setup that happens once before the training
 *        loop starts.
 *
 * Extracted from the first ~35 lines of the old train(Tensor, Tensor,
 * TrainOptions) function.
 *
 * Every check is a pure function that either passes silently or throws.
 * The resolve_* functions compute derived configuration from the raw
 * TrainOptions so the hot path never re-derives them.
 */

#include <cstddef>
#include <stdexcept>
#include <string>
#include <torch/torch.h>
/// GraphMode fwd-declared via training_policy.hpp
#include "training_policy.hpp"

namespace Nott::Training {

/**
 * @brief Validate basic training prerequisites.
 *
 * @param has_optimizer  Whether an optimizer is configured.
 * @param has_loss       Whether a loss function is configured.
 * @param inputs         Training input tensor.
 * @param targets        Training target tensor.
 * @param batch_size     Requested batch size.
 * @throws std::logic_error      If optimizer or loss is missing.
 * @throws std::invalid_argument If tensors or batch size are invalid.
 */
inline void validate_training_prerequisites(
    bool has_optimizer,
    bool has_loss,
    const torch::Tensor& inputs,
    const torch::Tensor& targets,
    std::size_t batch_size)
{
    if (!has_optimizer)
        throw std::logic_error("Cannot train without an optimizer.");
    if (!has_loss)
        throw std::logic_error("Cannot train without a loss function.");
    if (!inputs.defined() || !targets.defined())
        throw std::invalid_argument("Training tensors must be defined.");
    if (inputs.dim() == 0 || targets.dim() == 0)
        throw std::invalid_argument("Training tensors must not be scalars.");
    if (inputs.size(0) != targets.size(0))
        throw std::invalid_argument(
            "Mismatched number of training samples between inputs and targets.");
    if (batch_size == 0)
        throw std::invalid_argument("Batch size must be greater than zero.");
}

/**
 * @brief Validate fold options and return the fold count.
 *
 * @param inputs          Training input tensor.
 * @param targets         Training target tensor.
 * @param fold_requested  Whether k-fold mode is requested.
 * @return Number of folds (1 if fold not requested).
 * @throws std::invalid_argument If fold dimensions are inconsistent.
 */
inline std::int64_t validate_fold_options(
    const torch::Tensor& inputs,
    const torch::Tensor& targets,
    bool fold_requested)
{
    if (!fold_requested) return 1;

    if (inputs.dim() < 2 || targets.dim() < 2)
        throw std::invalid_argument(
            "Folded datasets must expose at least two dimensions (folds and samples).");
    if (targets.size(0) != inputs.size(0))
        throw std::invalid_argument(
            "Folded inputs and targets must share the same number of folds.");
    if (targets.size(1) != inputs.size(1))
        throw std::invalid_argument(
            "Folded inputs and targets must share the same number of samples per fold.");

    const auto fc = inputs.size(0);
    if (fc < 2)
        throw std::invalid_argument(
            "K-fold training requires at least two folds when fold mode is enabled.");
    return fc;
}

/**
 * @brief Validate that VRAM buffering is only requested on CUDA devices.
 *
 * @param buffer_requested  Whether VRAM buffering was requested.
 * @param is_cuda           Whether the model is on a CUDA device.
 * @throws std::runtime_error If buffering is requested without CUDA.
 */
inline void validate_vram_buffering(bool buffer_requested, bool is_cuda) {
    if (buffer_requested && !is_cuda)
        throw std::runtime_error("VRAM buffering requires the model to be on a CUDA device.");
}

/**
 * @brief Determine whether ChannelsLast memory format is applicable.
 *
 * @param requested_fmt    User-requested memory format.
 * @param is_cuda          Whether the model is on CUDA.
 * @param has_conv_layers  Whether the model contains conv layers.
 * @param sample_input     A sample input tensor for dimension check.
 * @return true if ChannelsLast should be used.
 */
inline bool resolve_channels_last(
    torch::MemoryFormat requested_fmt,
    bool is_cuda,
    bool has_conv_layers,
    const torch::Tensor& sample_input)
{
    if (requested_fmt != torch::MemoryFormat::ChannelsLast) return false;
    if (!is_cuda || !has_conv_layers) return false;
#ifdef TORCH_CUDA_AVAILABLE
    if (!torch::cuda::is_available()) return false;
#endif
    return sample_input.dim() >= 4;
}

/**
 * @brief Compute the effective memory format.
 *
 * @return ChannelsLast if applicable, otherwise Contiguous.
 */
inline torch::MemoryFormat resolve_memory_format(
    torch::MemoryFormat requested_fmt,
    bool is_cuda,
    bool has_conv_layers,
    const torch::Tensor& sample_input)
{
    return resolve_channels_last(requested_fmt, is_cuda, has_conv_layers, sample_input)
        ? torch::MemoryFormat::ChannelsLast
        : torch::MemoryFormat::Contiguous;
}

} // namespace Nott::Training

#endif // Nott_TRAINING_PREFLIGHT_HPP
