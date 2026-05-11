#ifndef Nott_STEP_EXECUTOR_HPP
#define Nott_STEP_EXECUTOR_HPP
/**
 * @file step_executor.hpp
 * @brief Per-batch training step dispatch.
 *
 * graph_train_step_impl() was a 230-line switch statement with an inner
 * lambda `run_training_step`.  We extract the inner step body into a
 * free-standing helper and keep only the graph-mode dispatch in the
 * actual Model method.
 *
 * The key simplification: the inner lambda's logic (forward -> loss ->
 * regularisation -> backward -> step -> zero_grad) is now a single
 * template function parameterised only by GraphMode.  The CUDA-graph
 * capture/replay scaffolding stays in Model where it has access to graph
 * state.
 */

#include <torch/torch.h>
// GraphMode fwd-declared via training_policy.hpp  // GraphMode

#ifdef TORCH_CUDA_AVAILABLE
#include <torch/cuda.h>
#include <torch/cuda/amp.h>
#endif

namespace Nott::Training {

/**
 * @brief Core step body: forward + loss + backward + step.
 *
 * Pure computation -- no graph-mode awareness beyond @p mode for
 * regularisation shape checks.
 *
 * @tparam ModelT       Has .forward(), .compute_loss(),
 *                      .compute_regularization_penalty(), .zero_grad().
 * @tparam StepOptimFn  void(ModelT&) or void(ModelT&, GradScaler&).
 * @tparam BackwardFn   void(Tensor& loss, bool retain_graph).
 * @param model                  The model being trained.
 * @param inputs                 Batch inputs.
 * @param targets                Batch targets.
 * @param mode                   Graph execution mode.
 * @param regularization_active  Whether to add penalty.
 * @param use_amp                Whether AMP is active.
 * @param step_optim             Optimizer step callable.
 * @param do_backward            Backward pass callable.
 * @return Detached loss tensor.
 */
template<typename ModelT, typename StepOptimFn, typename BackwardFn>
torch::Tensor execute_training_step(
    ModelT&           model,
    torch::Tensor     inputs,
    torch::Tensor     targets,
    GraphMode         mode,
    bool              regularization_active,
    bool              use_amp,
    StepOptimFn&&     step_optim,
    BackwardFn&&      do_backward)
{
    /* forward + loss */
    torch::Tensor prediction;
    torch::Tensor loss;
    {
        prediction = model.forward(std::move(inputs));

        if (!prediction.sizes().equals(targets.sizes())) {
            if (targets.numel() == prediction.numel())
                targets = targets.reshape_as(prediction);
        }

        loss = model.compute_loss(prediction, targets);
        if (loss.dim() != 0) loss = loss.mean();
    }

    /* regularisation */
    if (regularization_active) {
        auto penalty = model.compute_regularization_penalty(mode);
        if (penalty.defined()) {
            if (mode == GraphMode::Disabled) {
                if (penalty.device() != loss.device())
                    penalty = penalty.to(loss.device());
                if (penalty.scalar_type() != loss.scalar_type())
                    penalty = penalty.to(loss.scalar_type());
            } else {
                /* Graph mode: shapes must be static; throw on mismatch */
                if (penalty.device() != loss.device())
                    throw std::runtime_error(
                        "Regularisation penalty device changed during CUDA graph execution.");
                if (penalty.scalar_type() != loss.scalar_type())
                    throw std::runtime_error(
                        "Regularisation penalty dtype changed during CUDA graph execution.");
            }
            loss = loss + penalty;
        }
    }

    /* backward + step */
    const bool retain = (mode != GraphMode::Disabled);
    do_backward(loss, retain);
    step_optim(model);
    model.zero_grad();

    loss.detach_();
    return loss;
}

/**
 * @brief Create an AMP-aware backward wrapper.
 *
 * Returns a callable that handles amp_scaler.scale() -> backward -> step
 * or plain backward -> step depending on @p use_amp.
 *
 * @tparam ModelT  Model type.
 * @param model        The model (used for capture only).
 * @param use_amp      Whether AMP is active.
 * @param amp_scaler   Pointer to GradScaler (may be null if !use_amp).
 * @return Callable void(Tensor& loss, bool retain_graph).
 */
#ifdef TORCH_CUDA_AVAILABLE
template<typename ModelT>
auto make_amp_backward(
    ModelT&                               model,
    bool                                  use_amp,
    torch::cuda::amp::GradScaler*         amp_scaler)
{
    return [&model, use_amp, amp_scaler](torch::Tensor& loss, bool retain) {
        if (use_amp && amp_scaler) {
            auto scaled = amp_scaler->scale(loss);
            scaled.backward({}, retain);
            /* step_optim handles scaler.step() -- caller responsibility */
        } else {
            loss.backward({}, retain);
        }
    };
}
#else
template<typename ModelT>
auto make_amp_backward(ModelT&, bool, void*) {
    return [](torch::Tensor& loss, bool retain) {
        loss.backward({}, retain);
    };
}
#endif

} // namespace Nott::Training

#endif // Nott_STEP_EXECUTOR_HPP
