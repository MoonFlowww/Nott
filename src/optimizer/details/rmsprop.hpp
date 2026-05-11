#ifndef Nott_RMSPROP_HPP
#define Nott_RMSPROP_HPP

#include <torch/torch.h>
#include <vector>
#include <algorithm>
#include <memory>

namespace Nott::Optimizer::Details {

    struct RMSpropOptions {
        double learning_rate{1e-2};
        double alpha{0.99};
        double eps{1e-8};
        double weight_decay{0.0};
        double momentum{0.0};
        bool centered{false};
    };

    struct RMSpropDescriptor {
        RMSpropOptions options{};
    };

    inline torch::optim::RMSpropOptions to_torch_options(const RMSpropOptions& options) {
        torch::optim::RMSpropOptions torch_options(options.learning_rate);
        torch_options = torch_options.alpha(options.alpha);
        torch_options = torch_options.eps(options.eps);
        torch_options = torch_options.weight_decay(options.weight_decay);
        torch_options = torch_options.momentum(options.momentum);
        torch_options = torch_options.centered(options.centered);
        return torch_options;
    }

    /// Thin wrapper around torch::optim::RMSprop:
    /// - explicit ctors for vector<at::Tensor> and param_groups (lvalue + rvalue)
    /// - guarded forwarding ctor usable from templated factories
    /// - ensure_state_initialized() that creates square_avg, momentum_buffer, grad_avg
    class RMSProp : public torch::optim::RMSprop {
    public:
        /// 1) ctor for simple param list (vector<at::Tensor>)
        RMSProp(std::vector<at::Tensor> params,
                const torch::optim::RMSpropOptions& options,
                std::vector<std::vector<at::Tensor>> warmup_buckets = {})
            : torch::optim::RMSprop(std::move(params), options),
              warmup_buckets_(std::move(warmup_buckets)) {}

        /// 2) ctor for param groups (const lvalue ref)
        RMSProp(const std::vector<torch::optim::OptimizerParamGroup>& param_groups,
                const torch::optim::RMSpropOptions& options,
                std::vector<std::vector<at::Tensor>> warmup_buckets = {})
            : torch::optim::RMSprop(param_groups, options),
              warmup_buckets_(std::move(warmup_buckets)) {}

        /// 3) ctor for param groups (rvalue)
        RMSProp(std::vector<torch::optim::OptimizerParamGroup>&& param_groups,
                const torch::optim::RMSpropOptions& options,
                std::vector<std::vector<at::Tensor>> warmup_buckets = {})
            : torch::optim::RMSprop(std::move(param_groups), options),
              warmup_buckets_(std::move(warmup_buckets)) {}

        /// Generic forwarding ctor, enabled only if torch::optim::RMSprop is constructible.
        template <typename ParamsT,
                  typename = std::enable_if_t<
                      std::is_constructible_v<torch::optim::RMSprop, ParamsT, torch::optim::RMSpropOptions>
                  >>
        RMSProp(ParamsT&& params,
                const torch::optim::RMSpropOptions& options,
                std::vector<std::vector<at::Tensor>> warmup_buckets = {})
            : torch::optim::RMSprop(std::forward<ParamsT>(params), options),
              warmup_buckets_(std::move(warmup_buckets)) {}

        /// Idempotent initialization of per-param state (square_avg, momentum_buffer, grad_avg)
        void ensure_state_initialized() {
            /// build buckets (use provided warmup buckets if present)
            std::vector<std::vector<at::Tensor>> buckets = warmup_buckets_;
            if (buckets.empty()) {
                for (const auto& group : this->param_groups()) {
                    std::vector<at::Tensor> g;
                    for (const auto& p : group.params()) g.push_back(p);
                    if (!g.empty()) buckets.push_back(std::move(g));
                }
            }

            /// check whether momentum or centered terms are needed
            const bool needs_momentum = std::any_of(
                this->param_groups().begin(),
                this->param_groups().end(),
                [](const auto& group) {
                    return static_cast<const torch::optim::RMSpropOptions&>(group.options()).momentum() != 0.0;
                });
            const bool any_centered = std::any_of(
                this->param_groups().begin(),
                this->param_groups().end(),
                [](const auto& group) {
                    return static_cast<const torch::optim::RMSpropOptions&>(group.options()).centered();
                });

            /// quick exit if nothing to allocate? keep allocation for square_avg always,
            /// because RMSprop requires it even without momentum/centered.
            torch::NoGradGuard no_grad{};
            at::OptionalDeviceGuard device_guard;
            auto& state_map = this->state();

            for (const auto& bucket : buckets) {
                for (const auto& param : bucket) {
                    if (!param.defined() || !param.requires_grad()) continue;

                    device_guard.reset_device(param.device());

                    auto* key = param.unsafeGetTensorImpl();
                    auto it = state_map.find(key);
                    if (it == state_map.end()) {
                        /// create a new param state
                        auto state = std::make_unique<torch::optim::RMSpropParamState>();
                        /// square_avg always present
                        state->square_avg(torch::zeros_like(param, torch::MemoryFormat::Preserve));
                        /// momentum buffer if needed
                        if (needs_momentum) {
                            state->momentum_buffer(torch::zeros_like(param, torch::MemoryFormat::Preserve));
                        }
                        /// grad_avg if centered
                        if (any_centered) {
                            state->grad_avg(torch::zeros_like(param, torch::MemoryFormat::Preserve));
                        }
                        state_map.insert({key, std::move(state)});
                    } else {
                        auto& state = static_cast<torch::optim::RMSpropParamState&>(*it->second);
                        /// ensure square_avg exists and matches device/shape
                        if (!state.square_avg().defined()) {
                            state.square_avg(torch::zeros_like(param, torch::MemoryFormat::Preserve));
                        } else if (state.square_avg().device() != param.device() || state.square_avg().sizes() != param.sizes()) {
                            auto candidate = state.square_avg().to(param.options());
                            if (candidate.sizes() != param.sizes()) candidate = torch::zeros_like(param, torch::MemoryFormat::Preserve);
                            state.square_avg(candidate);
                        }

                        if (needs_momentum) {
                            if (!state.momentum_buffer().defined()) {
                                state.momentum_buffer(torch::zeros_like(param, torch::MemoryFormat::Preserve));
                            } else if (state.momentum_buffer().device() != param.device() || state.momentum_buffer().sizes() != param.sizes()) {
                                auto candidate = state.momentum_buffer().to(param.options());
                                if (candidate.sizes() != param.sizes()) candidate = torch::zeros_like(param, torch::MemoryFormat::Preserve);
                                state.momentum_buffer(candidate);
                            }
                        }

                        if (any_centered) {
                            if (!state.grad_avg().defined()) {
                                state.grad_avg(torch::zeros_like(param, torch::MemoryFormat::Preserve));
                            } else if (state.grad_avg().device() != param.device() || state.grad_avg().sizes() != param.sizes()) {
                                auto candidate = state.grad_avg().to(param.options());
                                if (candidate.sizes() != param.sizes()) candidate = torch::zeros_like(param, torch::MemoryFormat::Preserve);
                                state.grad_avg(candidate);
                            }
                        }
                    }
                }
            }
        }

    private:
        std::vector<std::vector<at::Tensor>> warmup_buckets_;
    };

    using RMSPropOptimizer = RMSProp;

}

#endif // Nott_RMSPROP_HPP
