#ifndef Nott_SGD_HPP
#define Nott_SGD_HPP

#include <stdexcept>
#include <torch/torch.h>
#include <functional>
#include <type_traits>
#include <vector>
#include <algorithm>
#include <memory>

namespace Nott::Optimizer::Details {

    struct SGDOptions {
        double learning_rate{1e-2};
        double momentum{0.0};
        double dampening{0.0};
        double weight_decay{0.0};
        bool nesterov{false};
        bool maximize{false};
    };

    struct SGDDescriptor {
        SGDOptions options{};
    };

    inline torch::optim::SGDOptions to_torch_options(const SGDOptions& options) {
        torch::optim::SGDOptions torch_options(options.learning_rate);
        torch_options = torch_options.momentum(options.momentum);
        torch_options = torch_options.dampening(options.dampening);
        torch_options = torch_options.weight_decay(options.weight_decay);
        torch_options = torch_options.nesterov(options.nesterov);
        if (options.maximize)
            throw std::invalid_argument("SGD maximize option is not supported by the configured libtorch version.");
        return torch_options;
    }

    /// Thin wrapper around torch::optim::SGD so the factory can be tiny.
    /// Provides explicit overloads for vector<at::Tensor> and param_groups and a
    /// guarded forwarding ctor so templated factories will resolve correctly.
    class SGD : public torch::optim::SGD {
    public:
        /// ctor for simple param list (vector<at::Tensor>)
        SGD(std::vector<at::Tensor> params,
            const torch::optim::SGDOptions& options,
            std::vector<std::vector<at::Tensor>> warmup_buckets = {})
            : torch::optim::SGD(std::move(params), options),
              warmup_buckets_(std::move(warmup_buckets)) {}

        /// ctor for param groups (const lvalue ref)
        SGD(const std::vector<torch::optim::OptimizerParamGroup>& param_groups,
            const torch::optim::SGDOptions& options,
            std::vector<std::vector<at::Tensor>> warmup_buckets = {})
            : torch::optim::SGD(param_groups, options),
              warmup_buckets_(std::move(warmup_buckets)) {}

        /// ctor for param groups (rvalue)
        SGD(std::vector<torch::optim::OptimizerParamGroup>&& param_groups,
            const torch::optim::SGDOptions& options,
            std::vector<std::vector<at::Tensor>> warmup_buckets = {})
            : torch::optim::SGD(std::move(param_groups), options),
              warmup_buckets_(std::move(warmup_buckets)) {}

        /// Guarded forwarding ctor: participates only if torch::optim::SGD is constructible
        template <typename ParamsT,
                  typename = std::enable_if_t<std::is_constructible_v<torch::optim::SGD, ParamsT, torch::optim::SGDOptions>>>
        SGD(ParamsT&& params,
            const torch::optim::SGDOptions& options,
            std::vector<std::vector<at::Tensor>> warmup_buckets = {})
            : torch::optim::SGD(std::forward<ParamsT>(params), options),
              warmup_buckets_(std::move(warmup_buckets)) {}

        /// Initialize momentum buffers (safe to call repeatedly)
        void ensure_state_initialized() {
            /// quick check: if no param uses momentum, nothing to do
            const bool momentum_enabled = std::any_of(
                this->param_groups().begin(),
                this->param_groups().end(),
                [](const auto& group) {
                    return static_cast<const torch::optim::SGDOptions&>(group.options()).momentum() != 0.0;
                });
            if (!momentum_enabled) return;

            /// build buckets (use provided warmup buckets if present)
            std::vector<std::vector<at::Tensor>> buckets = warmup_buckets_;
            if (buckets.empty()) {
                for (const auto& group : this->param_groups()) {
                    std::vector<at::Tensor> g;
                    for (const auto& p : group.params()) g.push_back(p);
                    if (!g.empty()) buckets.push_back(std::move(g));
                }
            }

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
                        auto state = std::make_unique<torch::optim::SGDParamState>();
                        state->momentum_buffer(torch::zeros_like(param, torch::MemoryFormat::Preserve));
                        state_map.insert({key, std::move(state)});
                    } else {
                        auto& state = static_cast<torch::optim::SGDParamState&>(*it->second);
                        if (!state.momentum_buffer().defined()) {
                            state.momentum_buffer(torch::zeros_like(param, torch::MemoryFormat::Preserve));
                        } else if (state.momentum_buffer().device() != param.device() ||
                                   state.momentum_buffer().sizes() != param.sizes()) {
                            /// be defensive: ensure buffer is on same device / compatible shape
                            state.momentum_buffer(state.momentum_buffer().to(param.options()));
                            if (state.momentum_buffer().sizes() != param.sizes()) {
                                /// replace with zeros_like if shapes don't match (rare)
                                state.momentum_buffer(torch::zeros_like(param, torch::MemoryFormat::Preserve));
                            }
                        }
                    }
                }
            }
        }

    private:
        std::vector<std::vector<at::Tensor>> warmup_buckets_;
    };

    using SGDOptimizer = SGD;

} // namespace Nott::Optimizer::Details

#endif //Nott_SGD_HPP
