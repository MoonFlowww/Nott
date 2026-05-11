#ifndef Nott_ADAGRAD_HPP
#define Nott_ADAGRAD_HPP



#include <torch/torch.h>
#include <vector>
#include <algorithm>
#include <memory>

namespace Nott::Optimizer::Details {

    struct AdagradOptions {
        double learning_rate{1e-2};
        double lr_decay{0.0};
        double weight_decay{0.0};
        double initial_accumulator_value{0.0};
        double eps{1e-10};
    };

    struct AdagradDescriptor {
        AdagradOptions options{};
    };

    inline torch::optim::AdagradOptions to_torch_options(const AdagradOptions& options) {
        torch::optim::AdagradOptions torch_options(options.learning_rate);
        torch_options = torch_options.lr_decay(options.lr_decay);
        torch_options = torch_options.weight_decay(options.weight_decay);
        torch_options = torch_options.initial_accumulator_value(options.initial_accumulator_value);
        torch_options = torch_options.eps(options.eps);
        return torch_options;
    }

    /// Adagrad wrapper: explicit ctors + guarded forwarding ctor + warmup helper
    class Adagrad : public torch::optim::Adagrad {
    public:
        /// 1) ctor for simple param list (vector<at::Tensor>)
        Adagrad(std::vector<at::Tensor> params,
                const torch::optim::AdagradOptions& options,
                std::vector<std::vector<at::Tensor>> warmup_buckets = {})
            : torch::optim::Adagrad(std::move(params), options),
              warmup_buckets_(std::move(warmup_buckets)) {}

        /// 2) ctor for param groups (const lvalue ref)
        Adagrad(const std::vector<torch::optim::OptimizerParamGroup>& param_groups,
                const torch::optim::AdagradOptions& options,
                std::vector<std::vector<at::Tensor>> warmup_buckets = {})
            : torch::optim::Adagrad(param_groups, options),
              warmup_buckets_(std::move(warmup_buckets)) {}

        /// 3) ctor for param groups (rvalue)
        Adagrad(std::vector<torch::optim::OptimizerParamGroup>&& param_groups,
                const torch::optim::AdagradOptions& options,
                std::vector<std::vector<at::Tensor>> warmup_buckets = {})
            : torch::optim::Adagrad(std::move(param_groups), options),
              warmup_buckets_(std::move(warmup_buckets)) {}

        /// Generic forwarding ctor, enabled only if torch::optim::Adagrad is constructible.
        template <typename ParamsT,
                  typename = std::enable_if_t<
                      std::is_constructible_v<torch::optim::Adagrad, ParamsT, torch::optim::AdagradOptions>
                  >>
        Adagrad(ParamsT&& params,
                const torch::optim::AdagradOptions& options,
                std::vector<std::vector<at::Tensor>> warmup_buckets = {})
            : torch::optim::Adagrad(std::forward<ParamsT>(params), options),
              warmup_buckets_(std::move(warmup_buckets)) {}

        /// Ensure per-parameter accumulator ("sum") exists and sits on correct device/shape.
        /// This mirrors the warmup behaviour you used inline in the factory.
        void ensure_state_initialized() {
            std::vector<std::vector<at::Tensor>> buckets = warmup_buckets_;
            if (buckets.empty()) {
                for (const auto& group : this->param_groups()) {
                    std::vector<at::Tensor> g;
                    for (const auto& p : group.params()) g.push_back(p);
                    if (!g.empty()) buckets.push_back(std::move(g));
                }
            }

            /// Helper: find the initial_accumulator_value from the param's group (fallback 0.0)
            auto initial_for_param = [this](const at::Tensor& param) -> double {
                for (const auto& group : this->param_groups()) {
                    for (const auto& p : group.params()) {
                        if (p.unsafeGetTensorImpl() == param.unsafeGetTensorImpl()) {
                            const auto& opts = static_cast<const torch::optim::AdagradOptions&>(group.options());
                            return opts.initial_accumulator_value();
                        }
                    }
                }
                return 0.0;
            };

            torch::NoGradGuard no_grad{};
            at::OptionalDeviceGuard device_guard;
            auto& state_map = this->state();

            for (const auto& bucket : buckets) {
                for (const auto& param : bucket) {
                    if (!param.defined() || !param.requires_grad()) continue;

                    device_guard.reset_device(param.device());

                    auto* key = param.unsafeGetTensorImpl();
                    auto it = state_map.find(key);
                    const double init_val = initial_for_param(param);

                    if (it == state_map.end()) {
                        auto state = std::make_unique<torch::optim::AdagradParamState>();
                        /// create sum initialized to initial_accumulator_value (or zeros)
                        if (init_val == 0.0) {
                            state->sum(torch::zeros_like(param, torch::MemoryFormat::Preserve));
                        } else {
                            state->sum(torch::full(param.sizes(), init_val, param.options().memory_format(torch::MemoryFormat::Preserve)));
                        }
                        state_map.insert({key, std::move(state)});
                    } else {
                        auto& state = static_cast<torch::optim::AdagradParamState&>(*it->second);
                        if (!state.sum().defined()) {
                            if (init_val == 0.0) {
                                state.sum(torch::zeros_like(param, torch::MemoryFormat::Preserve));
                            } else {
                                state.sum(torch::full(param.sizes(), init_val, param.options().memory_format(torch::MemoryFormat::Preserve)));
                            }
                        } else if (state.sum().device() != param.device() || state.sum().sizes() != param.sizes()) {
                            /// move/resize defensively
                            auto sum_candidate = state.sum().to(param.options());
                            if (sum_candidate.sizes() != param.sizes()) {
                                sum_candidate = torch::zeros_like(param, torch::MemoryFormat::Preserve);
                            }
                            state.sum(sum_candidate);
                        }
                    }
                }
            }
        }

    private:
        std::vector<std::vector<at::Tensor>> warmup_buckets_;
    };

    using AdagradOptimizer = Adagrad;

}

#endif // Nott_ADAGRAD_HPP
