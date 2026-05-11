#ifndef Nott_ADAM_HPP
#define Nott_ADAM_HPP
/// Adam: https://arxiv.org/pdf/1412.6980
/// AdamW: https://arxiv.org/pdf/1711.05101
/// Adan: https://arxiv.org/pdf/2208.06677


#include <tuple>
#include <functional>
#include <type_traits>
#include <torch/torch.h>
#include <algorithm>
#include <memory>
#include <vector>

namespace Nott::Optimizer::Details {

    struct AdamOptions {
        double learning_rate{1e-3};
        double beta1{0.9};
        double beta2{0.999};
        double eps{1e-8};
        double weight_decay{0.0};
        bool amsgrad{false};
    };

    struct AdamDescriptor {
        AdamOptions options{};
    };

    inline torch::optim::AdamOptions to_torch_options(const AdamOptions& options) {
        torch::optim::AdamOptions torch_options(options.learning_rate);
        torch_options = torch_options.betas(std::make_tuple(options.beta1, options.beta2));
        torch_options = torch_options.eps(options.eps);
        torch_options = torch_options.weight_decay(options.weight_decay);
        torch_options = torch_options.amsgrad(options.amsgrad);
        return torch_options;
    }

    struct AdamWOptions {
        double learning_rate{1e-3};
        double beta1{0.9};
        double beta2{0.999};
        double eps{1e-8};
        double weight_decay{1e-2};
        bool amsgrad{false};
    };

    struct AdamWDescriptor {
        AdamWOptions options{};
    };

    inline torch::optim::AdamWOptions to_torch_options(const AdamWOptions& options) {
        torch::optim::AdamWOptions torch_options(options.learning_rate);
        torch_options = torch_options.betas(std::make_tuple(options.beta1, options.beta2));
        torch_options = torch_options.eps(options.eps);
        torch_options = torch_options.weight_decay(options.weight_decay);
        torch_options = torch_options.amsgrad(options.amsgrad);
        return torch_options;
    }

    /// --- Adam wrapper ---
    class Adam : public torch::optim::Adam {
    public:
        /// 1) ctor for simple param list (vector<at::Tensor>)
        Adam(std::vector<at::Tensor> params,
             const torch::optim::AdamOptions& options,
             std::vector<std::vector<at::Tensor>> warmup_buckets = {})
            : torch::optim::Adam(std::move(params), options),
              warmup_buckets_(std::move(warmup_buckets)) {}

        /// 2) ctor for param groups (const lvalue ref)
        Adam(const std::vector<torch::optim::OptimizerParamGroup>& param_groups,
             const torch::optim::AdamOptions& options,
             std::vector<std::vector<at::Tensor>> warmup_buckets = {})
            : torch::optim::Adam(param_groups, options),
              warmup_buckets_(std::move(warmup_buckets)) {}

        /// 3) ctor for param groups (rvalue)
        Adam(std::vector<torch::optim::OptimizerParamGroup>&& param_groups,
             const torch::optim::AdamOptions& options,
             std::vector<std::vector<at::Tensor>> warmup_buckets = {})
            : torch::optim::Adam(std::move(param_groups), options),
              warmup_buckets_(std::move(warmup_buckets)) {}

        /// Generic forwarding ctor, only enabled if torch::optim::Adam is constructible.
        template <typename ParamsT,
                  typename = std::enable_if_t<
                      std::is_constructible_v<torch::optim::Adam, ParamsT, torch::optim::AdamOptions>
                  >>
        Adam(ParamsT&& params,
             const torch::optim::AdamOptions& options,
             std::vector<std::vector<at::Tensor>> warmup_buckets = {})
            : torch::optim::Adam(std::forward<ParamsT>(params), options),
              warmup_buckets_(std::move(warmup_buckets)) {}

        /// Initialize internal state (exp_avg, exp_avg_sq, max_exp_avg_sq) for all params.
        void ensure_state_initialized() {
            std::vector<std::vector<at::Tensor>> buckets = warmup_buckets_;
            if (buckets.empty()) {
                for (const auto& group : this->param_groups()) {
                    std::vector<at::Tensor> g;
                    for (const auto& p : group.params()) g.push_back(p);
                    if (!g.empty()) buckets.push_back(std::move(g));
                }
            }

            const bool any_amsgrad = std::any_of(
                this->param_groups().begin(),
                this->param_groups().end(),
                [](const auto& group) {
                    return static_cast<const torch::optim::AdamOptions&>(group.options()).amsgrad();
                });

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
                        auto state = std::make_unique<torch::optim::AdamParamState>();
                        state->step(0);
                        state->exp_avg(torch::zeros_like(param, torch::MemoryFormat::Preserve));
                        state->exp_avg_sq(torch::zeros_like(param, torch::MemoryFormat::Preserve));
                        if (any_amsgrad) {
                            state->max_exp_avg_sq(torch::zeros_like(param, torch::MemoryFormat::Preserve));
                        }
                        state_map.insert({key, std::move(state)});
                    } else {
                        auto& state = static_cast<torch::optim::AdamParamState&>(*it->second);
                        if (!state.exp_avg().defined()) {
                            state.exp_avg(torch::zeros_like(param, torch::MemoryFormat::Preserve));
                        }
                        if (!state.exp_avg_sq().defined()) {
                            state.exp_avg_sq(torch::zeros_like(param, torch::MemoryFormat::Preserve));
                        }
                        if (any_amsgrad && !state.max_exp_avg_sq().defined()) {
                            state.max_exp_avg_sq(torch::zeros_like(param, torch::MemoryFormat::Preserve));
                        }
                    }
                }
            }
        }

    private:
        std::vector<std::vector<at::Tensor>> warmup_buckets_;
    };

    using AdamOptimizer = Adam;

    /// --- AdamW wrapper ---
    class AdamW : public torch::optim::AdamW {
    public:
        /// 1) ctor for simple param list (vector<at::Tensor>)
        AdamW(std::vector<at::Tensor> params,
              const torch::optim::AdamWOptions& options,
              std::vector<std::vector<at::Tensor>> warmup_buckets = {})
            : torch::optim::AdamW(std::move(params), options),
              warmup_buckets_(std::move(warmup_buckets)) {}

        /// 2) ctor for param groups (const lvalue ref)
        AdamW(const std::vector<torch::optim::OptimizerParamGroup>& param_groups,
              const torch::optim::AdamWOptions& options,
              std::vector<std::vector<at::Tensor>> warmup_buckets = {})
            : torch::optim::AdamW(param_groups, options),
              warmup_buckets_(std::move(warmup_buckets)) {}

        /// 3) ctor for param groups (rvalue)
        AdamW(std::vector<torch::optim::OptimizerParamGroup>&& param_groups,
              const torch::optim::AdamWOptions& options,
              std::vector<std::vector<at::Tensor>> warmup_buckets = {})
            : torch::optim::AdamW(std::move(param_groups), options),
              warmup_buckets_(std::move(warmup_buckets)) {}

        /// Generic forwarding ctor, only enabled if torch::optim::AdamW is constructible.
        template <typename ParamsT,
                  typename = std::enable_if_t<
                      std::is_constructible_v<torch::optim::AdamW, ParamsT, torch::optim::AdamWOptions>
                  >>
        AdamW(ParamsT&& params,
              const torch::optim::AdamWOptions& options,
              std::vector<std::vector<at::Tensor>> warmup_buckets = {})
            : torch::optim::AdamW(std::forward<ParamsT>(params), options),
              warmup_buckets_(std::move(warmup_buckets)) {}

        /// Initialize internal state (exp_avg, exp_avg_sq, max_exp_avg_sq) for all params.
        void ensure_state_initialized() {
            std::vector<std::vector<at::Tensor>> buckets = warmup_buckets_;
            if (buckets.empty()) {
                for (const auto& group : this->param_groups()) {
                    std::vector<at::Tensor> g;
                    for (const auto& p : group.params()) g.push_back(p);
                    if (!g.empty()) buckets.push_back(std::move(g));
                }
            }

            const bool any_amsgrad = std::any_of(
                this->param_groups().begin(),
                this->param_groups().end(),
                [](const auto& group) {
                    return static_cast<const torch::optim::AdamWOptions&>(group.options()).amsgrad();
                });

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
                        auto state = std::make_unique<torch::optim::AdamWParamState>();
                        state->step(0);
                        state->exp_avg(torch::zeros_like(param, torch::MemoryFormat::Preserve));
                        state->exp_avg_sq(torch::zeros_like(param, torch::MemoryFormat::Preserve));
                        if (any_amsgrad) {
                            state->max_exp_avg_sq(torch::zeros_like(param, torch::MemoryFormat::Preserve));
                        }
                        state_map.insert({key, std::move(state)});
                    } else {
                        auto& state = static_cast<torch::optim::AdamWParamState&>(*it->second);
                        if (!state.exp_avg().defined()) {
                            state.exp_avg(torch::zeros_like(param, torch::MemoryFormat::Preserve));
                        }
                        if (!state.exp_avg_sq().defined()) {
                            state.exp_avg_sq(torch::zeros_like(param, torch::MemoryFormat::Preserve));
                        }
                        if (any_amsgrad && !state.max_exp_avg_sq().defined()) {
                            state.max_exp_avg_sq(torch::zeros_like(param, torch::MemoryFormat::Preserve));
                        }
                    }
                }
            }
        }

    private:
        std::vector<std::vector<at::Tensor>> warmup_buckets_;
    };

    using AdamWOptimizer = AdamW;

} // namespace Nott::Optimizer::Details

#endif // Nott_ADAM_HPP
