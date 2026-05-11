#ifndef Nott_LAMB_HPP
#define Nott_LAMB_HPP
/// "Large Batch Optimization for Deep Learning: Training BERT in 76 minutes" (LAMB) https://arxiv.org/pdf/1904.00962
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <tuple>

#include <torch/torch.h>
#include <torch/optim/optimizer.h>
#include <torch/optim/serialize.h>

namespace Nott::Optimizer::Details {

    struct LAMBOptions : public torch::optim::OptimizerCloneableOptions<LAMBOptions> {
        LAMBOptions(double lr = 1e-3) : lr_(lr) {}

        TORCH_ARG(double, lr) = 1e-3;
        TORCH_ARG(double, beta1) = 0.9;
        TORCH_ARG(double, beta2) = 0.999;
        TORCH_ARG(double, eps) = 1e-6;
        TORCH_ARG(double, weight_decay) = 0.0;
        TORCH_ARG(bool, adam) = false;

    public:
        [[nodiscard]] inline LAMBOptions validated() const {
            LAMBOptions copy = *this;
            copy.lr(std::max(0.0, copy.lr()));
            copy.beta1(std::clamp(copy.beta1(), 0.0, 1.0));
            copy.beta2(std::clamp(copy.beta2(), 0.0, 1.0));
            copy.eps(std::max(0.0, copy.eps()));
            copy.weight_decay(std::max(0.0, copy.weight_decay()));
            return copy;
        }

        void serialize(torch::serialize::InputArchive& archive) override {
            _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, lr);
            _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, beta1);
            _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, beta2);
            _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, eps);
            _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, weight_decay);
            _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(bool, adam);
        }

        void serialize(torch::serialize::OutputArchive& archive) const override {
            _TORCH_OPTIM_SERIALIZE_TORCH_ARG(lr);
            _TORCH_OPTIM_SERIALIZE_TORCH_ARG(beta1);
            _TORCH_OPTIM_SERIALIZE_TORCH_ARG(beta2);
            _TORCH_OPTIM_SERIALIZE_TORCH_ARG(eps);
            _TORCH_OPTIM_SERIALIZE_TORCH_ARG(weight_decay);
            _TORCH_OPTIM_SERIALIZE_TORCH_ARG(adam);
        }

        double get_lr() const override { return lr(); }
        void set_lr(double value) override { lr(value); }
    };

    struct LAMBParamState : public torch::optim::OptimizerCloneableParamState<LAMBParamState> {
        TORCH_ARG(torch::Tensor, exp_avg);
        TORCH_ARG(torch::Tensor, exp_avg_sq);
        TORCH_ARG(int64_t, step) = 0;

    public:
        void serialize(torch::serialize::InputArchive& archive) override {
            _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(torch::Tensor, exp_avg);
            _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(torch::Tensor, exp_avg_sq);
            _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(int64_t, step);
        }

        void serialize(torch::serialize::OutputArchive& archive) const override {
            _TORCH_OPTIM_SERIALIZE_TORCH_ARG(exp_avg);
            _TORCH_OPTIM_SERIALIZE_TORCH_ARG(exp_avg_sq);
            _TORCH_OPTIM_SERIALIZE_TORCH_ARG(step);
        }

        void reset() {
            exp_avg(torch::Tensor());
            exp_avg_sq(torch::Tensor());
            step(0);
        }
    };

    struct LAMBDescriptor {
        LAMBOptions options{};
    };

    class LAMB : public torch::optim::Optimizer {
    public:
        using Options = LAMBOptions;
        using ParamState = LAMBParamState;

        explicit LAMB(std::vector<torch::Tensor> params, Options options = {})
            : LAMB({torch::optim::OptimizerParamGroup(std::move(params))}, std::move(options)) {}

        explicit LAMB(std::vector<torch::optim::OptimizerParamGroup> param_groups, Options options = {})
            : torch::optim::Optimizer(std::move(param_groups), std::make_unique<Options>(options.validated())) {}

        torch::Tensor step(LossClosure closure = nullptr) override {
            torch::NoGradGuard no_grad;
            torch::Tensor loss;
            if (closure != nullptr) {
                torch::AutoGradMode enable_grad(true);
                loss = closure();
            }

            for (auto& group : this->param_groups_) {
                auto& raw_options = static_cast<Options&>(group.options());
                auto options = raw_options.validated();
                raw_options = options;

                for (auto& param : group.params()) {
                    auto grad = param.grad();
                    if (!grad.defined()) {
                        continue;
                    }
                    TORCH_CHECK(!grad.is_sparse(), "LAMB does not support sparse gradients.");

                    auto state_it = this->state_.find(param.unsafeGetTensorImpl());
                    if (state_it == this->state_.end()) {
                        auto state = std::make_unique<ParamState>();
                        state->reset();
                        state_it = this->state_.insert({param.unsafeGetTensorImpl(), std::move(state)}).first;
                    }

                    auto& state = static_cast<ParamState&>(*state_it->second);
                    auto& exp_avg = state.exp_avg();
                    auto& exp_avg_sq = state.exp_avg_sq();
                    if (!exp_avg.defined()) {
                        exp_avg = torch::zeros_like(param);
                    } else if (exp_avg.device() != param.device() || exp_avg.dtype() != param.dtype()) {
                        exp_avg = exp_avg.to(param.options());
                    }
                    if (!exp_avg_sq.defined()) {
                        exp_avg_sq = torch::zeros_like(param);
                    } else if (exp_avg_sq.device() != param.device() || exp_avg_sq.dtype() != param.dtype()) {
                        exp_avg_sq = exp_avg_sq.to(param.options());
                    }

                    state.step(state.step() + 1);

                    exp_avg.mul_(options.beta1()).add_(grad, 1.0 - options.beta1());
                    exp_avg_sq.mul_(options.beta2()).addcmul_(grad, grad, 1.0 - options.beta2());

                    const auto bias_correction1 = 1.0 - std::pow(options.beta1(), static_cast<double>(state.step()));
                    const auto bias_correction2 = 1.0 - std::pow(options.beta2(), static_cast<double>(state.step()));

                    auto m_hat = exp_avg / bias_correction1;
                    auto v_hat = exp_avg_sq / bias_correction2;
                    auto denom = v_hat.sqrt().add_(options.eps());
                    auto update = m_hat / denom;

                    if (options.weight_decay() != 0.0) {
                        update.add_(param, options.weight_decay());
                    }

                    double trust_ratio = 1.0;
                    if (!options.adam()) {
                        const auto w_norm = param.norm().item<double>();
                        const auto u_norm = update.norm().item<double>();
                        if (w_norm > 0.0 && u_norm > 0.0) {
                            trust_ratio = w_norm / u_norm;
                        }
                    }

                    param.add_(update, -options.lr() * trust_ratio);
                }
            }

            return loss;
        }

        void save(torch::serialize::OutputArchive& archive) const override {
            torch::optim::serialize<ParamState, Options>(archive, *this);
        }

        void load(torch::serialize::InputArchive& archive) override {
            torch::optim::serialize<ParamState, Options>(archive, *this);
        }

        void ensure_state_initialized() {
            for (auto& group : this->param_groups_) {
                for (auto& param : group.params()) {
                    auto state_it = this->state_.find(param.unsafeGetTensorImpl());
                    if (state_it == this->state_.end()) {
                        auto state = std::make_unique<ParamState>();
                        state->reset();
                        state_it = this->state_.insert({param.unsafeGetTensorImpl(), std::move(state)}).first;
                    }
                    auto& state = static_cast<ParamState&>(*state_it->second);
                    if (!state.exp_avg().defined()) {
                        state.exp_avg(torch::zeros_like(param));
                    }
                    if (!state.exp_avg_sq().defined()) {
                        state.exp_avg_sq(torch::zeros_like(param));
                    }
                }
            }
        }
    };

}

#endif // Nott_LAMB_HPP
