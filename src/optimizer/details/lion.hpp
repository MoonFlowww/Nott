#ifndef Nott_LION_HPP
#define Nott_LION_HPP
/// "Symbolic Discovery of Optimization Algorithms" (Lion) https://arxiv.org/pdf/2302.06675
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <tuple>

#include <torch/torch.h>
#include <torch/optim/optimizer.h>
#include <torch/optim/serialize.h>

namespace Nott::Optimizer::Details {

    struct LionOptions : public torch::optim::OptimizerCloneableOptions<LionOptions> {
        LionOptions(double lr = 1e-4) : lr_(lr) {}

        TORCH_ARG(double, lr) = 1e-4;
        TORCH_ARG(double, beta1) = 0.9;
        TORCH_ARG(double, beta2) = 0.99;
        TORCH_ARG(double, weight_decay) = 0.0;

    public:
        [[nodiscard]] inline LionOptions validated() const {
            LionOptions copy = *this;
            copy.lr(std::max(0.0, copy.lr()));
            copy.beta1(std::clamp(copy.beta1(), 0.0, 1.0));
            copy.beta2(std::clamp(copy.beta2(), 0.0, 1.0));
            copy.weight_decay(std::max(0.0, copy.weight_decay()));
            return copy;
        }

        void serialize(torch::serialize::InputArchive& archive) override {
            _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, lr);
            _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, beta1);
            _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, beta2);
            _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, weight_decay);
        }

        void serialize(torch::serialize::OutputArchive& archive) const override {
            _TORCH_OPTIM_SERIALIZE_TORCH_ARG(lr);
            _TORCH_OPTIM_SERIALIZE_TORCH_ARG(beta1);
            _TORCH_OPTIM_SERIALIZE_TORCH_ARG(beta2);
            _TORCH_OPTIM_SERIALIZE_TORCH_ARG(weight_decay);
        }

        double get_lr() const override { return lr(); }
        void set_lr(double value) override { lr(value); }
    };

    struct LionParamState : public torch::optim::OptimizerCloneableParamState<LionParamState> {
        TORCH_ARG(torch::Tensor, exp_avg);
        TORCH_ARG(int64_t, step) = 0;

    public:
        void serialize(torch::serialize::InputArchive& archive) override {
            _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(torch::Tensor, exp_avg);
            _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(int64_t, step);
        }

        void serialize(torch::serialize::OutputArchive& archive) const override {
            _TORCH_OPTIM_SERIALIZE_TORCH_ARG(exp_avg);
            _TORCH_OPTIM_SERIALIZE_TORCH_ARG(step);
        }

        void reset() {
            exp_avg(torch::Tensor());
            step(0);
        }
    };

    struct LionDescriptor {
        LionOptions options{};
    };

    class Lion : public torch::optim::Optimizer {
    public:
        using Options = LionOptions;
        using ParamState = LionParamState;

        explicit Lion(std::vector<torch::Tensor> params, Options options = {})
            : Lion({torch::optim::OptimizerParamGroup(std::move(params))}, std::move(options)) {}

        explicit Lion(std::vector<torch::optim::OptimizerParamGroup> param_groups, Options options = {})
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
                    TORCH_CHECK(!grad.is_sparse(), "Lion does not support sparse gradients.");

                    auto state_it = this->state_.find(param.unsafeGetTensorImpl());
                    if (state_it == this->state_.end()) {
                        auto state = std::make_unique<ParamState>();
                        state->reset();
                        state_it = this->state_.insert({param.unsafeGetTensorImpl(), std::move(state)}).first;
                    }

                    auto& state = static_cast<ParamState&>(*state_it->second);
                    auto& exp_avg = state.exp_avg();
                    if (!exp_avg.defined()) {
                        exp_avg = torch::zeros_like(param);
                    } else if (exp_avg.device() != param.device() || exp_avg.dtype() != param.dtype()) {
                        exp_avg = exp_avg.to(param.options());
                    }

                    state.step(state.step() + 1);

                    if (options.weight_decay() != 0.0) {
                        param.mul_(1.0 - options.lr() * options.weight_decay());
                    }

                    auto update = exp_avg.mul(options.beta1()).add(grad, 1.0 - options.beta1()).sign();
                    param.add_(update, -options.lr());

                    exp_avg.mul_(options.beta2()).add_(grad, 1.0 - options.beta2());
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
                }
            }
        }
    };

}

#endif // Nott_LION_HPP