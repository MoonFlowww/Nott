#ifndef Nott_SOPHIA_HPP
#define Nott_SOPHIA_HPP
/// Sophia G and H
/// https://arxiv.org/pdf/2305.14342
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>

#include <torch/csrc/autograd/variable.h>
#include <torch/optim/optimizer.h>
#include <torch/optim/serialize.h>
#include <torch/utils.h>

namespace Nott::Optimizer::Details {
    struct SophiaGOptions : public torch::optim::OptimizerCloneableOptions<SophiaGOptions> {
        SophiaGOptions(double lr = 1e-4) : lr_(lr) {}

        TORCH_ARG(double, lr) = 1e-4;
        TORCH_ARG(double, beta1) = 0.965;
        TORCH_ARG(double, beta2) = 0.99;
        TORCH_ARG(double, rho) = 0.04;
        TORCH_ARG(double, eps) = 1e-8;
        TORCH_ARG(double, weight_decay) = 0.0;
        TORCH_ARG(double, clip) = 1.0;
        TORCH_ARG(int64_t, hessian_update_interval) = 1;

    public:

        void serialize(torch::serialize::InputArchive& archive) override {
            _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, lr);
            _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, beta1);
            _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, beta2);
            _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, rho);
            _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, eps);
            _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, weight_decay);
            _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, clip);
            _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(int64_t, hessian_update_interval);
        }

        void serialize(torch::serialize::OutputArchive& archive) const override {
            _TORCH_OPTIM_SERIALIZE_TORCH_ARG(lr);
            _TORCH_OPTIM_SERIALIZE_TORCH_ARG(beta1);
            _TORCH_OPTIM_SERIALIZE_TORCH_ARG(beta2);
            _TORCH_OPTIM_SERIALIZE_TORCH_ARG(rho);
            _TORCH_OPTIM_SERIALIZE_TORCH_ARG(eps);
            _TORCH_OPTIM_SERIALIZE_TORCH_ARG(weight_decay);
            _TORCH_OPTIM_SERIALIZE_TORCH_ARG(clip);
            _TORCH_OPTIM_SERIALIZE_TORCH_ARG(hessian_update_interval);
        }

        double get_lr() const override { return lr(); }
        void set_lr(const double value) override { lr(value); }
    };

    struct SophiaHOptions : public torch::optim::OptimizerCloneableOptions<SophiaHOptions> {
        SophiaHOptions(double lr = 1e-4) : lr_(lr) {}

        TORCH_ARG(double, lr) = 1e-4;
        TORCH_ARG(double, beta1) = 0.965;
        TORCH_ARG(double, beta2) = 0.99;
        TORCH_ARG(double, rho) = 0.04;
        TORCH_ARG(double, eps) = 1e-8;
        TORCH_ARG(double, weight_decay) = 0.0;
        TORCH_ARG(double, clip) = 1.0;
        TORCH_ARG(int64_t, hessian_update_interval) = 1;

    public:

        void serialize(torch::serialize::InputArchive& archive) override {
            _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, lr);
            _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, beta1);
            _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, beta2);
            _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, rho);
            _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, eps);
            _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, weight_decay);
            _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, clip);
            _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(int64_t, hessian_update_interval);
        }

        void serialize(torch::serialize::OutputArchive& archive) const override {
            _TORCH_OPTIM_SERIALIZE_TORCH_ARG(lr);
            _TORCH_OPTIM_SERIALIZE_TORCH_ARG(beta1);
            _TORCH_OPTIM_SERIALIZE_TORCH_ARG(beta2);
            _TORCH_OPTIM_SERIALIZE_TORCH_ARG(rho);
            _TORCH_OPTIM_SERIALIZE_TORCH_ARG(eps);
            _TORCH_OPTIM_SERIALIZE_TORCH_ARG(weight_decay);
            _TORCH_OPTIM_SERIALIZE_TORCH_ARG(clip);
            _TORCH_OPTIM_SERIALIZE_TORCH_ARG(hessian_update_interval);
        }

        double get_lr() const override { return lr(); }
        void set_lr(const double value) override { lr(value); }
    };

    struct SophiaGDescriptor {
        SophiaGOptions options{};
    };

    struct SophiaHDescriptor {
        SophiaHOptions options{};
    };

    struct SophiaParamState : public torch::optim::OptimizerCloneableParamState<SophiaParamState> {
        TORCH_ARG(torch::Tensor, step);
        TORCH_ARG(torch::Tensor, momentum);
        TORCH_ARG(torch::Tensor, hessian);

    public:
        void serialize(torch::serialize::InputArchive& archive) override {
            _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(torch::Tensor, step);
            _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(torch::Tensor, momentum);
            _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(torch::Tensor, hessian);
        }

        void serialize(torch::serialize::OutputArchive& archive) const override {
            _TORCH_OPTIM_SERIALIZE_TORCH_ARG(step);
            _TORCH_OPTIM_SERIALIZE_TORCH_ARG(momentum);
            _TORCH_OPTIM_SERIALIZE_TORCH_ARG(hessian);
        }
    };

    namespace detail {
        inline torch::Tensor make_scalar_like(const torch::Tensor& reference, double value)
        {
            TORCH_CHECK(reference.defined(), "make_scalar_like requires a defined reference tensor.");
            return torch::full({}, value, reference.options());
        }

        inline torch::Tensor ensure_step_tensor(torch::Tensor& storage, const torch::Tensor& reference)
        {
            const auto options = torch::TensorOptions().dtype(torch::kFloat64).device(reference.device());
            if (!storage.defined()) {
                storage = torch::zeros({}, options);
            } else if (storage.device() != reference.device() || storage.scalar_type() != torch::kFloat64) {
                storage = storage.to(options);
            }
            return storage;
        }
        struct GaussNewtonHessian {
            static torch::Tensor update(const torch::Tensor& grad) {
                return grad.mul(grad);
            }
        };

        struct HessianDiagonal {
            static torch::Tensor update(const torch::Tensor& grad) {
                return grad.abs();
            }
        };

        template <class OptionsType, class HessianUpdater>
        class SophiaImpl : public torch::optim::Optimizer {
        public:
            explicit SophiaImpl(
                const std::vector<torch::optim::OptimizerParamGroup>& param_groups,
                OptionsType defaults = {})
                : torch::optim::Optimizer(param_groups, std::make_unique<OptionsType>(defaults)) {
                validate(defaults);
            }

            explicit SophiaImpl(std::vector<torch::Tensor> params, OptionsType defaults = {})
                : SophiaImpl({torch::optim::OptimizerParamGroup(std::move(params))}, std::move(defaults)) {}

            torch::Tensor step(LossClosure closure = nullptr) override {
                torch::NoGradGuard no_grad;
                torch::Tensor loss;
                if (closure != nullptr) {
                    torch::AutoGradMode enable_grad(true);
                    loss = closure();
                }

                for (auto& group : this->param_groups_) {
                    auto& options = static_cast<OptionsType&>(group.options());
                    for (auto& param : group.params()) {
                        if (!param.grad().defined()) {
                            continue;
                        }

                        const auto& grad = param.grad();
                        TORCH_CHECK(!grad.is_sparse(), "Sophia optimizers do not support sparse gradients.");

                        auto state_it = this->state_.find(param.unsafeGetTensorImpl());
                        if (state_it == this->state_.end()) {
                            auto state = std::make_unique<SophiaParamState>();
                            state->step(torch::zeros({}, torch::TensorOptions().dtype(torch::kFloat64).device(param.device())));
                            state->momentum(torch::zeros_like(param, torch::MemoryFormat::Preserve));
                            state->hessian(torch::zeros_like(param, torch::MemoryFormat::Preserve));
                            state_it = this->state_.insert({param.unsafeGetTensorImpl(), std::move(state)}).first;
                        }

                        auto& state = static_cast<SophiaParamState&>(*state_it->second);
                        auto& step_buffer = state.step();
                        auto& momentum = state.momentum();
                        auto& hessian = state.hessian();

                        detail::ensure_step_tensor(step_buffer, param);
                        auto step_increment = detail::make_scalar_like(step_buffer, 1.0);
                        step_buffer.add_(step_increment);

                        if (options.weight_decay() != 0.0) {
                            param.mul_(1.0 - options.lr() * options.weight_decay());
                        }

                        momentum.mul_(options.beta1()).add_(grad, 1.0 - options.beta1());

                        const auto update_interval = std::max<int64_t>(1, options.hessian_update_interval());
                        auto interval_tensor = detail::make_scalar_like(step_buffer, static_cast<double>(update_interval));
                        auto remainder = torch::remainder(step_buffer, interval_tensor);
                        auto update_mask = torch::eq(remainder, 0).to(hessian.scalar_type());

                        auto hessian_update = HessianUpdater::update(grad);
                        if (hessian_update.scalar_type() != hessian.scalar_type()) {
                            hessian_update = hessian_update.to(hessian.scalar_type());
                        }
                        hessian_update.mul_(update_mask);
                        hessian.mul_(options.beta2()).add_(hessian_update, 1.0 - options.beta2());

                        auto denom = torch::add(hessian, options.rho());
                        denom = denom.add(options.eps());
                        auto scaled = momentum / denom;
                        if (options.clip() > 0.0) {
                            scaled = torch::clamp(scaled, -options.clip(), options.clip());
                        }

                        param.add_(scaled, -options.lr());
                    }
                }

                return loss;
            }

            void save(torch::serialize::OutputArchive& archive) const override {
                torch::optim::serialize<SophiaParamState, OptionsType>(archive, *this);
            }

            void load(torch::serialize::InputArchive& archive) override {
                torch::optim::serialize<SophiaParamState, OptionsType>(archive, *this);
            }

            void ensure_state_initialized()
            {
                for (auto& group : this->param_groups_) {
                    for (auto& param : group.params()) {
                        auto state_it = this->state_.find(param.unsafeGetTensorImpl());
                        if (state_it == this->state_.end()) {
                            auto state = std::make_unique<SophiaParamState>();
                            state->step(torch::zeros({}, torch::TensorOptions().dtype(torch::kFloat64).device(param.device())));
                            state->momentum(torch::zeros_like(param, torch::MemoryFormat::Preserve));
                            state->hessian(torch::zeros_like(param, torch::MemoryFormat::Preserve));
                            this->state_.insert({param.unsafeGetTensorImpl(), std::move(state)});
                        } else {
                            auto& state = static_cast<SophiaParamState&>(*state_it->second);
                            if (!state.step().defined()) {
                                state.step(torch::zeros({}, torch::TensorOptions().dtype(torch::kFloat64).device(param.device())));
                            } else if (state.step().device() != param.device() || state.step().scalar_type() != torch::kFloat64) {
                                state.step(state.step().to(torch::TensorOptions().dtype(torch::kFloat64).device(param.device())));
                            }
                            if (!state.momentum().defined()) {
                                state.momentum(torch::zeros_like(param, torch::MemoryFormat::Preserve));
                            }
                            if (!state.hessian().defined()) {
                                state.hessian(torch::zeros_like(param, torch::MemoryFormat::Preserve));
                            }
                        }
                    }
                }
            }

        private:
            static void validate(const OptionsType& options) {
                TORCH_CHECK(options.lr() >= 0.0, "Invalid learning rate: ", options.lr());
                TORCH_CHECK(options.beta1() >= 0.0 && options.beta1() < 1.0,
                            "Invalid beta1 value: ", options.beta1());
                TORCH_CHECK(options.beta2() >= 0.0 && options.beta2() < 1.0,
                            "Invalid beta2 value: ", options.beta2());
                TORCH_CHECK(options.rho() >= 0.0, "Invalid rho value: ", options.rho());
                TORCH_CHECK(options.eps() >= 0.0, "Invalid epsilon value: ", options.eps());
                TORCH_CHECK(options.weight_decay() >= 0.0,
                            "Invalid weight_decay value: ", options.weight_decay());
                TORCH_CHECK(options.hessian_update_interval() >= 1,
                            "hessian_update_interval must be >= 1.");
            }
        };
    }

    using SophiaG = detail::SophiaImpl<SophiaGOptions, detail::GaussNewtonHessian>;
    using SophiaH = detail::SophiaImpl<SophiaHOptions, detail::HessianDiagonal>;
}
#endif //Nott_SOPHIA_HPP