#ifndef Nott_ADAFACTOR_HPP
#define Nott_ADAFACTOR_HPP
/// "Adafactor: Adaptive Learning Rates with Sublinear Memory Cost" https://arxiv.org/pdf/1804.04235
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include <torch/torch.h>
#include <torch/optim/optimizer.h>
#include <torch/optim/serialize.h>

namespace Nott::Optimizer::Details {

    struct AdafactorOptions : public torch::optim::OptimizerCloneableOptions<AdafactorOptions> {
        AdafactorOptions(double lr = 1e-3) : lr_(lr) {}

        TORCH_ARG(double, lr) = 1e-3;
        TORCH_ARG(double, eps1) = 1e-30;
        TORCH_ARG(double, eps2) = 1e-3;
        TORCH_ARG(double, clip_threshold) = 1.0;
        TORCH_ARG(double, decay_rate) = -0.8;
        TORCH_ARG(double, beta1) = 0.9;
        TORCH_ARG(bool, use_first_moment) = false;
        TORCH_ARG(double, weight_decay) = 0.0;
        TORCH_ARG(bool, scale_parameter) = true;
        TORCH_ARG(bool, relative_step) = true;
        TORCH_ARG(bool, warmup_init) = false;

    public:
        [[nodiscard]] inline AdafactorOptions validated() const {
            AdafactorOptions copy = *this;
            copy.lr(std::max(0.0, copy.lr()));
            copy.eps1(std::max(0.0, copy.eps1()));
            copy.eps2(std::max(0.0, copy.eps2()));
            copy.clip_threshold(std::max(0.0, copy.clip_threshold()));
            copy.weight_decay(std::max(0.0, copy.weight_decay()));
            copy.beta1(std::clamp(copy.beta1(), 0.0, 1.0));
            return copy;
        }

        void serialize(torch::serialize::InputArchive& archive) override {
            _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, lr);
            _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, eps1);
            _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, eps2);
            _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, clip_threshold);
            _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, decay_rate);
            _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, beta1);
            _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(bool, use_first_moment);
            _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, weight_decay);
            _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(bool, scale_parameter);
            _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(bool, relative_step);
            _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(bool, warmup_init);
        }

        void serialize(torch::serialize::OutputArchive& archive) const override {
            _TORCH_OPTIM_SERIALIZE_TORCH_ARG(lr);
            _TORCH_OPTIM_SERIALIZE_TORCH_ARG(eps1);
            _TORCH_OPTIM_SERIALIZE_TORCH_ARG(eps2);
            _TORCH_OPTIM_SERIALIZE_TORCH_ARG(clip_threshold);
            _TORCH_OPTIM_SERIALIZE_TORCH_ARG(decay_rate);
            _TORCH_OPTIM_SERIALIZE_TORCH_ARG(beta1);
            _TORCH_OPTIM_SERIALIZE_TORCH_ARG(use_first_moment);
            _TORCH_OPTIM_SERIALIZE_TORCH_ARG(weight_decay);
            _TORCH_OPTIM_SERIALIZE_TORCH_ARG(scale_parameter);
            _TORCH_OPTIM_SERIALIZE_TORCH_ARG(relative_step);
            _TORCH_OPTIM_SERIALIZE_TORCH_ARG(warmup_init);
        }

        double get_lr() const override { return lr(); }
        void set_lr(double value) override { lr(value); }
    };

    struct AdafactorParamState : public torch::optim::OptimizerCloneableParamState<AdafactorParamState> {
        TORCH_ARG(torch::Tensor, exp_avg);
        TORCH_ARG(torch::Tensor, exp_avg_sq);
        TORCH_ARG(torch::Tensor, exp_avg_sq_row);
        TORCH_ARG(torch::Tensor, exp_avg_sq_col);
        TORCH_ARG(int64_t, step) = 0;
        TORCH_ARG(double, rms) = 0.0;

    public:
        void serialize(torch::serialize::InputArchive& archive) override {
            _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(torch::Tensor, exp_avg);
            _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(torch::Tensor, exp_avg_sq);
            _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(torch::Tensor, exp_avg_sq_row);
            _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(torch::Tensor, exp_avg_sq_col);
            _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(int64_t, step);
            _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, rms);
        }

        void serialize(torch::serialize::OutputArchive& archive) const override {
            _TORCH_OPTIM_SERIALIZE_TORCH_ARG(exp_avg);
            _TORCH_OPTIM_SERIALIZE_TORCH_ARG(exp_avg_sq);
            _TORCH_OPTIM_SERIALIZE_TORCH_ARG(exp_avg_sq_row);
            _TORCH_OPTIM_SERIALIZE_TORCH_ARG(exp_avg_sq_col);
            _TORCH_OPTIM_SERIALIZE_TORCH_ARG(step);
            _TORCH_OPTIM_SERIALIZE_TORCH_ARG(rms);
        }

        void reset() {
            exp_avg(torch::Tensor());
            exp_avg_sq(torch::Tensor());
            exp_avg_sq_row(torch::Tensor());
            exp_avg_sq_col(torch::Tensor());
            step(0);
            rms(0.0);
        }
    };

    struct AdafactorDescriptor {
        AdafactorOptions options{};
    };

    class Adafactor : public torch::optim::Optimizer {
    public:
        using Options = AdafactorOptions;
        using ParamState = AdafactorParamState;

        explicit Adafactor(std::vector<torch::Tensor> params, Options options = {})
            : Adafactor({torch::optim::OptimizerParamGroup(std::move(params))}, std::move(options)) {}

        explicit Adafactor(std::vector<torch::optim::OptimizerParamGroup> param_groups, Options options = {})
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
                    TORCH_CHECK(!grad.is_sparse(), "Adafactor does not support sparse gradients.");

                    auto state_it = this->state_.find(param.unsafeGetTensorImpl());
                    if (state_it == this->state_.end()) {
                        auto state = std::make_unique<ParamState>();
                        state->reset();
                        state_it = this->state_.insert({param.unsafeGetTensorImpl(), std::move(state)}).first;
                    }

                    auto& state = static_cast<ParamState&>(*state_it->second);
                    state.step(state.step() + 1);

                    auto grad_working = grad;
                    if (grad.scalar_type() == torch::kFloat16 || grad.scalar_type() == torch::kBFloat16) {
                        grad_working = grad_working.to(torch::kFloat32);
                    }

                    auto param_data = param;
                    if (param.scalar_type() == torch::kFloat16 || param.scalar_type() == torch::kBFloat16) {
                        param_data = param_data.to(torch::kFloat32);
                    }

                    state.rms(param_data.pow(2).mean().sqrt().item<double>());

                    double rel_step = options.lr();
                    if (options.relative_step()) {
                        const double step = static_cast<double>(state.step());
                        const double min_step = options.warmup_init() ? 1e-6 * step : 1e-2;
                        double relative = std::min(min_step, 1.0 / std::sqrt(step));
                        if (rel_step > 0.0) {
                            rel_step *= relative;
                        } else {
                            rel_step = relative;
                        }
                    }

                    double param_scale = 1.0;
                    if (options.scale_parameter()) {
                        param_scale = std::max(options.eps2(), state.rms());
                    }

                    const double lr = rel_step * param_scale;
                    const double beta2t = 1.0 - std::pow(static_cast<double>(state.step()), options.decay_rate());

                    const bool factored = grad_working.dim() >= 2;
                    torch::Tensor update;

                    if (factored) {
                        auto& exp_avg_sq_row = state.exp_avg_sq_row();
                        auto& exp_avg_sq_col = state.exp_avg_sq_col();

                        if (!exp_avg_sq_row.defined()) {
                            std::vector<int64_t> row_shape(grad_working.sizes().begin(), grad_working.sizes().end() - 1);
                            exp_avg_sq_row = torch::zeros(row_shape, grad_working.options());
                        } else if (exp_avg_sq_row.device() != grad_working.device() || exp_avg_sq_row.dtype() != grad_working.dtype()) {
                            exp_avg_sq_row = exp_avg_sq_row.to(grad_working.options());
                        }
                        if (!exp_avg_sq_col.defined()) {
                            std::vector<int64_t> col_shape(grad_working.sizes().begin(), grad_working.sizes().end() - 2);
                            col_shape.push_back(grad_working.size(-1));
                            exp_avg_sq_col = torch::zeros(col_shape, grad_working.options());
                        } else if (exp_avg_sq_col.device() != grad_working.device() || exp_avg_sq_col.dtype() != grad_working.dtype()) {
                            exp_avg_sq_col = exp_avg_sq_col.to(grad_working.options());
                        }

                        auto grad_sq = grad_working.pow(2).add_(options.eps1());
                        exp_avg_sq_row.mul_(beta2t).add_(grad_sq.mean(-1), 1.0 - beta2t);
                        exp_avg_sq_col.mul_(beta2t).add_(grad_sq.mean(-2), 1.0 - beta2t);

                        auto row_mean = exp_avg_sq_row.mean(-1, true);
                        auto r_factor = (exp_avg_sq_row / row_mean).rsqrt().unsqueeze(-1);
                        auto c_factor = exp_avg_sq_col.unsqueeze(-2).rsqrt();
                        update = r_factor * c_factor * grad_working;
                    } else {
                        auto& exp_avg_sq = state.exp_avg_sq();
                        if (!exp_avg_sq.defined()) {
                            exp_avg_sq = torch::zeros_like(grad_working);
                        } else if (exp_avg_sq.device() != grad_working.device() || exp_avg_sq.dtype() != grad_working.dtype()) {
                            exp_avg_sq = exp_avg_sq.to(grad_working.options());
                        }
                        auto grad_sq = grad_working.pow(2).add_(options.eps1());
                        exp_avg_sq.mul_(beta2t).add_(grad_sq, 1.0 - beta2t);
                        update = grad_working / (exp_avg_sq.sqrt() + options.eps2());
                    }

                    const double update_rms = std::sqrt(update.pow(2).mean().item<double>() + 1e-16);
                    const double clipped = std::max(1.0, update_rms / std::max(1e-16, options.clip_threshold()));
                    update.div_(clipped);
                    update.mul_(lr);

                    if (options.use_first_moment()) {
                        auto& exp_avg = state.exp_avg();
                        if (!exp_avg.defined()) {
                            exp_avg = torch::zeros_like(update);
                        } else if (exp_avg.device() != update.device() || exp_avg.dtype() != update.dtype()) {
                            exp_avg = exp_avg.to(update.options());
                        }
                        exp_avg.mul_(options.beta1()).add_(update, 1.0 - options.beta1());
                        update = exp_avg;
                    }

                    if (options.weight_decay() != 0.0) {
                        param_data.add_(param_data, -options.weight_decay() * lr);
                    }

                    param_data.add_(-update);

                    if (param_data.data_ptr() != param.data_ptr()) {
                        param.copy_(param_data);
                    }
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
                    if (!state.exp_avg_sq().defined() && param.dim() < 2) {
                        state.exp_avg_sq(torch::zeros_like(param));
                    }
                }
            }
        }
    };

}

#endif // Nott_ADAFACTOR_HPP