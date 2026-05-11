#ifndef Nott_MUON_HPP
#define Nott_MUON_HPP
/// Muon, AdaMuon, MuonManifold
/// Muon: https://arxiv.org/pdf/2502.16982
/// AdaMuon: https://arxiv.org/pdf/2507.11005
/// MuonManifold: https://thinkingmachines.ai/blog/modular-manifolds/

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <tuple>
#include <utility>
#include <type_traits>

#include <torch/torch.h>
#include <torch/optim/optimizer.h>
#include <torch/optim/serialize.h>

namespace Nott::Optimizer::Details {

    namespace detail {
        namespace muon {
            inline auto make_scalar_like(const torch::Tensor& reference, double value) -> torch::Tensor {
                if (reference.defined()) {
                    return torch::full({}, value, reference.options());
                }
                return torch::full({}, value, torch::TensorOptions().dtype(torch::kFloat));
            }

            inline void ensure_step_tensor(torch::Tensor& storage, const torch::Tensor& reference)
            {
                const auto options = torch::TensorOptions().dtype(torch::kFloat64).device(reference.device());
                if (!storage.defined()) {
                    storage = torch::zeros({}, options);
                } else if (storage.device() != reference.device() || storage.scalar_type() != torch::kFloat64) {
                    storage = storage.to(options);
                }
            }
        }

        inline auto make_step_scalar_like(const torch::Tensor& reference, double value) -> torch::Tensor
        {
            auto options = torch::TensorOptions().dtype(torch::kFloat64);
            if (reference.defined()) {
                options = options.device(reference.device());
            }
            return torch::full({}, value, options);
        }

        inline auto safe_norm(const torch::Tensor& tensor, double eps) -> torch::Tensor {
            auto norm = torch::linalg_vector_norm(tensor);
            return norm.clamp_min(eps);
        }

        inline auto clip_by_max_norm(const torch::Tensor& update, double max_norm, double eps) -> torch::Tensor {
            if (max_norm <= 0.0) {
                return update;
            }

            auto current_norm = torch::linalg_vector_norm(update);
            auto threshold = detail::muon::make_scalar_like(current_norm, max_norm);
            auto eps_scalar = detail::muon::make_scalar_like(current_norm, eps);
            auto scale = (threshold / (current_norm + eps_scalar)).clamp_max(1.0);
            return update * scale;
        }

        inline void ensure_tensor_like(torch::Tensor& storage, const torch::Tensor& reference) {
            if (!storage.defined()) {
                storage = torch::zeros_like(reference);
            }
        }


        inline void apply_weight_decay(torch::Tensor& update, const torch::Tensor& param, double weight_decay) {
            if (weight_decay == 0.0) {
                return;
            }

            update = update + weight_decay * param;
        }
    }

    struct MuonOptions : public torch::optim::OptimizerCloneableOptions<MuonOptions> {
        MuonOptions(double lr = 1e-3) : lr_(lr) {}

        TORCH_ARG(double, lr) = 1e-3;
        TORCH_ARG(double, beta) = 0.95;
        TORCH_ARG(double, weight_decay) = 0.0;
        TORCH_ARG(double, eps) = 1e-8;
        TORCH_ARG(double, max_update_norm) = 0.0;

    public:

        [[nodiscard]] inline auto validated() const -> MuonOptions {
            MuonOptions copy = *this;
            copy.beta(std::clamp(copy.beta(), 0.0, 1.0));
            copy.lr(std::max(copy.lr(), 0.0));
            copy.eps(std::max(copy.eps(), 0.0));
            copy.max_update_norm(std::max(copy.max_update_norm(), 0.0));
            copy.weight_decay(std::max(copy.weight_decay(), 0.0));
            return copy;
        }
        void serialize(torch::serialize::InputArchive& archive) override {
            _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, lr);
            _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, beta);
            _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, weight_decay);
            _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, eps);
            _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, max_update_norm);
        }

        void serialize(torch::serialize::OutputArchive& archive) const override {
            _TORCH_OPTIM_SERIALIZE_TORCH_ARG(lr);
            _TORCH_OPTIM_SERIALIZE_TORCH_ARG(beta);
            _TORCH_OPTIM_SERIALIZE_TORCH_ARG(weight_decay);
            _TORCH_OPTIM_SERIALIZE_TORCH_ARG(eps);
            _TORCH_OPTIM_SERIALIZE_TORCH_ARG(max_update_norm);
        }

        double get_lr() const override { return lr(); }
        void set_lr(double value) override { lr(value); }
    };

    struct MuonState : public torch::optim::OptimizerCloneableParamState<MuonState> {
        TORCH_ARG(torch::Tensor, exp_avg);
        TORCH_ARG(torch::Tensor, step);

    public:

        void serialize(torch::serialize::InputArchive& archive) override {
            _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(torch::Tensor, exp_avg);
            _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(torch::Tensor, step);
        }

        void serialize(torch::serialize::OutputArchive& archive) const override {
            _TORCH_OPTIM_SERIALIZE_TORCH_ARG(exp_avg);
            _TORCH_OPTIM_SERIALIZE_TORCH_ARG(step);
        }

        void reset() {
            exp_avg(torch::Tensor());
            step(torch::Tensor());
        }
    };

    inline void muon_step_(torch::Tensor& param,
                           const torch::Tensor& grad,
                           MuonState& state,
                           const MuonOptions& raw_options) {
        auto options = raw_options.validated();

        auto& exp_avg = state.exp_avg();
        auto& step = state.step();
        detail::ensure_tensor_like(exp_avg, grad);
        detail::muon::ensure_step_tensor(step, grad);

        auto step_increment = detail::make_step_scalar_like(step, 1.0);
        step.add_(step_increment);

        exp_avg.mul_(options.beta()).add_(grad, 1.0 - options.beta());

        auto parameter_norm = detail::safe_norm(param, options.eps());
        auto direction_norm = detail::safe_norm(exp_avg, options.eps());
        auto direction = exp_avg * (parameter_norm / direction_norm);

        detail::apply_weight_decay(direction, param, options.weight_decay());
        direction = detail::clip_by_max_norm(direction, options.max_update_norm(), options.eps());

        param.add_(direction, -options.lr());
    }

    struct MuonDescriptor {
        MuonOptions options{};
    };

    struct AdaMuonOptions : public torch::optim::OptimizerCloneableOptions<AdaMuonOptions> {
        AdaMuonOptions(double lr = 1e-3) : lr_(lr) {}

        TORCH_ARG(double, lr) = 1e-3;
        TORCH_ARG(double, beta) = 0.95;
        TORCH_ARG(double, beta2) = 0.999;
        TORCH_ARG(double, weight_decay) = 0.0;
        TORCH_ARG(double, eps) = 1e-8;
        TORCH_ARG(double, max_update_norm) = 0.0;
    public:
        [[nodiscard]] inline auto validated() const -> AdaMuonOptions {
            AdaMuonOptions copy = *this;
            copy.beta(std::clamp(copy.beta(), 0.0, 1.0));
            copy.beta2(std::clamp(copy.beta2(), 0.0, 1.0));
            copy.lr(std::max(copy.lr(), 0.0));
            copy.eps(std::max(copy.eps(), 0.0));
            copy.max_update_norm(std::max(copy.max_update_norm(), 0.0));
            copy.weight_decay(std::max(copy.weight_decay(), 0.0));
            return copy;
        }

        void serialize(torch::serialize::InputArchive& archive) override {
            _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, lr);
            _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, beta);
            _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, beta2);
            _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, weight_decay);
            _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, eps);
            _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, max_update_norm);
        }

        void serialize(torch::serialize::OutputArchive& archive) const override {
            _TORCH_OPTIM_SERIALIZE_TORCH_ARG(lr);
            _TORCH_OPTIM_SERIALIZE_TORCH_ARG(beta);
            _TORCH_OPTIM_SERIALIZE_TORCH_ARG(beta2);
            _TORCH_OPTIM_SERIALIZE_TORCH_ARG(weight_decay);
            _TORCH_OPTIM_SERIALIZE_TORCH_ARG(eps);
            _TORCH_OPTIM_SERIALIZE_TORCH_ARG(max_update_norm);
        }

        double get_lr() const override { return lr(); }
        void set_lr(double value) override { lr(value); }
    };

    struct AdaMuonState : public torch::optim::OptimizerCloneableParamState<AdaMuonState> {
        TORCH_ARG(torch::Tensor, exp_avg);
        TORCH_ARG(torch::Tensor, exp_avg_sq);
        TORCH_ARG(torch::Tensor, step);

    public:

        void serialize(torch::serialize::InputArchive& archive) override {
            _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(torch::Tensor, exp_avg);
            _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(torch::Tensor, exp_avg_sq);
            _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(torch::Tensor, step);
        }

        void serialize(torch::serialize::OutputArchive& archive) const override {
            _TORCH_OPTIM_SERIALIZE_TORCH_ARG(exp_avg);
            _TORCH_OPTIM_SERIALIZE_TORCH_ARG(exp_avg_sq);
            _TORCH_OPTIM_SERIALIZE_TORCH_ARG(step);
        }

        void reset() {
            exp_avg(torch::Tensor());
            exp_avg_sq(torch::Tensor());
            step(torch::Tensor());
        }
    };

    inline void ada_muon_step_(torch::Tensor& param,
                               const torch::Tensor& grad,
                               AdaMuonState& state,
                               const AdaMuonOptions& raw_options) {
        auto options = raw_options.validated();

        auto& exp_avg = state.exp_avg();
        auto& exp_avg_sq = state.exp_avg_sq();
        auto& step = state.step();
        detail::ensure_tensor_like(exp_avg, grad);
        detail::ensure_tensor_like(exp_avg_sq, grad);

        detail::muon::ensure_step_tensor(step, grad);
        auto step_increment = detail::make_step_scalar_like(step, 1.0);
        step.add_(step_increment);


        exp_avg.mul_(options.beta()).add_(grad, 1.0 - options.beta());
        exp_avg_sq.mul_(options.beta2()).addcmul_(grad, grad, 1.0 - options.beta2());

        auto beta_tensor = detail::make_step_scalar_like(step, options.beta());
        auto beta2_tensor = detail::make_step_scalar_like(step, options.beta2());
        auto ones = torch::ones_like(step);
        auto bias_correction1 = (ones - torch::pow(beta_tensor, step)).to(exp_avg.scalar_type());
        auto bias_correction2 = (ones - torch::pow(beta2_tensor, step)).to(exp_avg_sq.scalar_type());

        auto corrected_exp_avg = exp_avg / bias_correction1;
        auto corrected_exp_avg_sq = exp_avg_sq / bias_correction2;
        auto denom = torch::sqrt(corrected_exp_avg_sq).add_(options.eps());
        auto direction = corrected_exp_avg / denom;

        auto parameter_norm = detail::safe_norm(param, options.eps());
        auto direction_norm = detail::safe_norm(direction, options.eps());
        direction.mul_(parameter_norm / direction_norm);

        detail::apply_weight_decay(direction, param, options.weight_decay());
        direction = detail::clip_by_max_norm(direction, options.max_update_norm(), options.eps());

        param.add_(direction, -options.lr());
    }

    struct AdaMuonDescriptor {
        AdaMuonOptions options{};
    };

    enum class ManifoldKind {
        Euclidean,
        UnitSphere,
        Stiefel
    };

    struct MuonManifoldOptions : public torch::optim::OptimizerCloneableOptions<MuonManifoldOptions> {
        MuonManifoldOptions(double lr = 1e-3) : lr_(lr) {}

        TORCH_ARG(double, lr) = 1e-3;
        TORCH_ARG(double, beta) = 0.95;
        TORCH_ARG(double, beta2) = 0.999;
        TORCH_ARG(double, weight_decay) = 0.0;
        TORCH_ARG(double, eps) = 1e-8;
        TORCH_ARG(double, max_update_norm) = 0.0;
        TORCH_ARG(double, retraction_epsilon) = 1e-12;
        TORCH_ARG(bool, renormalize) = true;

    private:
        static constexpr int64_t to_index(ManifoldKind kind) {
            return static_cast<int64_t>(kind);
        }

        static constexpr auto manifold_from_index(int64_t index) -> ManifoldKind {
            switch (static_cast<ManifoldKind>(index)) {
                case ManifoldKind::Euclidean:
                case ManifoldKind::UnitSphere:
                case ManifoldKind::Stiefel:
                    return static_cast<ManifoldKind>(index);
            }
            return ManifoldKind::UnitSphere;
        }

        TORCH_ARG(int64_t, manifold_index) = to_index(ManifoldKind::UnitSphere);

    public:

        [[nodiscard]] inline auto validated() const -> MuonManifoldOptions {
            MuonManifoldOptions copy = *this;
            copy.beta(std::clamp(copy.beta(), 0.0, 1.0));
            copy.beta2(std::clamp(copy.beta2(), 0.0, 1.0));
            copy.lr(std::max(copy.lr(), 0.0));
            copy.eps(std::max(copy.eps(), 0.0));
            copy.max_update_norm(std::max(copy.max_update_norm(), 0.0));
            copy.weight_decay(std::max(copy.weight_decay(), 0.0));
            copy.retraction_epsilon(std::max(copy.retraction_epsilon(), 0.0));
            copy.renormalize(renormalize());
            copy.manifold(manifold());
            return copy;
        }
        void serialize(torch::serialize::InputArchive& archive) override {
            _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, lr);
            _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, beta);
            _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, beta2);
            _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, weight_decay);
            _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, eps);
            _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, max_update_norm);
            _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(int64_t, manifold_index);
            _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, retraction_epsilon);
            _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(bool, renormalize);
        }

        void serialize(torch::serialize::OutputArchive& archive) const override {
            _TORCH_OPTIM_SERIALIZE_TORCH_ARG(lr);
            _TORCH_OPTIM_SERIALIZE_TORCH_ARG(beta);
            _TORCH_OPTIM_SERIALIZE_TORCH_ARG(beta2);
            _TORCH_OPTIM_SERIALIZE_TORCH_ARG(weight_decay);
            _TORCH_OPTIM_SERIALIZE_TORCH_ARG(eps);
            _TORCH_OPTIM_SERIALIZE_TORCH_ARG(max_update_norm);
            _TORCH_OPTIM_SERIALIZE_TORCH_ARG(manifold_index);
            _TORCH_OPTIM_SERIALIZE_TORCH_ARG(retraction_epsilon);
            _TORCH_OPTIM_SERIALIZE_TORCH_ARG(renormalize);
        }

        double get_lr() const override { return lr(); }
        void set_lr(double value) override { lr(value); }

        [[nodiscard]] inline auto manifold() const -> ManifoldKind {
            return manifold_from_index(manifold_index());
        }

        inline auto manifold(ManifoldKind value) -> MuonManifoldOptions& {
            manifold_index(to_index(value));
            return *this;
        }
    };
    struct MuonManifoldState : public torch::optim::OptimizerCloneableParamState<MuonManifoldState> {
        TORCH_ARG(torch::Tensor, exp_avg);
        TORCH_ARG(torch::Tensor, exp_avg_sq);
        TORCH_ARG(torch::Tensor, step);


    public:
        void serialize(torch::serialize::InputArchive& archive) override {
            _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(torch::Tensor, exp_avg);
            _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(torch::Tensor, exp_avg_sq);
            _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(torch::Tensor, step);
        }

        void serialize(torch::serialize::OutputArchive& archive) const override {
            _TORCH_OPTIM_SERIALIZE_TORCH_ARG(exp_avg);
            _TORCH_OPTIM_SERIALIZE_TORCH_ARG(exp_avg_sq);
            _TORCH_OPTIM_SERIALIZE_TORCH_ARG(step);
        }

        void reset() {
            exp_avg(torch::Tensor());
            exp_avg_sq(torch::Tensor());
            step(torch::Tensor());
        }
    };

    inline void retract_to_manifold(torch::Tensor& param,
                                    const MuonManifoldOptions& options) {
        switch (options.manifold()) {
            case ManifoldKind::Euclidean:
                return;
            case ManifoldKind::UnitSphere: {
                auto norm = detail::safe_norm(param, options.retraction_epsilon());
                param = param / norm;
                return;
            }
            case ManifoldKind::Stiefel: {
                auto qr = std::get<0>(torch::linalg_qr(param));
                param = qr;
                return;
            }
        }
    }

    inline void muon_manifold_step_(torch::Tensor& param,
                                    const torch::Tensor& grad,
                                    MuonManifoldState& state,
                                    const MuonManifoldOptions& raw_options) {
        auto options = raw_options.validated();
        auto& exp_avg = state.exp_avg();
        auto& exp_avg_sq = state.exp_avg_sq();
        auto& step = state.step();
        detail::ensure_tensor_like(exp_avg, grad);
        detail::ensure_tensor_like(exp_avg_sq, grad);

        detail::muon::ensure_step_tensor(step, grad);
        auto step_increment = detail::make_step_scalar_like(step, 1.0);
        step.add_(step_increment);


        exp_avg.mul_(options.beta()).add_(grad, 1.0 - options.beta());
        exp_avg_sq.mul_(options.beta2()).addcmul_(grad, grad, 1.0 - options.beta2());

        auto beta_tensor = detail::make_step_scalar_like(step, options.beta());
        auto beta2_tensor = detail::make_step_scalar_like(step, options.beta2());
        auto ones = torch::ones_like(step);
        auto bias_correction1 = (ones - torch::pow(beta_tensor, step)).to(exp_avg.scalar_type());
        auto bias_correction2 = (ones - torch::pow(beta2_tensor, step)).to(exp_avg_sq.scalar_type());

        auto corrected_exp_avg = exp_avg / bias_correction1;
        auto corrected_exp_avg_sq = exp_avg_sq / bias_correction2;
        auto denom = torch::sqrt(corrected_exp_avg_sq).add_(options.eps());
        auto direction = corrected_exp_avg / denom;

        auto parameter_norm = detail::safe_norm(param, options.eps());
        auto direction_norm = detail::safe_norm(direction, options.eps());
        direction.mul_(parameter_norm / direction_norm);

        detail::apply_weight_decay(direction, param, options.weight_decay());
        direction = detail::clip_by_max_norm(direction, options.max_update_norm(), options.eps());

        param.add_(direction, -options.lr());

        if (options.renormalize()) {
            retract_to_manifold(param, options);
        }
    }

    struct MuonManifoldDescriptor {
        MuonManifoldOptions options{};
    };

        namespace detail {
        template <class OptionsType, class StateType, auto StepImpl>
        class MuonOptimizerImpl : public torch::optim::Optimizer {
        public:
            explicit MuonOptimizerImpl(
                const std::vector<torch::optim::OptimizerParamGroup>& param_groups,
                OptionsType defaults = {})
                : torch::optim::Optimizer(param_groups, std::make_unique<OptionsType>(defaults.validated())) {}

            explicit MuonOptimizerImpl(std::vector<torch::Tensor> params, OptionsType defaults = {})
                : MuonOptimizerImpl({torch::optim::OptimizerParamGroup(std::move(params))}, std::move(defaults)) {}

            torch::Tensor step(LossClosure closure = nullptr) override {
                torch::NoGradGuard no_grad;
                torch::Tensor loss;
                if (closure != nullptr) {
                    torch::AutoGradMode enable_grad(true);
                    loss = closure();
                }

                for (auto& group : this->param_groups_) {
                    auto& options = static_cast<OptionsType&>(group.options());
                    auto validated = options.validated();
                    options = validated;
                    for (auto& param : group.params()) {
                        if (!param.grad().defined()) {
                            continue;
                        }
                        const auto& grad = param.grad();
                        TORCH_CHECK(!grad.is_sparse(), "Muon optimizers do not support sparse gradients.");

                        auto state_it = this->state_.find(param.unsafeGetTensorImpl());
                        if (state_it == this->state_.end()) {
                            auto state = std::make_unique<StateType>();
                            state->reset();
                            state_it = this->state_.insert({param.unsafeGetTensorImpl(), std::move(state)}).first;
                        }

                        auto& state = static_cast<StateType&>(*state_it->second);
                        StepImpl(param, grad, state, validated);
                    }
                }

                return loss;
            }

            void save(torch::serialize::OutputArchive& archive) const override {
                torch::optim::serialize<StateType, OptionsType>(archive, *this);
            }

            void load(torch::serialize::InputArchive& archive) override {
                torch::optim::serialize<StateType, OptionsType>(archive, *this);
            }

            void ensure_state_initialized()
            {
                for (auto& group : this->param_groups_) {
                    for (auto& param : group.params()) {
                        auto state_it = this->state_.find(param.unsafeGetTensorImpl());
                        if (state_it == this->state_.end()) {
                            auto state = std::make_unique<StateType>();
                            state->reset();
                            state_it = this->state_.insert({param.unsafeGetTensorImpl(), std::move(state)}).first;
                        }
                        auto& state = static_cast<StateType&>(*state_it->second);
                        if constexpr (std::is_same_v<StateType, MuonState>) {
                            detail::ensure_tensor_like(state.exp_avg(), param);
                        } else {
                            detail::ensure_tensor_like(state.exp_avg(), param);
                            detail::ensure_tensor_like(state.exp_avg_sq(), param);
                        }
                        detail::muon::ensure_step_tensor(state.step(), param);
                    }
                }
            }
        };
    }

    using Muon = detail::MuonOptimizerImpl<MuonOptions, MuonState, muon_step_>;
    using AdaMuon = detail::MuonOptimizerImpl<AdaMuonOptions, AdaMuonState, ada_muon_step_>;
    using MuonManifold = detail::MuonOptimizerImpl<MuonManifoldOptions, MuonManifoldState, muon_manifold_step_>;
}

#endif //Nott_MUON_HPP