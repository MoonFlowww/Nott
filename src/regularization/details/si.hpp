#ifndef Nott_SI_HPP
#define Nott_SI_HPP
/// https://arxiv.org/pdf/1703.04200
#include <torch/torch.h>

namespace Nott::Regularization::Details {

    struct SIOptions {
        double strength{0.0};
        double damping{1e-3};
    };

    struct SIDescriptor {
        SIOptions options{};
    };

    struct SIState {
        torch::Tensor reference;
        torch::Tensor importance;
    };

    [[nodiscard]] inline torch::Tensor penalty(const SIDescriptor& descriptor,
                                               const torch::Tensor& params,
                                               const SIState& state) {
        TORCH_CHECK(state.reference.defined(), "SI requires a reference parameter tensor.");
        TORCH_CHECK(state.importance.defined(), "SI requires an importance tensor.");

        const auto& options = descriptor.options;
        if (options.strength == 0.0) {
            return params.new_zeros({});
        }

        auto reference = state.reference.to(params.device(), params.scalar_type());
        auto importance = state.importance.to(params.device(), params.scalar_type());

        TORCH_CHECK(reference.sizes() == params.sizes(), "SI reference tensor must match parameter shape.");
        TORCH_CHECK(importance.sizes() == params.sizes(), "SI importance tensor must match parameter shape.");

        auto diff = params - reference;
        auto quadratic = diff.pow(2);
        auto stabilized = quadratic / (importance + options.damping);
        return stabilized.mul(importance).sum().mul(options.strength);
    }

}

#endif //Nott_SI_HPP