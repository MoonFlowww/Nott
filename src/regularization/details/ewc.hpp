#ifndef Nott_EWC_HPP
#define Nott_EWC_HPP
/// https://arxiv.org/pdf/1612.00796
#include <torch/torch.h>

namespace Nott::Regularization::Details {

    struct EWCOptions {
        double strength{0.0};
    };

    struct EWCDescriptor {
        EWCOptions options{};
    };

    struct EWCState {
        torch::Tensor reference;
        torch::Tensor fisher_information;
    };

    [[nodiscard]] inline torch::Tensor penalty(const EWCDescriptor& descriptor,
                                               const torch::Tensor& params,
                                               const EWCState& state) {
        TORCH_CHECK(state.reference.defined(), "EWC requires a reference parameter tensor.");
        TORCH_CHECK(state.fisher_information.defined(), "EWC requires a Fisher information tensor.");

        const auto& options = descriptor.options;
        if (options.strength == 0.0) {
            return params.new_zeros({});
        }

        auto reference = state.reference.to(params.device(), params.scalar_type());
        auto fisher = state.fisher_information.to(params.device(), params.scalar_type());

        TORCH_CHECK(reference.sizes() == params.sizes(), "EWC reference tensor must match parameter shape.");
        TORCH_CHECK(fisher.sizes() == params.sizes(), "EWC Fisher tensor must match parameter shape.");

        auto diff = params - reference;
        auto quadratic = fisher * diff.pow(2);
        return quadratic.sum().mul(options.strength);
    }

}

#endif //Nott_EWC_HPP