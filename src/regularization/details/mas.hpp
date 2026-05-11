#ifndef Nott_MAS_HPP
#define Nott_MAS_HPP
/// https://arxiv.org/pdf/1711.09601
#include <torch/torch.h>

namespace Nott::Regularization::Details {

    struct MASOptions {
        double strength{0.0};
    };

    struct MASDescriptor {
        MASOptions options{};
    };

    struct MASState {
        torch::Tensor reference;
        torch::Tensor importance;
    };

    [[nodiscard]] inline torch::Tensor penalty(const MASDescriptor& descriptor,
                                               const torch::Tensor& params,
                                               const MASState& state) {
        TORCH_CHECK(state.reference.defined(), "MAS requires a reference parameter tensor.");
        TORCH_CHECK(state.importance.defined(), "MAS requires an importance tensor.");

        const auto& options = descriptor.options;
        if (options.strength == 0.0) {
            return params.new_zeros({});
        }

        auto reference = state.reference.to(params.device(), params.scalar_type());
        auto importance = state.importance.to(params.device(), params.scalar_type());

        TORCH_CHECK(reference.sizes() == params.sizes(), "MAS reference tensor must match parameter shape.");
        TORCH_CHECK(importance.sizes() == params.sizes(), "MAS importance tensor must match parameter shape.");

        auto diff = params - reference;
        auto weighted = importance * diff.pow(2);
        return weighted.sum().mul(options.strength);
    }

}
#endif //Nott_MAS_HPP