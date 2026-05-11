#ifndef Nott_SWA_HPP
#define Nott_SWA_HPP
/// "Averaging Weights Leads to Wider Optima in Deep Learning" (SWA) https://arxiv.org/pdf/1803.05407
#include <torch/torch.h>

namespace Nott::Regularization::Details {

    struct SWAOptions {
        double coefficient{0.0};
    };

    struct SWADescriptor {
        SWAOptions options{};
    };

    struct SWAState {
        torch::Tensor average;
    };

    [[nodiscard]] inline torch::Tensor penalty(const SWADescriptor& descriptor,
                                           const torch::Tensor& params) {
        const auto& options = descriptor.options;
        if (options.coefficient == 0.0) {
            return params.new_zeros({});
        }

        return params.new_zeros({});
    }

    [[nodiscard]] inline torch::Tensor penalty(const SWADescriptor& descriptor,
                                               const torch::Tensor& params,
                                               const SWAState& state) {
        const auto& options = descriptor.options;
        if (options.coefficient == 0.0 || !state.average.defined()) {
            return params.new_zeros({});
        }

        auto average = state.average.to(params.device(), params.scalar_type());
        TORCH_CHECK(average.sizes() == params.sizes(), "SWA average tensor must match parameter shape.");

        auto diff = params - average;
        return diff.pow(2).mean().mul(options.coefficient);
    }

}

#endif // Nott_SWA_HPP