#ifndef Nott_SWAG_HPP
#define Nott_SWAG_HPP
/// "SWAG: A Simple Baseline for Bayesian Uncertainty in Deep Learning" https://arxiv.org/pdf/1902.02476
#include <torch/torch.h>

namespace Nott::Regularization::Details {

    struct SWAGOptions {
        double coefficient{0.0};
        double variance_epsilon{1e-8};
        std::size_t start_step{0};
        std::size_t accumulation_stride{1};
        std::size_t max_snapshots{0};
    };

    struct SWAGDescriptor {
        SWAGOptions options{};
    };

    struct SWAGState {
        torch::Tensor mean;
        torch::Tensor variance;
        std::size_t snapshot_count{0};
    };

    [[nodiscard]] inline torch::Tensor penalty(const SWAGDescriptor&, const torch::Tensor& params) {
        return params.new_zeros({});
    }

    [[nodiscard]] inline torch::Tensor penalty(const SWAGDescriptor& descriptor,
                                               const torch::Tensor& params,
                                               const SWAGState& state) {
        const auto& options = descriptor.options;
        if (options.coefficient == 0.0 || !state.mean.defined() || !state.variance.defined() || state.snapshot_count < 2) {
            return params.new_zeros({});
        }

        auto mean = state.mean.to(params.device(), params.scalar_type());
        auto variance = state.variance.to(params.device(), params.scalar_type());

        TORCH_CHECK(mean.sizes() == params.sizes(), "SWAG mean tensor must match parameter shape.");
        TORCH_CHECK(variance.sizes() == params.sizes(), "SWAG variance tensor must match parameter shape.");

        const auto normalization = static_cast<double>(state.snapshot_count - 1);
        auto unbiased_variance = variance / normalization;

        auto diff = params - mean;
        auto safe_variance = unbiased_variance + options.variance_epsilon;
        auto scaled = diff.pow(2) / safe_variance;
        return scaled.mean().mul(options.coefficient);
    }

}

#endif // Nott_SWAG_HPP