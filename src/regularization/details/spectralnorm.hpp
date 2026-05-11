#ifndef Nott_REGULARIZATION_DETAILS_SPECTRALNORM_HPP
#define Nott_REGULARIZATION_DETAILS_SPECTRALNORM_HPP
/// "Spectral Normalization for Generative Adversarial Networks" https://arxiv.org/pdf/1802.05957
#include <torch/torch.h>

#include <tuple>

namespace Nott::Regularization::Details {

    struct SpectralNormOptions {
        double coefficient{0.0};
        double target{1.0};
    };

    struct SpectralNormDescriptor {
        SpectralNormOptions options{};
    };

    [[nodiscard]] inline torch::Tensor penalty(const SpectralNormDescriptor& descriptor, const torch::Tensor& params)
    {
        const auto& options = descriptor.options;
        if (options.coefficient == 0.0 || params.numel() == 0) {
            return params.new_zeros({});
        }

        auto tensor = params;
        if (tensor.dim() < 2) {
            tensor = tensor.view({1, -1});
        } else {
            tensor = tensor.flatten(1);
        }

        auto singular_values = torch::linalg_svdvals(tensor);
        auto max_result = singular_values.max(0);
        auto sigma_max = std::get<0>(max_result);
        auto penalty_value = torch::relu(sigma_max - options.target).pow(2);
        return penalty_value.mul(options.coefficient);
    }

}

#endif // Nott_REGULARIZATION_DETAILS_SPECTRALNORM_HPP