#ifndef Nott_DATA_TRANSFORM_DIMENSIONALITY_REDUCTION_RPCA_HPP
#define Nott_DATA_TRANSFORM_DIMENSIONALITY_REDUCTION_RPCA_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <utility>

#include <torch/torch.h>

#include "common.hpp"
#include "../../../core/core.hpp"

namespace Nott::Data::Transform::DimensionalityReduction {
    template <bool BufferVRAM = false, class DevicePolicyT = Core::DevicePolicy<BufferVRAM>>
    [[nodiscard]] inline std::pair<torch::Tensor, torch::Tensor>
    RPCA(const torch::Tensor& input,
         double lambda = -1.0,
         double tolerance = 1e-7,
         std::size_t max_iterations = 1000,
         double rho = 1.5,
         bool keep_on_device = BufferVRAM) {
        if (input.dim() != 2) {
            throw std::invalid_argument("RPCA expects a two-dimensional tensor (observations x features).");
        }

        auto device = torch::Device(torch::kCPU);
        if constexpr (BufferVRAM) {
            device = DevicePolicyT::select();
        }

        auto data = input.to(device, torch::kFloat32).clone();
        if (!data.is_contiguous()) {
            data = data.contiguous();
        }

        const auto rows = data.size(0);
        const auto cols = data.size(1);

        if (lambda <= 0.0) {
            lambda = 1.0 / std::sqrt(static_cast<double>(std::max(rows, cols)));
        }

        const auto norm_two = Details::spectral_norm(data);
        const auto norm_inf = data.abs().sum(1).amax().item<double>();
        const auto dual_norm = std::max(norm_two, norm_inf / lambda);

        auto Y = data / (dual_norm > 0.0 ? dual_norm : 1.0);
        double mu = 1.25 / (norm_two > 0.0 ? norm_two : 1.0);
        const double mu_bar = mu * 1e7;

        auto L = torch::zeros_like(data);
        auto S = torch::zeros_like(data);

        const double fro_norm = Details::safe_norm(data, 1.0);

        std::size_t iteration = 0;
        double error = std::numeric_limits<double>::max();

        while (iteration < max_iterations && error > tolerance) {
            const auto current = data - S + (1.0 / mu) * Y;
            auto svd = torch::linalg_svd(current, /*full_matrices=*/false);
            auto U = std::get<0>(svd);
            auto Sigma = std::get<1>(svd);
            auto Vh = std::get<2>(svd);

            auto sigma_threshold = Details::shrinkage(Sigma, 1.0 / mu);
            auto diag_sigma = torch::diag(sigma_threshold);
            auto L_next = U.matmul(diag_sigma).matmul(Vh);

            auto temp = data - L_next + (1.0 / mu) * Y;
            auto S_next = Details::shrinkage(temp, lambda / mu);

            auto residual = data - L_next - S_next;
            Y = Y + mu * residual;
            mu = std::min(mu * rho, mu_bar);

            error = residual.norm().item<double>() / fro_norm;
            L = L_next;
            S = S_next;
            ++iteration;
        }

        if (!keep_on_device) {
            L = L.to(torch::kCPU);
            S = S.to(torch::kCPU);
        }

        return {L, S};
    }
}

#endif // Nott_DATA_TRANSFORM_DIMENSIONALITY_REDUCTION_RPCA_HPP