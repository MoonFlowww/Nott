#ifndef Nott_DATA_TRANSFORM_DIMENSIONALITY_REDUCTION_PCA_HPP
#define Nott_DATA_TRANSFORM_DIMENSIONALITY_REDUCTION_PCA_HPP

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <tuple>

#include <torch/torch.h>

#include "common.hpp"
#include "../../../core/core.hpp"

namespace Nott::Data::Transform::DimensionalityReduction {
    struct PCAResult {
        torch::Tensor components;
        torch::Tensor explained_variance;
        torch::Tensor singular_values;
        torch::Tensor mean;
        torch::Tensor transformed;
    };

    template <bool BufferVRAM = false, class DevicePolicyT = Core::DevicePolicy<BufferVRAM>>
    [[nodiscard]] inline PCAResult PCA(const torch::Tensor& input,
                                       std::size_t components = 0,
                                       bool center = true,
                                       bool whiten = false,
                                       bool keep_on_device = BufferVRAM) {
        if (input.dim() != 2) {
            throw std::invalid_argument("PCA expects a two-dimensional tensor (observations x features).");
        }

        auto device = torch::Device(torch::kCPU);
        if constexpr (BufferVRAM) {
            device = DevicePolicyT::select();
        }

        auto data = input.to(device, torch::kFloat32);
        if (!data.is_contiguous()) {
            data = data.contiguous();
        }

        const auto samples = data.size(0);
        const auto features = data.size(1);

        if (components == 0 || components > static_cast<std::size_t>(std::min(samples, features))) {
            components = static_cast<std::size_t>(std::min(samples, features));
        }

        torch::Tensor mean = torch::zeros({features}, data.options());
        if (center) {
            mean = data.mean(0);
            data = data - mean;
        }

        auto svd = torch::linalg_svd(data, /*full_matrices=*/false);
        auto U = std::get<0>(svd);
        auto S = std::get<1>(svd);
        auto Vh = std::get<2>(svd);

        const auto component_count = static_cast<int64_t>(components);
        auto selected_singular = S.narrow(0, 0, component_count);
        auto components_matrix = Vh.narrow(0, 0, component_count);

        auto explained_variance = (selected_singular.pow(2)) / std::max<int64_t>(samples - 1, 1);
        auto transformed = U.narrow(1, 0, component_count) * selected_singular;

        if (whiten) {
            auto eps = 1e-12f;
            auto scaling = torch::reciprocal(torch::sqrt(explained_variance + eps));
            transformed = transformed * scaling;
        }

        if (!keep_on_device) {
            components_matrix = components_matrix.to(torch::kCPU);
            explained_variance = explained_variance.to(torch::kCPU);
            selected_singular = selected_singular.to(torch::kCPU);
            mean = mean.to(torch::kCPU);
            transformed = transformed.to(torch::kCPU);
        }

        return {components_matrix, explained_variance, selected_singular, mean, transformed};
    }

    template <bool BufferVRAM = false, class DevicePolicyT = Core::DevicePolicy<BufferVRAM>>
    [[nodiscard]] inline torch::Tensor ProjectPCA(const torch::Tensor& input,
                                                 const PCAResult& pca,
                                                 bool keep_on_device = BufferVRAM) {
        auto device = torch::Device(torch::kCPU);
        if constexpr (BufferVRAM) {
            device = DevicePolicyT::select();
        }

        auto components = keep_on_device ? pca.components.to(device) : pca.components.to(torch::kCPU);
        auto mean = keep_on_device ? pca.mean.to(device) : pca.mean.to(torch::kCPU);
        auto data = input.to(components.device(), torch::kFloat32);
        if (mean.numel() == data.size(1)) {
            data = data - mean;
        }

        auto projected = torch::matmul(data, components.transpose(0, 1));

        if (!keep_on_device && projected.device().is_cuda()) {
            projected = projected.to(torch::kCPU);
        }
        return projected;
    }
}

#endif // Nott_DATA_TRANSFORM_DIMENSIONALITY_REDUCTION_PCA_HPP