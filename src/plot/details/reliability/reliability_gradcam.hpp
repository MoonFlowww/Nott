#ifndef Nott_RELIABILITY_GRADCAM_HPP
#define Nott_RELIABILITY_GRADCAM_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <torch/torch.h>

#include "../data.hpp"
#include "reliability_curve_utils.hpp"

namespace Nott::Plot::Details::Reliability {
    namespace detail {
        inline auto ResolveModules(Nott::Model& model) -> std::vector<std::shared_ptr<torch::nn::Module>>
        {
            auto modules = model.modules(/*include_self=*/false);
            std::vector<std::shared_ptr<torch::nn::Module>> filtered;
            filtered.reserve(modules.size());
            for (auto& module : modules) {
                if (!module) {
                    continue;
                }
                filtered.push_back(module);
            }
            return filtered;
        }

        inline auto ResolveTargetLayer(Nott::Model& model, std::optional<std::size_t> requested) -> std::shared_ptr<torch::nn::Module> {
            auto modules = ResolveModules(model);
            if (modules.empty()) {
                throw std::runtime_error("Model does not expose any modules.");
            }

            const auto adjust_index = [&](std::size_t index) -> std::shared_ptr<torch::nn::Module> {
                if (index >= modules.size()) {
                    throw std::out_of_range("Requested Grad-CAM layer index exceeds available modules.");
                }
                return modules[index];
            };

            if (requested) {
                return adjust_index(*requested);
            }

            std::shared_ptr<torch::nn::Module> fallback{};
            for (auto it = modules.rbegin(); it != modules.rend(); ++it) {
                if (!*it || it->get() == &model) {
                    continue;
                }
                if (dynamic_cast<torch::nn::Conv2dImpl*>(it->get()) != nullptr
                    || dynamic_cast<torch::nn::Conv1dImpl*>(it->get()) != nullptr
                    || dynamic_cast<torch::nn::BatchNorm2dImpl*>(it->get()) != nullptr) {
                    return *it;
                }
                if (!fallback) {
                    fallback = *it;
                }
            }

            if (!fallback) {
                fallback = adjust_index(0);
            }
            return fallback;
        }

        inline auto EnsureThreeChannel(torch::Tensor tensor) -> torch::Tensor
        {
            tensor = tensor.detach();
            if (tensor.dim() == 2) {
                tensor = tensor.unsqueeze(0);
            }
            if (tensor.dim() != 3) {
                throw std::invalid_argument(
                    "Grad-CAM expects image-like inputs with at least two spatial dimensions.");
            }

            tensor = tensor.to(torch::kFloat32).cpu();
            if (tensor.size(0) == 1) {
                tensor = tensor.repeat({3, 1, 1});
            } else if (tensor.size(0) >= 3) {
                tensor = tensor.index({torch::indexing::Slice(0, 3), torch::indexing::Ellipsis})
                             .contiguous();
            } else {
                tensor = tensor.repeat({3, 1, 1});
                tensor = tensor.index({torch::indexing::Slice(0, 3), torch::indexing::Ellipsis})
                             .contiguous();
            }

            const auto min_val = tensor.min().item<double>();
            const auto max_val = tensor.max().item<double>();
            if (max_val > min_val) {
                tensor = (tensor - min_val) / (max_val - min_val);
            }
            return tensor.clamp(0.0, 1.0);
        }

        inline auto ApplyColormap(const torch::Tensor& heatmap) -> torch::Tensor
        {
            auto map = heatmap.detach().to(torch::kFloat32).cpu();
            if (map.dim() != 2) {
                throw std::invalid_argument("Grad-CAM heatmap expects a 2D tensor.");
            }

            auto normalized = map.clamp(0.0, 1.0);
            const auto height = normalized.size(0);
            const auto width = normalized.size(1);

            auto colored = torch::empty({3, height, width}, torch::TensorOptions().dtype(torch::kFloat32));
            auto src = normalized.accessor<float, 2>();
            auto dst = colored.accessor<float, 3>();

            auto compute = [](float value) {
                value = std::clamp(value, 0.0f, 1.0f);
                const float r = std::clamp(1.5f - std::fabs(4.0f * value - 3.0f), 0.0f, 1.0f);
                const float g = std::clamp(1.5f - std::fabs(4.0f * value - 2.0f), 0.0f, 1.0f);
                const float b = std::clamp(1.5f - std::fabs(4.0f * value - 1.0f), 0.0f, 1.0f);
                return std::array<float, 3>{r, g, b};
            };

            for (int64_t y = 0; y < height; ++y) {
                for (int64_t x = 0; x < width; ++x) {
                    const auto color = compute(src[y][x]);
                    dst[0][y][x] = color[0];
                    dst[1][y][x] = color[1];
                    dst[2][y][x] = color[2];
                }
            }
            return colored;
        }

        inline auto OverlayHeatmap(const torch::Tensor& base, const torch::Tensor& heatmap, double alpha) -> torch::Tensor {
            auto basePrepared = EnsureThreeChannel(base);
            auto heatmapPrepared = ApplyColormap(heatmap);
            auto blended = (1.0 - alpha) * basePrepared + alpha * heatmapPrepared;
            return blended.clamp(0.0, 1.0);
        }

        inline auto ToTitles(std::size_t sample_index, bool overlay) -> std::vector<std::string> {
            std::vector<std::string> titles;
            titles.emplace_back("Sample " + std::to_string(sample_index) + " - input");
            titles.emplace_back("Sample " + std::to_string(sample_index) + " - Grad-CAM");
            if (overlay) {
                titles.emplace_back("Sample " + std::to_string(sample_index) + " - overlay");
            }
            return titles;
        }

        inline auto BuildHeatmap(torch::Tensor activation,
                                 torch::Tensor gradients,
                                 const std::vector<int64_t>& target_size,
                                 bool normalize) -> torch::Tensor
        {
            if (!activation.defined() || !gradients.defined()) {
                throw std::runtime_error("Grad-CAM failed to capture layer activations.");
            }

            auto grad = gradients.detach();
            auto act = activation.detach();

            if (grad.dim() < 3 || act.dim() < 3) {
                throw std::invalid_argument(
                    "Grad-CAM currently supports convolutional style activations.");
            }

            std::vector<int64_t> spatial_axes;
            for (int64_t dim = 2; dim < grad.dim(); ++dim) {
                spatial_axes.push_back(dim);
            }
            auto pooled = grad.mean(spatial_axes, /*keepdim=*/true);
            auto weighted = pooled * act;
            auto cam = weighted.sum(1, true);
            cam = torch::relu(cam);

            if (!target_size.empty()) {
                auto options = torch::nn::functional::InterpolateFuncOptions().size(target_size).align_corners(false);
                if (target_size.size() == 1) {
                    options.mode(torch::kLinear);
                } else {
                    options.mode(torch::kBilinear);
                }
                cam = torch::nn::functional::interpolate(cam, options);
            }

            cam = cam.squeeze();
            if (normalize) {
                const auto min_val = cam.min().item<double>();
                const auto max_val = cam.max().item<double>();
                const double epsilon = 1e-8;
                if (max_val - min_val > epsilon) {
                    cam = (cam - min_val) / (max_val - min_val);
                } else {
                    cam = torch::zeros_like(cam);
                }
            }
            return cam.to(torch::kFloat32).cpu();
        }
    }

    inline void RenderGradCAM(Model& model, const Plot::Reliability::GradCAMDescriptor& descriptor,
                              torch::Tensor inputs, torch::Tensor targets, std::optional<std::size_t> targetLayer) {
        if (!inputs.defined()) {
            throw std::invalid_argument("Grad-CAM requires defined input tensors.");
        }
        if (inputs.dim() < 4) {
            throw std::invalid_argument(
                "Grad-CAM expects batched image tensors shaped as (N, C, H, W).");
        }

        const auto total_samples = static_cast<std::size_t>(inputs.size(0));
        if (total_samples == 0) {
            throw std::invalid_argument("Grad-CAM requires at least one input sample.");
        }

        auto flattened_targets = Interpretability::FlattenTargets(std::move(targets), total_samples);
        auto selected_indices = Interpretability::SelectIndices(total_samples,
                                                                descriptor.options.samples,
                                                                descriptor.options.random);

        const auto device = model.device();
        inputs = inputs.to(device);
        auto module = detail::ResolveTargetLayer(model, targetLayer);


        const bool was_training = model.is_training();
        model.eval();

        std::vector<torch::Tensor> render_images;
        std::vector<std::string> titles;
        render_images.reserve(selected_indices.size() * (descriptor.options.overlay ? 3 : 2));
        titles.reserve(selected_indices.size() * (descriptor.options.overlay ? 3 : 2));

        for (auto index : selected_indices) {
            auto sample = inputs.index({static_cast<int64_t>(index)}).unsqueeze(0).clone();
            auto target_class = flattened_targets.index({static_cast<int64_t>(index)}).item<int64_t>();

            auto capture = model.forward_with_activation_capture(sample, module.get());
            auto logits = std::move(capture.logits);
            if (logits.dim() == 1) {
                logits = logits.unsqueeze(0);
            }
            if (logits.dim() != 2) {
                throw std::runtime_error("Grad-CAM expects model outputs shaped as (batch, classes).");
            }

            auto selected = logits.index({0, target_class});
            auto activation = capture.activation;
            if (!activation.defined() || !activation.requires_grad()) {
                throw std::runtime_error("Target layer activation does not require gradients.");
            }

            model.zero_grad();
            auto gradients_vec = torch::autograd::grad({selected}, {activation});
            if (gradients_vec.empty() || !gradients_vec.front().defined()) {
                throw std::runtime_error("Failed to compute Grad-CAM gradients for the requested activation.");
            }

            auto gradients = gradients_vec.front();
            std::vector<int64_t> target_size;
            if (sample.dim() >= 4) {
                for (int64_t dim = 2; dim < sample.dim(); ++dim) {
                    target_size.push_back(sample.size(dim));
                }
            }
            auto heatmap = detail::BuildHeatmap(activation, gradients, target_size, descriptor.options.normalize);

            auto prepared_input = sample.squeeze().to(torch::kCPU);
            render_images.push_back(detail::EnsureThreeChannel(prepared_input));
            render_images.push_back(detail::ApplyColormap(heatmap));
            if (descriptor.options.overlay) {
                render_images.push_back(detail::OverlayHeatmap(prepared_input, heatmap, 0.45));
            }

            auto sample_titles = detail::ToTitles(index, descriptor.options.overlay);
            titles.insert(titles.end(), sample_titles.begin(), sample_titles.end());
        }

        if (was_training) {
            model.train();
        }


        if (render_images.empty()) {
            throw std::runtime_error("Grad-CAM did not capture any visualisations.");
        }

        auto stacked = torch::stack(render_images);
        std::vector<std::size_t> indices(stacked.size(0));
        std::iota(indices.begin(), indices.end(), 0);

        Plot::Data::ImagePlotOptions options;
        options.layoutTitle = "Grad-CAM";
        options.imageTitles = titles;
        options.showColorBox = true;
        options.columns = descriptor.options.overlay ? 3 : 2;
        Plot::Data::Image images(stacked, indices, options);
        (void)images;
    }
}
#endif // Nott_RELIABILITY_GRADCAM_HPP