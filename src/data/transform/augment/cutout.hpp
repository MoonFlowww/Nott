#ifndef Nott_DATA_TRANSFORM_AUGMENTATION_CUTOUT_HPP
#define Nott_DATA_TRANSFORM_AUGMENTATION_CUTOUT_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <random>
#include <utility>
#include <vector>


#include <torch/torch.h>

#include "common.hpp"

namespace Nott::Data::Transform::Augmentation {
    namespace Options {
        struct CutoutOptions {
            const std::vector<int64_t>& offsets;
            const std::vector<int64_t>& sizes;
            const std::vector<int64_t> fill_value{-1, -1, -1};
            std::optional<double> frequency = 0.3;
            std::optional<bool> data_augment = true;
            bool show_progress = true;
        };
    }
    inline std::pair<torch::Tensor, torch::Tensor> Cutout(const torch::Tensor& inputs, const torch::Tensor& targets, Options::CutoutOptions opt) {
        [[maybe_unused]] const bool show = opt.show_progress;
        if (!Details::augmentation_enabled(opt.data_augment)) {
            return {inputs, targets};
        }
        if (!inputs.defined() || inputs.dim() < 2) {
            return {inputs, targets};
        }
        if (inputs.size(0) != targets.size(0)) {
            throw std::invalid_argument("Inputs and targets must have matching batch dimensions for Cutout augmentation.");
        }

        auto selected_indices = Details::select_indices_by_frequency(inputs.size(0), opt.frequency, inputs.device());
        if (selected_indices.numel() == 0) {
            return {inputs, targets};
        }

        auto selected_inputs = inputs.index_select(0, selected_indices).clone();
        auto target_indices = selected_indices.device() == targets.device() ? selected_indices : selected_indices.to(targets.device());
        auto selected_targets = targets.index_select(0, target_indices).clone();

        auto result = selected_inputs.clone();

        const auto height_dim = result.dim() - 2;
        const auto width_dim = result.dim() - 1;
        const auto height = result.size(height_dim);
        const auto width = result.size(width_dim);

        auto y0 = opt.offsets.size() > 0 ? opt.offsets[0] : int64_t{0};
        auto x0 = opt.offsets.size() > 1 ? opt.offsets[1] : int64_t{0};
        auto h  = opt.sizes.size()   > 0 ? opt.sizes[0]   : height;
        auto w  = opt.sizes.size()   > 1 ? opt.sizes[1]   : width;

        y0 = std::clamp<int64_t>(y0, 0, height);
        x0 = std::clamp<int64_t>(x0, 0, width);
        h  = std::clamp<int64_t>(h,  0, height);
        w  = std::clamp<int64_t>(w,  0, width);

        static thread_local std::mt19937 rng{std::random_device{}()};

        const bool random_y = opt.offsets.size() > 0 && opt.offsets[0] < 0 && height > 0;
        const bool random_x = opt.offsets.size() > 1 && opt.offsets[1] < 0 && width > 0;

        if (random_y) {
            std::uniform_int_distribution<int64_t> distribution(0, height - 1);
            y0 = distribution(rng);
        }
        if (random_x) {
            std::uniform_int_distribution<int64_t> distribution(0, width - 1);
            x0 = distribution(rng);
        }

        const bool requires_toroidal = random_y || random_x;

        const auto batch_size = result.size(0);

        auto random_color_like = [](const torch::Tensor& tensor, int64_t height_dimension, int64_t width_dimension, const std::vector<int64_t>& fill ) {
            auto color_shape = tensor.sizes().vec();

            if (height_dimension >= 0 && height_dimension < static_cast<int64_t>(color_shape.size())) {
                color_shape[static_cast<std::size_t>(height_dimension)] = 1;
            }
            if (width_dimension >= 0 && width_dimension < static_cast<int64_t>(color_shape.size())) {
                color_shape[static_cast<std::size_t>(width_dimension)] = 1;
            }
            auto color = torch::rand(color_shape, tensor.options());

            const int64_t channel_dim = static_cast<int64_t>(color.dim()) - 1;
            const int64_t n_channels  = color.size(channel_dim);

            for (int64_t c = 0; c < static_cast<int64_t>(fill.size()) && c < n_channels; ++c) {
                const auto v = fill[c];
                if (v >= 0) {
                    /// Map 0-255 to [0,1] for floating types, or keep as is otherwise
                    double val;
                    if (tensor.is_floating_point()) {
                        val = static_cast<double>(v) / 255.0;
                    } else {
                        val = static_cast<double>(v);
                    }
                    color.select(channel_dim, c).fill_(val);
                }
            }
            return color.expand(tensor.sizes());
        };

        std::uniform_int_distribution<int64_t> y_distribution(0, std::max<int64_t>(height - 1, 0));
        std::uniform_int_distribution<int64_t> x_distribution(0, std::max<int64_t>(width - 1, 0));

        for (int64_t sample_idx = 0; sample_idx < batch_size; ++sample_idx) {
            auto sample = result.select(0, sample_idx);
            const auto sample_height_dim = sample.dim() - 2;
            const auto sample_width_dim = sample.dim() - 1;

            auto sample_y0 = random_y && height > 0 ? y_distribution(rng) : y0;
            auto sample_x0 = random_x && width > 0 ? x_distribution(rng) : x0;

            if (!requires_toroidal) {
                const auto y1 = std::min<int64_t>(height, sample_y0 + h);
                const auto x1 = std::min<int64_t>(width,  sample_x0 + w);

                if (sample_y0 < y1 && sample_x0 < x1) {
                    auto patch = sample.slice(sample_height_dim, sample_y0, y1).slice(sample_width_dim, sample_x0, x1);
                    auto color = random_color_like(patch, sample_height_dim, sample_width_dim, opt.fill_value);
                    patch.copy_(color);
                }
                continue;
            }
            auto effective_h = std::min<int64_t>(h, height);
            auto effective_w = std::min<int64_t>(w, width);

            if (!random_y) {
                effective_h = std::min<int64_t>(effective_h, height - sample_y0);
            }
            if (!random_x) {
                effective_w = std::min<int64_t>(effective_w, width - sample_x0);
            }

            if (effective_h <= 0 || effective_w <= 0) {
                continue;
            }

            std::vector<int64_t> y_coords(static_cast<std::size_t>(effective_h));
            std::vector<int64_t> x_coords(static_cast<std::size_t>(effective_w));

            for (int64_t i = 0; i < effective_h; ++i) {
                auto pos = sample_y0 + i;
                if (random_y) {
                    pos %= height;
                    if (pos < 0) {
                        pos += height;
                    }
                }
                y_coords[static_cast<std::size_t>(i)] = pos;
            }
            for (int64_t i = 0; i < effective_w; ++i) {
                auto pos = sample_x0 + i;
                if (random_x) {
                    pos %= width;
                    if (pos < 0) {
                        pos += width;
                    }

                }
                x_coords[static_cast<std::size_t>(i)] = pos;
            }
            auto options = torch::TensorOptions().dtype(torch::kLong).device(sample.device());
            auto options_bool = torch::TensorOptions().dtype(torch::kBool).device(sample.device());
            auto mask = torch::zeros({height, width}, options_bool);
            auto y_tensor = torch::tensor(y_coords, options);
            auto x_tensor = torch::tensor(x_coords, options);;

            if (y_tensor.numel() == 0 || x_tensor.numel() == 0) {
                continue;
            }

            mask.index_put_({y_tensor.unsqueeze(-1), x_tensor.unsqueeze(0)}, true);

            auto mask_view_shape = std::vector<int64_t>(sample.dim(), 1);
            mask_view_shape[static_cast<std::size_t>(sample_height_dim)] = height;
            mask_view_shape[static_cast<std::size_t>(sample_width_dim)] = width;

            auto mask_expanded = mask.view(mask_view_shape).expand(sample.sizes());


            auto color = random_color_like(sample, sample_height_dim, sample_width_dim, opt.fill_value);
            auto updated = torch::where(mask_expanded, color, sample);
            sample.copy_(updated);
        }

        auto augmented_inputs = torch::cat({inputs, result}, 0);
        auto augmented_targets = torch::cat({targets, selected_targets}, 0);
        return {std::move(augmented_inputs), std::move(augmented_targets)};
    }
}

#endif // Nott_DATA_TRANSFORMS_AUGMENTATION_CUTOUT_HPP