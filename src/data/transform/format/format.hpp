#ifndef Nott_DATA_TRANSFORM_FORMAT_HPP
#define Nott_DATA_TRANSFORM_FORMAT_HPP

#include <array>
#include <optional>
#include <stdexcept>
#include <vector>

#include <torch/torch.h>
#include <torch/nn/functional.h>

namespace Nott::Data::Transform::Format {
    namespace Options {

        enum class InterpMode {
            Bilinear,
            Nearest,
            Bicubic,
            Area,
        };

        struct ScaleOptions {
            std::vector<int> size{};
            InterpMode interp = InterpMode::Bilinear;
            bool showprogress = false;
        };

        using UpsampleOptions = ScaleOptions;
        using DownsampleOptions = ScaleOptions;
    }

    namespace Details {

        inline torch::Tensor to_float32(const torch::Tensor& tensor) {
            if (tensor.scalar_type() == torch::kFloat32) {
                return tensor;
            }
            return tensor.to(tensor.options().dtype(torch::kFloat32));
        }

        inline torch::Tensor clone_as_dtype(const torch::Tensor& tensor, torch::Dtype dtype) {
            return tensor.to(tensor.options().dtype(dtype));
        }

        inline torch::Tensor resize_spatial(
                const torch::Tensor& tensor,
                const std::vector<int>& target_size,
                Options::InterpMode interp_mode) {

            if (tensor.dim() < 3)
                throw std::invalid_argument("Format::resize_spatial expects (C,H,W) or (N,C,H,W).");
            if (target_size.size() != 2 || target_size[0] <= 0 || target_size[1] <= 0)
                throw std::invalid_argument("Format::resize_spatial expects positive H,W.");

            auto working = tensor;
            bool added_batch_dim = false;
            if (working.dim() == 3) {
                working = working.unsqueeze(0);
                added_batch_dim = true;
            } else if (working.dim() != 4) {
                throw std::invalid_argument("Format::resize_spatial supports 3D or 4D tensors.");
            }

            auto opts = torch::nn::functional::InterpolateFuncOptions()
                            .size(std::vector<int64_t>{target_size[0], target_size[1]});

            /// Ici on fait le mapping, SANS helper de retour
            switch (interp_mode) {
                case Options::InterpMode::Bilinear:
                    opts = opts.mode(torch::kBilinear).align_corners(false);
                    break;
                case Options::InterpMode::Nearest:
                    opts = opts.mode(torch::kNearest);
                    break;
                case Options::InterpMode::Bicubic:
                    opts = opts.mode(torch::kBicubic).align_corners(false);
                    break;
                case Options::InterpMode::Area:
                    opts = opts.mode(torch::kArea);
                    break;
            }

            auto resized = torch::nn::functional::interpolate(working, opts);
            if (added_batch_dim) {
                resized = resized.squeeze(0);
            }
            return resized;
        }
    }

    inline torch::Tensor Upsample(const torch::Tensor& tensor, Options::UpsampleOptions options = {}) {
        const auto requested_size = options.size;
        [[maybe_unused]] const bool show_progress = options.showprogress;

        if (!tensor.defined()) {
            throw std::invalid_argument("Format::Upsample expects a defined tensor.");
        }
        if (tensor.numel() == 0) {
            return Details::clone_as_dtype(tensor, tensor.scalar_type());
        }

        auto float_tensor = Details::to_float32(tensor);
        if (requested_size[0] != -1 || requested_size[1] != -1) {
            float_tensor = Details::resize_spatial(float_tensor, requested_size, options.interp);
        }

        return Details::clone_as_dtype(float_tensor, tensor.scalar_type());
    }

    inline torch::Tensor Downsample(const torch::Tensor& tensor, Options::DownsampleOptions options = {}) {
        const auto requested_size = options.size;
        [[maybe_unused]] const bool show_progress = options.showprogress;

        if (!tensor.defined()) {
            throw std::invalid_argument("Format::Downsample expects a defined tensor.");
        }
        if (tensor.numel() == 0) {
            return Details::clone_as_dtype(tensor, tensor.scalar_type());
        }

        auto float_tensor = Details::to_float32(tensor);
        if (requested_size[0] != -1 || requested_size[1] != -1) {
            float_tensor = Details::resize_spatial(float_tensor, requested_size, options.interp);
        }

        return Details::clone_as_dtype(float_tensor, tensor.scalar_type());
    }
}

#endif // Nott_DATA_TRANSFORM_FORMAT_HPP
