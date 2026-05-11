#ifndef Nott_LAYER_RESIZING_HPP
#define Nott_LAYER_RESIZING_HPP

#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#include <torch/torch.h>
#include <torch/nn/functional.h>

#include "../../activation/activation.hpp"
#include "../../common/local.hpp"
#include "../registry.hpp"

namespace Nott {
    enum class UpsampleMode {
        Nearest,
        Bilinear,
        Bicubic,
    };

    enum class DownsampleMode {
        Nearest,
        Bilinear,
        Bicubic,
    };
}

namespace Nott::Layer::Details {

    inline torch::nn::functional::InterpolateFuncOptions::mode_t to_interpolate_mode(UpsampleMode mode)
    {
        switch (mode) {
            case UpsampleMode::Bilinear: return torch::kBilinear;
            case UpsampleMode::Bicubic: return torch::kBicubic;
            case UpsampleMode::Nearest:
            default: return torch::kNearest;
        }
    }

    inline torch::nn::functional::InterpolateFuncOptions::mode_t to_interpolate_mode(DownsampleMode mode)
    {
        switch (mode) {
            case DownsampleMode::Bilinear: return torch::kBilinear;
            case DownsampleMode::Bicubic: return torch::kBicubic;
            case DownsampleMode::Nearest:
            default: return torch::kNearest;
        }
    }

    struct UpsampleOptions {
        std::vector<double> scale{2.0, 2.0};
        UpsampleMode mode{UpsampleMode::Nearest};
        bool align_corners{false};
        bool recompute_scale_factor{false};
    };

    struct DownsampleOptions {
        std::vector<double> scale{2.0, 2.0};
        DownsampleMode mode{DownsampleMode::Nearest};
        bool align_corners{false};
        bool recompute_scale_factor{false};
    };

    class UpsampleImpl : public torch::nn::Module {
    public:
        UpsampleImpl() = default;

        explicit UpsampleImpl(UpsampleOptions options)
        {
            reset(std::move(options));
        }

        void reset(UpsampleOptions options)
        {
            options_ = std::move(options);
        }

        torch::Tensor forward(torch::Tensor input)
        {
            using namespace torch::nn::functional;
            InterpolateFuncOptions opts;
            opts = opts.mode(to_interpolate_mode(options_.mode));
            if (options_.mode != UpsampleMode::Nearest) {
                opts = opts.align_corners(options_.align_corners);
            }
            opts = opts.recompute_scale_factor(options_.recompute_scale_factor);
            opts = opts.scale_factor(options_.scale);
            return interpolate(std::move(input), opts);
        }

        [[nodiscard]] const UpsampleOptions& options() const noexcept { return options_; }

    private:
        UpsampleOptions options_{};
    };

    TORCH_MODULE(Upsample);

    class DownsampleImpl : public torch::nn::Module {
    public:
        DownsampleImpl() = default;

        explicit DownsampleImpl(DownsampleOptions options)
        {
            reset(std::move(options));
        }

        void reset(DownsampleOptions options)
        {
            options_ = std::move(options);
        }

        torch::Tensor forward(torch::Tensor input)
        {
            using namespace torch::nn::functional;
            InterpolateFuncOptions opts;
            opts = opts.mode(to_interpolate_mode(options_.mode));
            if (options_.mode != DownsampleMode::Nearest) {
                opts = opts.align_corners(options_.align_corners);
            }
            opts = opts.recompute_scale_factor(options_.recompute_scale_factor);
            opts = opts.scale_factor(compute_scale_factors());
            return interpolate(std::move(input), opts);
        }

        [[nodiscard]] const DownsampleOptions& options() const noexcept { return options_; }

    private:
        [[nodiscard]] std::vector<double> compute_scale_factors() const
        {
            std::vector<double> factors = options_.scale;
            if (factors.empty()) {
                throw std::invalid_argument("Downsample expects at least one scale factor.");
            }
            for (auto& value : factors) {
                if (value <= 0.0) {
                    throw std::invalid_argument("Downsample scale factors must be positive.");
                }
                value = 1.0 / value;
            }
            return factors;
        }

        DownsampleOptions options_{};
    };

    TORCH_MODULE(Downsample);

    struct UpsampleDescriptor {
        UpsampleOptions options{};
        ::Nott::Activation::Descriptor activation{::Nott::Activation::Identity};
        ::Nott::LocalConfig local{};
    };

    struct DownsampleDescriptor {
        DownsampleOptions options{};
        ::Nott::Activation::Descriptor activation{::Nott::Activation::Identity};
        ::Nott::LocalConfig local{};
    };

    template <class Owner>
    RegisteredLayer build_registered_layer(Owner& owner, const UpsampleDescriptor& descriptor, std::size_t index)
    {
        auto module = owner.register_module("upsample_" + std::to_string(index), Upsample(descriptor.options));

        RegisteredLayer registered_layer{};
        registered_layer.activation = descriptor.activation.type;
        registered_layer.module = to_shared_module_ptr(module);
        registered_layer.local = descriptor.local;
        registered_layer.bind_module_forward(module.get());
        return registered_layer;
    }

    template <class Owner>
    RegisteredLayer build_registered_layer(Owner& owner, const DownsampleDescriptor& descriptor, std::size_t index)
    {
        auto module = owner.register_module("downsample_" + std::to_string(index), Downsample(descriptor.options));

        RegisteredLayer registered_layer{};
        registered_layer.activation = descriptor.activation.type;
        registered_layer.module = to_shared_module_ptr(module);
        registered_layer.local = descriptor.local;
        registered_layer.bind_module_forward(module.get());
        return registered_layer;
    }
}

#endif // Nott_LAYER_RESIZING_HPP