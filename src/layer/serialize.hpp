#ifndef Nott_LAYER_SERIALIZE_HPP
#define Nott_LAYER_SERIALIZE_HPP
/// Serialization for Layer::Descriptor. Adding a layer means adding its
/// name/write/read triple here and one line to the variant in layer.hpp.
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>

#include "../activation/serialize.hpp"
#include "../common/local_serialize.hpp"
#include "../common/serialize.hpp"
#include "../initialization/serialize.hpp"
#include "layer.hpp"

namespace Nott::Layer::Details {
    using ::Nott::Serialize::PropertyTree;
    using ::Nott::Serialize::Tag;

    inline std::string soft_dropout_noise_type_to_string(SoftDropoutOptions::NoiseType type)
    {
        using NoiseType = SoftDropoutOptions::NoiseType;
        switch (type) {
            case NoiseType::Gaussian: return "gaussian";
            case NoiseType::Poisson: return "poisson";
            case NoiseType::Dithering: return "dithering";
            case NoiseType::InterleavedGradientNoise: return "interleaved_gradient_noise";
            case NoiseType::BlueNoise: return "blue_noise";
            case NoiseType::Bayer: return "bayer";
        }
        throw std::runtime_error("Unsupported SoftDropout noise type during serialisation.");
    }

    inline SoftDropoutOptions::NoiseType soft_dropout_noise_type_from_string(const std::string& value)
    {
        using NoiseType = SoftDropoutOptions::NoiseType;
        const auto lowered = ::Nott::Serialize::to_lower(value);
        if (lowered == "gaussian") return NoiseType::Gaussian;
        if (lowered == "poisson") return NoiseType::Poisson;
        if (lowered == "dithering") return NoiseType::Dithering;
        if (lowered == "interleaved_gradient_noise") return NoiseType::InterleavedGradientNoise;
        if (lowered == "blue_noise") return NoiseType::BlueNoise;
        if (lowered == "bayer") return NoiseType::Bayer;
        std::ostringstream message;
        message << "Unknown SoftDropout noise type '" << value << "'.";
        throw std::runtime_error(message.str());
    }

    inline std::string reduce_op_to_string(ReduceOp op)
    {
        switch (op) {
            case ReduceOp::Sum: return "sum";
            case ReduceOp::Mean: return "mean";
            case ReduceOp::Max: return "max";
            case ReduceOp::Min: return "min";
        }
        throw std::runtime_error("Unsupported reduce operation during serialisation.");
    }

    inline ReduceOp reduce_op_from_string(const std::string& value)
    {
        const auto lowered = ::Nott::Serialize::to_lower(value);
        if (lowered == "sum") return ReduceOp::Sum;
        if (lowered == "mean") return ReduceOp::Mean;
        if (lowered == "max") return ReduceOp::Max;
        if (lowered == "min") return ReduceOp::Min;
        std::ostringstream message;
        message << "Unknown reduce operation '" << value << "'.";
        throw std::runtime_error(message.str());
    }

    inline std::string s4_initialization_to_string(S4Initialization initialization)
    {
        switch (initialization) {
            case S4Initialization::HiPPO: return "hippo";
            case S4Initialization::S4D: return "s4d";
        }
        throw std::runtime_error("Unsupported S4 initialization during serialisation.");
    }

    inline S4Initialization s4_initialization_from_string(const std::string& value)
    {
        const auto lowered = ::Nott::Serialize::to_lower(value);
        if (lowered == "hippo") return S4Initialization::HiPPO;
        if (lowered == "s4d") return S4Initialization::S4D;
        std::ostringstream message;
        message << "Unknown S4 initialization '" << value << "'.";
        throw std::runtime_error(message.str());
    }

    inline std::string pooling_variant_to_string(const PoolingOptions& options)
    {
        return std::visit(
            [](const auto& concrete) -> std::string {
                using OptionType = std::decay_t<decltype(concrete)>;
                if constexpr (std::is_same_v<OptionType, MaxPool1dOptions>) {
                    return "max1d";
                } else if constexpr (std::is_same_v<OptionType, AvgPool1dOptions>) {
                    return "avg1d";
                } else if constexpr (std::is_same_v<OptionType, AdaptiveAvgPool1dOptions>) {
                    return "adaptive_avg1d";
                } else if constexpr (std::is_same_v<OptionType, AdaptiveMaxPool1dOptions>) {
                    return "adaptive_max1d";
                } else if constexpr (std::is_same_v<OptionType, MaxPool2dOptions>) {
                    return "max2d";
                } else if constexpr (std::is_same_v<OptionType, AvgPool2dOptions>) {
                    return "avg2d";
                } else if constexpr (std::is_same_v<OptionType, AdaptiveAvgPool2dOptions>) {
                    return "adaptive_avg2d";
                } else if constexpr (std::is_same_v<OptionType, AdaptiveMaxPool2dOptions>) {
                    return "adaptive_max2d";
                } else {
                    static_assert(sizeof(OptionType) == 0, "Unsupported pooling options variant.");
                }
            },
            options);
    }

    inline PoolingOptions pooling_variant_from_string(const std::string& value)
    {
        const auto lowered = ::Nott::Serialize::to_lower(value);
        if (lowered == "max1d") return PoolingOptions{MaxPool1dOptions{}};
        if (lowered == "avg1d") return PoolingOptions{AvgPool1dOptions{}};
        if (lowered == "adaptive_avg1d") return PoolingOptions{AdaptiveAvgPool1dOptions{}};
        if (lowered == "adaptive_max1d") return PoolingOptions{AdaptiveMaxPool1dOptions{}};
        if (lowered == "max2d") return PoolingOptions{MaxPool2dOptions{}};
        if (lowered == "avg2d") return PoolingOptions{AvgPool2dOptions{}};
        if (lowered == "adaptive_avg2d") return PoolingOptions{AdaptiveAvgPool2dOptions{}};
        if (lowered == "adaptive_max2d") return PoolingOptions{AdaptiveMaxPool2dOptions{}};
        std::ostringstream message;
        message << "Unknown pooling variant '" << value << "'.";
        throw std::runtime_error(message.str());
    }

    inline std::string upsample_mode_to_string(::Nott::UpsampleMode mode)
    {
        switch (mode) {
            case ::Nott::UpsampleMode::Bilinear: return "bilinear";
            case ::Nott::UpsampleMode::Bicubic: return "bicubic";
            case ::Nott::UpsampleMode::Nearest:
            default: return "nearest";
        }
    }

    inline ::Nott::UpsampleMode upsample_mode_from_string(const std::string& value, const std::string& context)
    {
        const auto lowered = ::Nott::Serialize::to_lower(value);
        if (lowered == "nearest") return ::Nott::UpsampleMode::Nearest;
        if (lowered == "bilinear") return ::Nott::UpsampleMode::Bilinear;
        if (lowered == "bicubic") return ::Nott::UpsampleMode::Bicubic;
        std::ostringstream message;
        message << "Unknown upsample mode '" << value << "' in " << context;
        throw std::runtime_error(message.str());
    }

    inline std::string downsample_mode_to_string(::Nott::DownsampleMode mode)
    {
        switch (mode) {
            case ::Nott::DownsampleMode::Bilinear: return "bilinear";
            case ::Nott::DownsampleMode::Bicubic: return "bicubic";
            case ::Nott::DownsampleMode::Nearest:
            default: return "nearest";
        }
    }

    inline ::Nott::DownsampleMode downsample_mode_from_string(const std::string& value, const std::string& context)
    {
        const auto lowered = ::Nott::Serialize::to_lower(value);
        if (lowered == "nearest") return ::Nott::DownsampleMode::Nearest;
        if (lowered == "bilinear") return ::Nott::DownsampleMode::Bilinear;
        if (lowered == "bicubic") return ::Nott::DownsampleMode::Bicubic;
        std::ostringstream message;
        message << "Unknown downsample mode '" << value << "' in " << context;
        throw std::runtime_error(message.str());
    }

    /// Every layer descriptor ends with activation and local, and the ones that
    /// own weights also carry initialization. Factored so each layer below
    /// writes only the fields that are its own.
    inline void write_trailer(PropertyTree& tree, const ::Nott::Activation::Descriptor& activation,
                              const ::Nott::LocalConfig& local)
    {
        tree.add_child("activation", ::Nott::Activation::serialize_activation_descriptor(activation));
        tree.add_child("local", ::Nott::serialize_local_config(local));
    }

    inline void write_trailer(PropertyTree& tree, const ::Nott::Activation::Descriptor& activation,
                              const ::Nott::Initialization::Descriptor& initialization,
                              const ::Nott::LocalConfig& local)
    {
        tree.add_child("activation", ::Nott::Activation::serialize_activation_descriptor(activation));
        tree.add_child("initialization",
                       ::Nott::Initialization::serialize_initialization_descriptor(initialization));
        tree.add_child("local", ::Nott::serialize_local_config(local));
    }

    template <class Descriptor>
    void read_trailer(Descriptor& descriptor, const PropertyTree& tree, const std::string& context)
    {
        descriptor.activation =
            ::Nott::Activation::deserialize_activation_descriptor(tree.get_child("activation"), context);
        descriptor.local = ::Nott::deserialize_local_config(tree.get_child("local"), context);
    }

    template <class Descriptor>
    void read_trailer_with_initialization(Descriptor& descriptor, const PropertyTree& tree,
                                          const std::string& context)
    {
        read_trailer(descriptor, tree, context);
        descriptor.initialization = ::Nott::Initialization::deserialize_initialization_descriptor(
            tree.get_child("initialization"), context);
    }

    constexpr std::string_view descriptor_type_name(Tag<FCDescriptor>) { return "fc"; }

    inline PropertyTree serialize_descriptor(const FCDescriptor& descriptor)
    {
        PropertyTree tree;
        tree.put("options.in_features", descriptor.options.in_features);
        tree.put("options.out_features", descriptor.options.out_features);
        tree.put("options.bias", descriptor.options.bias);
        write_trailer(tree, descriptor.activation, descriptor.initialization, descriptor.local);
        return tree;
    }

    inline FCDescriptor deserialize_descriptor(Tag<FCDescriptor>, const PropertyTree& tree,
                                               const std::string& context)
    {
        namespace S = ::Nott::Serialize;
        FCDescriptor descriptor;
        descriptor.options.in_features = S::get_numeric<std::int64_t>(tree, "options.in_features", context);
        descriptor.options.out_features = S::get_numeric<std::int64_t>(tree, "options.out_features", context);
        descriptor.options.bias = S::get_boolean(tree, "options.bias", context);
        read_trailer_with_initialization(descriptor, tree, context);
        return descriptor;
    }

/// Conv1d and Conv2d serialize identically; only the name differs.
#define Nott_LAYER_CONVOLUTION(DescriptorType, name)                                                     \
    constexpr std::string_view descriptor_type_name(Tag<DescriptorType>) { return name; }                \
    inline PropertyTree serialize_descriptor(const DescriptorType& descriptor)                           \
    {                                                                                                    \
        namespace S = ::Nott::Serialize;                                                                 \
        const auto& options = descriptor.options;                                                        \
        PropertyTree tree;                                                                               \
        tree.put("options.in_channels", options.in_channels);                                            \
        tree.put("options.out_channels", options.out_channels);                                          \
        tree.add_child("options.kernel_size", S::write_array(options.kernel_size));                      \
        tree.add_child("options.stride", S::write_array(options.stride));                                \
        tree.add_child("options.padding", S::write_array(options.padding));                              \
        tree.add_child("options.dilation", S::write_array(options.dilation));                            \
        tree.put("options.groups", options.groups);                                                      \
        tree.put("options.bias", options.bias);                                                          \
        tree.put("options.padding_mode", options.padding_mode);                                          \
        write_trailer(tree, descriptor.activation, descriptor.initialization, descriptor.local);         \
        return tree;                                                                                     \
    }                                                                                                    \
    inline DescriptorType deserialize_descriptor(Tag<DescriptorType>, const PropertyTree& tree,          \
                                                 const std::string& context)                             \
    {                                                                                                    \
        namespace S = ::Nott::Serialize;                                                                 \
        DescriptorType descriptor;                                                                       \
        auto& options = descriptor.options;                                                              \
        options.in_channels = S::get_numeric<std::int64_t>(tree, "options.in_channels", context);        \
        options.out_channels = S::get_numeric<std::int64_t>(tree, "options.out_channels", context);      \
        options.kernel_size = S::read_array<std::int64_t>(tree.get_child("options.kernel_size"), context); \
        options.stride = S::read_array<std::int64_t>(tree.get_child("options.stride"), context);         \
        options.padding = S::read_array<std::int64_t>(tree.get_child("options.padding"), context);       \
        options.dilation = S::read_array<std::int64_t>(tree.get_child("options.dilation"), context);     \
        options.groups = S::get_numeric<std::int64_t>(tree, "options.groups", context);                  \
        options.bias = S::get_boolean(tree, "options.bias", context);                                    \
        options.padding_mode = S::get_string(tree, "options.padding_mode", context);                     \
        read_trailer_with_initialization(descriptor, tree, context);                                     \
        return descriptor;                                                                               \
    }

    Nott_LAYER_CONVOLUTION(Conv1dDescriptor, "conv1d")
    Nott_LAYER_CONVOLUTION(Conv2dDescriptor, "conv2d")

#undef Nott_LAYER_CONVOLUTION

    constexpr std::string_view descriptor_type_name(Tag<BatchNorm2dDescriptor>) { return "batch_norm2d"; }

    inline PropertyTree serialize_descriptor(const BatchNorm2dDescriptor& descriptor)
    {
        const auto& options = descriptor.options;
        PropertyTree tree;
        tree.put("options.num_features", options.num_features);
        tree.put("options.eps", options.eps);
        tree.put("options.momentum", options.momentum);
        tree.put("options.affine", options.affine);
        tree.put("options.track_running_stats", options.track_running_stats);
        write_trailer(tree, descriptor.activation, descriptor.initialization, descriptor.local);
        return tree;
    }

    inline BatchNorm2dDescriptor deserialize_descriptor(Tag<BatchNorm2dDescriptor>, const PropertyTree& tree,
                                                        const std::string& context)
    {
        namespace S = ::Nott::Serialize;
        BatchNorm2dDescriptor descriptor;
        auto& options = descriptor.options;
        options.num_features = S::get_numeric<std::int64_t>(tree, "options.num_features", context);
        options.eps = S::get_numeric<double>(tree, "options.eps", context);
        options.momentum = S::get_numeric<double>(tree, "options.momentum", context);
        options.affine = S::get_boolean(tree, "options.affine", context);
        options.track_running_stats = S::get_boolean(tree, "options.track_running_stats", context);
        read_trailer_with_initialization(descriptor, tree, context);
        return descriptor;
    }

    constexpr std::string_view descriptor_type_name(Tag<PoolingDescriptor>) { return "pooling"; }

    inline PropertyTree serialize_descriptor(const PoolingDescriptor& descriptor)
    {
        namespace S = ::Nott::Serialize;
        PropertyTree tree;
        tree.put("options.variant", pooling_variant_to_string(descriptor.options));
        std::visit(
            [&](const auto& options) {
                using OptionType = std::decay_t<decltype(options)>;
                if constexpr (std::is_same_v<OptionType, MaxPool1dOptions> ||
                              std::is_same_v<OptionType, MaxPool2dOptions>) {
                    tree.add_child("options.kernel_size", S::write_array(options.kernel_size));
                    tree.add_child("options.stride", S::write_array(options.stride));
                    tree.add_child("options.padding", S::write_array(options.padding));
                    tree.add_child("options.dilation", S::write_array(options.dilation));
                    tree.put("options.ceil_mode", options.ceil_mode);
                } else if constexpr (std::is_same_v<OptionType, AvgPool1dOptions> ||
                                     std::is_same_v<OptionType, AvgPool2dOptions>) {
                    tree.add_child("options.kernel_size", S::write_array(options.kernel_size));
                    tree.add_child("options.stride", S::write_array(options.stride));
                    tree.add_child("options.padding", S::write_array(options.padding));
                    tree.put("options.ceil_mode", options.ceil_mode);
                    tree.put("options.count_include_pad", options.count_include_pad);
                } else {
                    tree.add_child("options.output_size", S::write_array(options.output_size));
                }
            },
            descriptor.options);
        write_trailer(tree, descriptor.activation, descriptor.local);
        return tree;
    }

    inline PoolingDescriptor deserialize_descriptor(Tag<PoolingDescriptor>, const PropertyTree& tree,
                                                     const std::string& context)
    {
        namespace S = ::Nott::Serialize;
        PoolingDescriptor descriptor;
        descriptor.options = pooling_variant_from_string(S::get_string(tree, "options.variant", context));
        read_trailer(descriptor, tree, context);
        std::visit(
            [&](auto& options) {
                using OptionType = std::decay_t<decltype(options)>;
                if constexpr (std::is_same_v<OptionType, MaxPool1dOptions> ||
                              std::is_same_v<OptionType, MaxPool2dOptions>) {
                    options.kernel_size = S::read_array<std::int64_t>(tree.get_child("options.kernel_size"), context);
                    options.stride = S::read_array<std::int64_t>(tree.get_child("options.stride"), context);
                    options.padding = S::read_array<std::int64_t>(tree.get_child("options.padding"), context);
                    options.dilation = S::read_array<std::int64_t>(tree.get_child("options.dilation"), context);
                    options.ceil_mode = S::get_boolean(tree, "options.ceil_mode", context);
                } else if constexpr (std::is_same_v<OptionType, AvgPool1dOptions> ||
                                     std::is_same_v<OptionType, AvgPool2dOptions>) {
                    options.kernel_size = S::read_array<std::int64_t>(tree.get_child("options.kernel_size"), context);
                    options.stride = S::read_array<std::int64_t>(tree.get_child("options.stride"), context);
                    options.padding = S::read_array<std::int64_t>(tree.get_child("options.padding"), context);
                    options.ceil_mode = S::get_boolean(tree, "options.ceil_mode", context);
                    options.count_include_pad = S::get_boolean(tree, "options.count_include_pad", context);
                } else {
                    options.output_size = S::read_array<std::int64_t>(tree.get_child("options.output_size"), context);
                }
            },
            descriptor.options);
        return descriptor;
    }

    constexpr std::string_view descriptor_type_name(Tag<HardDropoutDescriptor>) { return "harddropout"; }

    inline PropertyTree serialize_descriptor(const HardDropoutDescriptor& descriptor)
    {
        PropertyTree tree;
        tree.put("options.probability", descriptor.options.probability);
        tree.put("options.inplace", descriptor.options.inplace);
        write_trailer(tree, descriptor.activation, descriptor.local);
        return tree;
    }

    inline HardDropoutDescriptor deserialize_descriptor(Tag<HardDropoutDescriptor>, const PropertyTree& tree,
                                                         const std::string& context)
    {
        namespace S = ::Nott::Serialize;
        HardDropoutDescriptor descriptor;
        descriptor.options.probability = S::get_numeric<double>(tree, "options.probability", context);
        descriptor.options.inplace = S::get_boolean(tree, "options.inplace", context);
        read_trailer(descriptor, tree, context);
        return descriptor;
    }

    constexpr std::string_view descriptor_type_name(Tag<SoftDropoutDescriptor>) { return "softdropout"; }

    inline PropertyTree serialize_descriptor(const SoftDropoutDescriptor& descriptor)
    {
        const auto& options = descriptor.options;
        PropertyTree tree;
        tree.put("options.probability", options.probability);
        tree.put("options.inplace", options.inplace);
        tree.put("options.noise_mean", options.noise_mean);
        tree.put("options.noise_std", options.noise_std);
        tree.put("options.noise_type", soft_dropout_noise_type_to_string(options.noise_type));
        write_trailer(tree, descriptor.activation, descriptor.local);
        return tree;
    }

    inline SoftDropoutDescriptor deserialize_descriptor(Tag<SoftDropoutDescriptor>, const PropertyTree& tree,
                                                         const std::string& context)
    {
        namespace S = ::Nott::Serialize;
        SoftDropoutDescriptor descriptor;
        auto& options = descriptor.options;
        options.probability = S::get_numeric<double>(tree, "options.probability", context);
        options.inplace = S::get_boolean(tree, "options.inplace", context);
        options.noise_mean = S::get_numeric<double>(tree, "options.noise_mean", context);
        options.noise_std = S::get_numeric<double>(tree, "options.noise_std", context);
        if (const auto noise_type = tree.get_optional<std::string>("options.noise_type")) {
            options.noise_type = soft_dropout_noise_type_from_string(*noise_type);
        }
        read_trailer(descriptor, tree, context);
        return descriptor;
    }

    constexpr std::string_view descriptor_type_name(Tag<FlattenDescriptor>) { return "flatten"; }

    inline PropertyTree serialize_descriptor(const FlattenDescriptor& descriptor)
    {
        PropertyTree tree;
        tree.put("options.start_dim", descriptor.options.start_dim);
        tree.put("options.end_dim", descriptor.options.end_dim);
        write_trailer(tree, descriptor.activation, descriptor.local);
        return tree;
    }

    inline FlattenDescriptor deserialize_descriptor(Tag<FlattenDescriptor>, const PropertyTree& tree,
                                                     const std::string& context)
    {
        namespace S = ::Nott::Serialize;
        FlattenDescriptor descriptor;
        descriptor.options.start_dim = S::get_numeric<std::int64_t>(tree, "options.start_dim", context);
        descriptor.options.end_dim = S::get_numeric<std::int64_t>(tree, "options.end_dim", context);
        read_trailer(descriptor, tree, context);
        return descriptor;
    }

    constexpr std::string_view descriptor_type_name(Tag<UpsampleDescriptor>) { return "upsample"; }

    inline PropertyTree serialize_descriptor(const UpsampleDescriptor& descriptor)
    {
        const auto& options = descriptor.options;
        PropertyTree tree;
        tree.add_child("options.scale", ::Nott::Serialize::write_array(options.scale));
        tree.put("options.mode", upsample_mode_to_string(options.mode));
        tree.put("options.align_corners", options.align_corners);
        tree.put("options.recompute_scale_factor", options.recompute_scale_factor);
        write_trailer(tree, descriptor.activation, descriptor.local);
        return tree;
    }

    inline UpsampleDescriptor deserialize_descriptor(Tag<UpsampleDescriptor>, const PropertyTree& tree,
                                                      const std::string& context)
    {
        namespace S = ::Nott::Serialize;
        UpsampleDescriptor descriptor;
        auto& options = descriptor.options;
        options.scale = S::read_array<double>(tree.get_child("options.scale"), context);
        options.mode = upsample_mode_from_string(S::get_string(tree, "options.mode", context), context);
        options.align_corners = S::get_boolean(tree, "options.align_corners", context);
        options.recompute_scale_factor = S::get_boolean(tree, "options.recompute_scale_factor", context);
        read_trailer(descriptor, tree, context);
        return descriptor;
    }

    constexpr std::string_view descriptor_type_name(Tag<DownsampleDescriptor>) { return "downsample"; }

    inline PropertyTree serialize_descriptor(const DownsampleDescriptor& descriptor)
    {
        const auto& options = descriptor.options;
        PropertyTree tree;
        tree.add_child("options.scale", ::Nott::Serialize::write_array(options.scale));
        tree.put("options.mode", downsample_mode_to_string(options.mode));
        tree.put("options.align_corners", options.align_corners);
        tree.put("options.recompute_scale_factor", options.recompute_scale_factor);
        write_trailer(tree, descriptor.activation, descriptor.local);
        return tree;
    }

    inline DownsampleDescriptor deserialize_descriptor(Tag<DownsampleDescriptor>, const PropertyTree& tree,
                                                        const std::string& context)
    {
        namespace S = ::Nott::Serialize;
        DownsampleDescriptor descriptor;
        auto& options = descriptor.options;
        options.scale = S::read_array<double>(tree.get_child("options.scale"), context);
        options.mode = downsample_mode_from_string(S::get_string(tree, "options.mode", context), context);
        options.align_corners = S::get_boolean(tree, "options.align_corners", context);
        options.recompute_scale_factor = S::get_boolean(tree, "options.recompute_scale_factor", context);
        read_trailer(descriptor, tree, context);
        return descriptor;
    }

    constexpr std::string_view descriptor_type_name(Tag<ReduceDescriptor>) { return "reduce"; }

    inline PropertyTree serialize_descriptor(const ReduceDescriptor& descriptor)
    {
        const auto& options = descriptor.options;
        PropertyTree tree;
        tree.put("options.op", reduce_op_to_string(options.op));
        tree.add_child("options.dims", ::Nott::Serialize::write_array(options.dims));
        tree.put("options.keep_dim", options.keep_dim);
        write_trailer(tree, descriptor.activation, descriptor.local);
        return tree;
    }

    inline ReduceDescriptor deserialize_descriptor(Tag<ReduceDescriptor>, const PropertyTree& tree,
                                                    const std::string& context)
    {
        namespace S = ::Nott::Serialize;
        ReduceDescriptor descriptor;
        descriptor.options.op = reduce_op_from_string(S::get_string(tree, "options.op", context));
        if (const auto dims = tree.get_child_optional("options.dims")) {
            descriptor.options.dims = S::read_array<std::int64_t>(*dims, context);
        }
        descriptor.options.keep_dim = S::get_boolean(tree, "options.keep_dim", context);
        read_trailer(descriptor, tree, context);
        return descriptor;
    }

    constexpr std::string_view descriptor_type_name(Tag<RNNDescriptor>) { return "rnn"; }

    inline PropertyTree serialize_descriptor(const RNNDescriptor& descriptor)
    {
        const auto& options = descriptor.options;
        PropertyTree tree;
        tree.put("options.input_size", options.input_size);
        tree.put("options.hidden_size", options.hidden_size);
        tree.put("options.num_layers", options.num_layers);
        tree.put("options.dropout", options.dropout);
        tree.put("options.batch_first", options.batch_first);
        tree.put("options.bidirectional", options.bidirectional);
        tree.put("options.nonlinearity", options.nonlinearity);
        write_trailer(tree, descriptor.activation, descriptor.initialization, descriptor.local);
        return tree;
    }

    inline RNNDescriptor deserialize_descriptor(Tag<RNNDescriptor>, const PropertyTree& tree,
                                                 const std::string& context)
    {
        namespace S = ::Nott::Serialize;
        RNNDescriptor descriptor;
        auto& options = descriptor.options;
        options.input_size = S::get_numeric<std::int64_t>(tree, "options.input_size", context);
        options.hidden_size = S::get_numeric<std::int64_t>(tree, "options.hidden_size", context);
        options.num_layers = S::get_numeric<std::int64_t>(tree, "options.num_layers", context);
        options.dropout = S::get_numeric<double>(tree, "options.dropout", context);
        options.batch_first = S::get_boolean(tree, "options.batch_first", context);
        options.bidirectional = S::get_boolean(tree, "options.bidirectional", context);
        options.nonlinearity = S::get_string(tree, "options.nonlinearity", context);
        read_trailer_with_initialization(descriptor, tree, context);
        return descriptor;
    }

/// LSTM, xLSTM and GRU carry the same recurrent option set.
#define Nott_LAYER_RECURRENT(DescriptorType, name)                                                       \
    constexpr std::string_view descriptor_type_name(Tag<DescriptorType>) { return name; }                \
    inline PropertyTree serialize_descriptor(const DescriptorType& descriptor)                           \
    {                                                                                                    \
        const auto& options = descriptor.options;                                                        \
        PropertyTree tree;                                                                               \
        tree.put("options.input_size", options.input_size);                                              \
        tree.put("options.hidden_size", options.hidden_size);                                            \
        tree.put("options.num_layers", options.num_layers);                                              \
        tree.put("options.dropout", options.dropout);                                                    \
        tree.put("options.batch_first", options.batch_first);                                            \
        tree.put("options.bidirectional", options.bidirectional);                                        \
        write_trailer(tree, descriptor.activation, descriptor.initialization, descriptor.local);         \
        return tree;                                                                                     \
    }                                                                                                    \
    inline DescriptorType deserialize_descriptor(Tag<DescriptorType>, const PropertyTree& tree,          \
                                                 const std::string& context)                             \
    {                                                                                                    \
        namespace S = ::Nott::Serialize;                                                                 \
        DescriptorType descriptor;                                                                       \
        auto& options = descriptor.options;                                                              \
        options.input_size = S::get_numeric<std::int64_t>(tree, "options.input_size", context);          \
        options.hidden_size = S::get_numeric<std::int64_t>(tree, "options.hidden_size", context);        \
        options.num_layers = S::get_numeric<std::int64_t>(tree, "options.num_layers", context);          \
        options.dropout = S::get_numeric<double>(tree, "options.dropout", context);                      \
        options.batch_first = S::get_boolean(tree, "options.batch_first", context);                      \
        options.bidirectional = S::get_boolean(tree, "options.bidirectional", context);                  \
        read_trailer_with_initialization(descriptor, tree, context);                                     \
        return descriptor;                                                                               \
    }

    Nott_LAYER_RECURRENT(LSTMDescriptor, "lstm")
    Nott_LAYER_RECURRENT(xLSTMDescriptor, "xlstm")
    Nott_LAYER_RECURRENT(GRUDescriptor, "gru")

#undef Nott_LAYER_RECURRENT

    constexpr std::string_view descriptor_type_name(Tag<S4Descriptor>) { return "s4"; }

    inline PropertyTree serialize_descriptor(const S4Descriptor& descriptor)
    {
        const auto& options = descriptor.options;
        PropertyTree tree;
        tree.put("options.input_size", options.input_size);
        tree.put("options.state_size", options.state_size);
        tree.put("options.rank", options.rank);
        tree.put("options.output_size", options.output_size);
        tree.put("options.batch_first", options.batch_first);
        tree.put("options.bidirectional", options.bidirectional);
        tree.put("options.dropout", options.dropout);
        tree.put("options.initialization", s4_initialization_to_string(options.initialization));
        tree.put("options.maximum_length", options.maximum_length);
        write_trailer(tree, descriptor.activation, descriptor.initialization, descriptor.local);
        return tree;
    }

    inline S4Descriptor deserialize_descriptor(Tag<S4Descriptor>, const PropertyTree& tree,
                                                const std::string& context)
    {
        namespace S = ::Nott::Serialize;
        S4Descriptor descriptor;
        auto& options = descriptor.options;
        options.input_size = S::get_numeric<std::int64_t>(tree, "options.input_size", context);
        options.state_size = S::get_numeric<std::int64_t>(tree, "options.state_size", context);
        options.rank = S::get_numeric<std::int64_t>(tree, "options.rank", context);
        options.output_size = S::get_numeric<std::int64_t>(tree, "options.output_size", context);
        options.batch_first = S::get_boolean(tree, "options.batch_first", context);
        options.bidirectional = S::get_boolean(tree, "options.bidirectional", context);
        options.dropout = S::get_numeric<double>(tree, "options.dropout", context);
        options.initialization = s4_initialization_from_string(S::get_string(tree, "options.initialization", context));
        options.maximum_length = S::get_numeric<std::int64_t>(tree, "options.maximum_length", context);
        read_trailer_with_initialization(descriptor, tree, context);
        return descriptor;
    }

    constexpr std::string_view descriptor_type_name(Tag<PatchUnembedDescriptor>) { return "patchunembed"; }

    inline PropertyTree serialize_descriptor(const PatchUnembedDescriptor& descriptor)
    {
        const auto& options = descriptor.options;
        PropertyTree tree;
        tree.put("options.channels", options.channels);
        tree.put("options.tokens_height", options.tokens_height);
        tree.put("options.tokens_width", options.tokens_width);
        tree.put("options.patch_size", options.patch_size);
        tree.put("options.target_height", options.target_height);
        tree.put("options.target_width", options.target_width);
        tree.put("options.align_corners", options.align_corners);
        write_trailer(tree, descriptor.activation, descriptor.local);
        return tree;
    }

    inline PatchUnembedDescriptor deserialize_descriptor(Tag<PatchUnembedDescriptor>, const PropertyTree& tree,
                                                          const std::string& context)
    {
        namespace S = ::Nott::Serialize;
        PatchUnembedDescriptor descriptor;
        auto& options = descriptor.options;
        options.channels = S::get_numeric<std::int64_t>(tree, "options.channels", context);
        options.tokens_height = S::get_numeric<std::int64_t>(tree, "options.tokens_height", context);
        options.tokens_width = S::get_numeric<std::int64_t>(tree, "options.tokens_width", context);
        options.patch_size = S::get_numeric<std::int64_t>(tree, "options.patch_size", context);
        options.target_height = S::get_numeric<std::int64_t>(tree, "options.target_height", context);
        options.target_width = S::get_numeric<std::int64_t>(tree, "options.target_width", context);
        options.align_corners = S::get_boolean(tree, "options.align_corners", context);
        read_trailer(descriptor, tree, context);
        return descriptor;
    }
}

namespace Nott::Layer {
    inline ::Nott::Serialize::PropertyTree serialize_layer_descriptor(const Descriptor& descriptor)
    {
        return ::Nott::Serialize::serialize_descriptor(descriptor);
    }

    inline Descriptor deserialize_layer_descriptor(const ::Nott::Serialize::PropertyTree& tree,
                                                   const std::string& context)
    {
        return ::Nott::Serialize::deserialize_variant<Descriptor>(tree, context);
    }
}

#endif // Nott_LAYER_SERIALIZE_HPP
