#ifndef Nott_BLOCK_TRANSFORMERS_SERIALIZE_VISION_HPP
#define Nott_BLOCK_TRANSFORMERS_SERIALIZE_VISION_HPP
/// Serialization for the vision encoder (ViT and Swin).
#include <cstddef>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>

#include "../../../../activation/serialize.hpp"
#include "../../../../attention/serialize.hpp"
#include "../../../../common/serialize.hpp"
#include "../vision.hpp"

namespace Nott::Layer::Details {
    using ::Nott::Serialize::PropertyTree;

    inline std::string positional_encoding_type_to_string(PositionalEncodingType type)
    {
        switch (type) {
            case PositionalEncodingType::None: return "none";
            case PositionalEncodingType::Sinusoidal: return "sinusoidal";
            case PositionalEncodingType::Learned: return "learned";
        }
        throw std::runtime_error("Unsupported positional encoding type during serialisation.");
    }

    inline PositionalEncodingType positional_encoding_type_from_string(const std::string& value,
                                                                       const std::string& context)
    {
        const auto lowered = ::Nott::Serialize::to_lower(value);
        if (lowered == "none") return PositionalEncodingType::None;
        if (lowered == "sinusoidal") return PositionalEncodingType::Sinusoidal;
        if (lowered == "learned") return PositionalEncodingType::Learned;
        std::ostringstream message;
        message << "Unknown positional encoding type '" << value << "' in " << context;
        throw std::runtime_error(message.str());
    }

    inline PropertyTree serialize_positional_encoding_options(const PositionalEncodingOptions& options)
    {
        PropertyTree tree;
        tree.put("type", positional_encoding_type_to_string(options.type));
        tree.put("dropout", options.dropout);
        return tree;
    }

    inline PositionalEncodingOptions deserialize_positional_encoding_options(const PropertyTree& tree,
                                                                             const std::string& context)
    {
        PositionalEncodingOptions options;
        options.type =
            positional_encoding_type_from_string(::Nott::Serialize::get_string(tree, "type", context), context);
        options.dropout = ::Nott::Serialize::get_numeric<double>(tree, "dropout", context);
        return options;
    }
}

namespace Nott::Block::Details::Transformer::Vision {
    using ::Nott::Serialize::PropertyTree;
    using ::Nott::Serialize::Tag;

    inline std::string variant_to_string(Variant variant)
    {
        switch (variant) {
            case Variant::ViT: return "vit";
            case Variant::Swin: return "swin";
        }
        throw std::runtime_error("Unsupported vision variant during serialisation.");
    }

    inline Variant variant_from_string(const std::string& value, const std::string& context)
    {
        const auto lowered = ::Nott::Serialize::to_lower(value);
        if (lowered == "vit") return Variant::ViT;
        if (lowered == "swin") return Variant::Swin;
        std::ostringstream message;
        message << "Unknown vision variant '" << value << "' in " << context;
        throw std::runtime_error(message.str());
    }

    inline PropertyTree serialize_attention_options(const AttentionOptions& options)
    {
        PropertyTree tree;
        tree.put("embed_dim", options.embed_dim);
        tree.put("num_heads", options.num_heads);
        tree.put("dropout", options.dropout);
        tree.put("bias", options.bias);
        tree.put("batch_first", options.batch_first);
        tree.put("variant", ::Nott::Attention::attention_variant_to_string(options.variant));
        return tree;
    }

    inline AttentionOptions deserialize_attention_options(const PropertyTree& tree, const std::string& context)
    {
        namespace S = ::Nott::Serialize;
        AttentionOptions options;
        options.embed_dim = S::get_numeric<std::int64_t>(tree, "embed_dim", context);
        options.num_heads = S::get_numeric<std::int64_t>(tree, "num_heads", context);
        options.dropout = S::get_numeric<double>(tree, "dropout", context);
        options.bias = S::get_boolean(tree, "bias", context);
        options.batch_first = S::get_boolean(tree, "batch_first", context);
        options.variant = ::Nott::Attention::attention_variant_from_string(S::get_string(tree, "variant", context));
        return options;
    }

    inline PropertyTree serialize_feed_forward_options(const FeedForwardOptions& options)
    {
        PropertyTree tree;
        tree.put("embed_dim", options.embed_dim);
        tree.put("mlp_ratio", options.mlp_ratio);
        tree.add_child("activation", ::Nott::Activation::serialize_activation_descriptor(options.activation));
        tree.put("bias", options.bias);
        return tree;
    }

    inline FeedForwardOptions deserialize_feed_forward_options(const PropertyTree& tree, const std::string& context)
    {
        namespace S = ::Nott::Serialize;
        FeedForwardOptions options;
        options.embed_dim = S::get_numeric<std::int64_t>(tree, "embed_dim", context);
        options.mlp_ratio = S::get_numeric<double>(tree, "mlp_ratio", context);
        options.activation = ::Nott::Activation::deserialize_activation_field(tree, "activation", context);
        options.bias = S::get_boolean(tree, "bias", context);
        return options;
    }

    inline PropertyTree serialize_layer_norm_options(const LayerNormOptions& options)
    {
        PropertyTree tree;
        tree.put("eps", options.eps);
        tree.put("elementwise_affine", options.elementwise_affine);
        return tree;
    }

    inline LayerNormOptions deserialize_layer_norm_options(const PropertyTree& tree, const std::string& context)
    {
        LayerNormOptions options;
        options.eps = ::Nott::Serialize::get_numeric<double>(tree, "eps", context);
        options.elementwise_affine = ::Nott::Serialize::get_boolean(tree, "elementwise_affine", context);
        return options;
    }

    inline PropertyTree serialize_patch_embedding_options(const PatchEmbeddingOptions& options)
    {
        PropertyTree tree;
        tree.put("in_channels", options.in_channels);
        tree.put("embed_dim", options.embed_dim);
        tree.put("patch_size", options.patch_size);
        tree.put("add_class_token", options.add_class_token);
        tree.put("normalize", options.normalize);
        tree.put("dropout", options.dropout);
        return tree;
    }

    inline PatchEmbeddingOptions deserialize_patch_embedding_options(const PropertyTree& tree,
                                                                     const std::string& context)
    {
        namespace S = ::Nott::Serialize;
        PatchEmbeddingOptions options;
        options.in_channels = S::get_numeric<std::int64_t>(tree, "in_channels", context);
        options.embed_dim = S::get_numeric<std::int64_t>(tree, "embed_dim", context);
        options.patch_size = S::get_numeric<std::int64_t>(tree, "patch_size", context);
        options.add_class_token = S::get_boolean(tree, "add_class_token", context);
        options.normalize = S::get_boolean(tree, "normalize", context);
        options.dropout = S::get_numeric<double>(tree, "dropout", context);
        return options;
    }

    inline PropertyTree serialize_window_options(const WindowOptions& options)
    {
        PropertyTree tree;
        tree.put("size", options.size);
        tree.put("shift", options.shift);
        return tree;
    }

    inline WindowOptions deserialize_window_options(const PropertyTree& tree, const std::string& context)
    {
        namespace S = ::Nott::Serialize;
        WindowOptions options;
        options.size = S::get_numeric<std::int64_t>(tree, "size", context);
        options.shift = S::get_boolean(tree, "shift", context);
        return options;
    }

    inline PropertyTree serialize_encoder_layer_descriptor(const EncoderLayerDescriptor& layer)
    {
        PropertyTree tree;
        tree.add_child("attention", serialize_attention_options(layer.attention));
        tree.add_child("feed_forward", serialize_feed_forward_options(layer.feed_forward));
        tree.add_child("window", serialize_window_options(layer.window));
        return tree;
    }

    inline EncoderLayerDescriptor deserialize_encoder_layer_descriptor(const PropertyTree& tree,
                                                                       const std::string& context)
    {
        EncoderLayerDescriptor layer;
        layer.attention = deserialize_attention_options(tree.get_child("attention"), context + " attention");
        layer.feed_forward =
            deserialize_feed_forward_options(tree.get_child("feed_forward"), context + " feed_forward");
        layer.window = deserialize_window_options(tree.get_child("window"), context + " window");
        return layer;
    }

    constexpr std::string_view descriptor_type_name(Tag<EncoderDescriptor>) { return "vision_encoder"; }

    inline PropertyTree serialize_descriptor(const EncoderDescriptor& descriptor)
    {
        const auto& options = descriptor.options;
        PropertyTree tree;
        tree.put("options.layers", static_cast<std::uint64_t>(options.layers));
        tree.put("options.embed_dim", options.embed_dim);
        tree.put("options.variant", variant_to_string(options.variant));
        tree.add_child("options.attention", serialize_attention_options(options.attention));
        tree.add_child("options.feed_forward", serialize_feed_forward_options(options.feed_forward));
        tree.add_child("options.layer_norm", serialize_layer_norm_options(options.layer_norm));
        tree.add_child("options.patch_embedding", serialize_patch_embedding_options(options.patch_embedding));
        tree.add_child("options.window", serialize_window_options(options.window));
        tree.add_child("options.positional_encoding",
                       ::Nott::Layer::Details::serialize_positional_encoding_options(options.positional_encoding));
        tree.put("options.residual_dropout", options.residual_dropout);
        tree.put("options.attention_dropout", options.attention_dropout);
        tree.put("options.feed_forward_dropout", options.feed_forward_dropout);
        tree.put("options.pre_norm", options.pre_norm);
        tree.put("options.final_layer_norm", options.final_layer_norm);
        PropertyTree layers;
        for (const auto& layer : descriptor.layers) {
            layers.push_back({"", serialize_encoder_layer_descriptor(layer)});
        }
        tree.add_child("layers", layers);
        return tree;
    }

    inline EncoderDescriptor deserialize_descriptor(Tag<EncoderDescriptor>, const PropertyTree& tree,
                                                    const std::string& context)
    {
        namespace S = ::Nott::Serialize;
        EncoderDescriptor descriptor;
        auto& options = descriptor.options;
        options.layers = static_cast<std::size_t>(S::get_numeric<std::uint64_t>(tree, "options.layers", context));
        options.embed_dim = S::get_numeric<std::int64_t>(tree, "options.embed_dim", context);
        options.variant = variant_from_string(S::get_string(tree, "options.variant", context), context);
        options.attention = deserialize_attention_options(tree.get_child("options.attention"), context);
        options.feed_forward = deserialize_feed_forward_options(tree.get_child("options.feed_forward"), context);
        options.layer_norm = deserialize_layer_norm_options(tree.get_child("options.layer_norm"), context);
        options.patch_embedding =
            deserialize_patch_embedding_options(tree.get_child("options.patch_embedding"), context);
        options.window = deserialize_window_options(tree.get_child("options.window"), context);
        options.positional_encoding = ::Nott::Layer::Details::deserialize_positional_encoding_options(
            tree.get_child("options.positional_encoding"), context);
        options.residual_dropout = S::get_numeric<double>(tree, "options.residual_dropout", context);
        options.attention_dropout = S::get_numeric<double>(tree, "options.attention_dropout", context);
        options.feed_forward_dropout = S::get_numeric<double>(tree, "options.feed_forward_dropout", context);
        options.pre_norm = S::get_boolean(tree, "options.pre_norm", context);
        options.final_layer_norm = S::get_boolean(tree, "options.final_layer_norm", context);
        for (const auto& node : tree.get_child("layers")) {
            descriptor.layers.push_back(
                deserialize_encoder_layer_descriptor(node.second, context + " vision encoder layer"));
        }
        return descriptor;
    }
}

#endif // Nott_BLOCK_TRANSFORMERS_SERIALIZE_VISION_HPP
