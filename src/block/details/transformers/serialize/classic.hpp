#ifndef Nott_BLOCK_TRANSFORMERS_SERIALIZE_CLASSIC_HPP
#define Nott_BLOCK_TRANSFORMERS_SERIALIZE_CLASSIC_HPP
/// Serialization for the classic transformer encoder/decoder.
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include "../../../../activation/serialize.hpp"
#include "../../../../attention/serialize.hpp"
#include "../../../../common/serialize.hpp"
#include "../../../../initialization/serialize.hpp"
#include "../../../../layer/serialize.hpp"
#include "../classic.hpp"

namespace Nott::Block::Details::Transformer::Classic {
    using ::Nott::Serialize::PropertyTree;
    using ::Nott::Serialize::Tag;

    inline std::string positional_encoding_type_to_string(PositionalEncodingType type)
    {
        switch (type) {
            case PositionalEncodingType::None: return "none";
            case PositionalEncodingType::Sinusoidal: return "sinusoidal";
            case PositionalEncodingType::Learned: return "learned";
        }
        throw std::runtime_error("Unsupported positional encoding type during serialisation.");
    }

    inline PositionalEncodingType positional_encoding_type_from_string(const std::string& value)
    {
        const auto lowered = ::Nott::Serialize::to_lower(value);
        if (lowered == "none") return PositionalEncodingType::None;
        if (lowered == "sinusoidal") return PositionalEncodingType::Sinusoidal;
        if (lowered == "learned") return PositionalEncodingType::Learned;
        std::ostringstream message;
        message << "Unknown positional encoding type '" << value << "'.";
        throw std::runtime_error(message.str());
    }

    inline PropertyTree serialize_attention_options(const AttentionOptions& options)
    {
        PropertyTree tree;
        tree.put("embed_dim", options.embed_dim);
        tree.put("num_heads", options.num_heads);
        tree.put("harddropout", options.dropout);
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
        options.dropout = S::get_numeric<double>(tree, "harddropout", context);
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
        tree.put("bias", options.bias);
        tree.add_child("activation", ::Nott::Activation::serialize_activation_descriptor(options.activation));
        tree.add_child("initialization",
                       ::Nott::Initialization::serialize_initialization_descriptor(options.initialization));
        return tree;
    }

    inline FeedForwardOptions deserialize_feed_forward_options(const PropertyTree& tree, const std::string& context)
    {
        namespace S = ::Nott::Serialize;
        FeedForwardOptions options;
        options.embed_dim = S::get_numeric<std::int64_t>(tree, "embed_dim", context);
        options.mlp_ratio = S::get_numeric<double>(tree, "mlp_ratio", context);
        options.bias = S::get_boolean(tree, "bias", context);
        options.activation =
            ::Nott::Activation::deserialize_activation_descriptor(tree.get_child("activation"), context);
        options.initialization = ::Nott::Initialization::deserialize_initialization_descriptor(
            tree.get_child("initialization"), context);
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

    inline PropertyTree serialize_positional_encoding_options(const PositionalEncodingOptions& options)
    {
        PropertyTree tree;
        tree.put("type", positional_encoding_type_to_string(options.type));
        tree.put("harddropout", options.dropout);
        tree.put("max_length", static_cast<std::uint64_t>(options.max_length));
        tree.put("batch_first", options.batch_first);
        return tree;
    }

    inline PositionalEncodingOptions deserialize_positional_encoding_options(const PropertyTree& tree,
                                                                             const std::string& context)
    {
        namespace S = ::Nott::Serialize;
        PositionalEncodingOptions options;
        options.type = positional_encoding_type_from_string(S::get_string(tree, "type", context));
        options.dropout = S::get_numeric<double>(tree, "harddropout", context);
        options.max_length = static_cast<std::size_t>(S::get_numeric<std::uint64_t>(tree, "max_length", context));
        options.batch_first = S::get_boolean(tree, "batch_first", context);
        return options;
    }

    constexpr std::string_view descriptor_type_name(Tag<EncoderDescriptor>) { return "transformer_encoder"; }

    inline PropertyTree serialize_descriptor(const EncoderDescriptor& descriptor)
    {
        const auto& options = descriptor.options;
        PropertyTree tree;
        tree.add_child("options.attention", serialize_attention_options(options.attention));
        tree.add_child("options.feed_forward", serialize_feed_forward_options(options.feed_forward));
        tree.add_child("options.layer_norm", serialize_layer_norm_options(options.layer_norm));
        tree.add_child("options.positional_encoding",
                       serialize_positional_encoding_options(options.positional_encoding));
        tree.put("options.layers", static_cast<std::uint64_t>(options.layers));
        tree.put("options.embed_dim", options.embed_dim);
        tree.put("options.dropout", options.dropout);
        PropertyTree layers;
        for (const auto& layer : descriptor.layers) {
            PropertyTree entry;
            entry.add_child("attention", ::Nott::Attention::serialize_attention(layer.attention));
            entry.add_child("attention_dropout",
                            ::Nott::Layer::serialize_layer_descriptor(layer.attention_dropout));
            PropertyTree feed_forward_layers;
            for (const auto& feed_forward : layer.feed_forward) {
                feed_forward_layers.push_back({"", ::Nott::Layer::serialize_layer_descriptor(feed_forward)});
            }
            entry.add_child("feed_forward", feed_forward_layers);
            entry.add_child("feed_forward_dropout",
                            ::Nott::Layer::serialize_layer_descriptor(layer.feed_forward_dropout));
            layers.push_back({"", entry});
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
        options.attention = deserialize_attention_options(tree.get_child("options.attention"), context);
        options.feed_forward = deserialize_feed_forward_options(tree.get_child("options.feed_forward"), context);
        options.layer_norm = deserialize_layer_norm_options(tree.get_child("options.layer_norm"), context);
        options.positional_encoding =
            deserialize_positional_encoding_options(tree.get_child("options.positional_encoding"), context);
        options.layers = static_cast<std::size_t>(S::get_numeric<std::uint64_t>(tree, "options.layers", context));
        options.embed_dim = S::get_numeric<std::int64_t>(tree, "options.embed_dim", context);
        options.dropout = S::get_numeric<double>(tree, "options.dropout", context);
        for (const auto& node : tree.get_child("layers")) {
            EncoderLayerDescriptor layer;
            layer.attention = ::Nott::Attention::deserialize_attention(node.second.get_child("attention"),
                                                                       context + " encoder attention");
            layer.attention_dropout = ::Nott::Layer::deserialize_layer_descriptor(
                node.second.get_child("attention_dropout"), context + " encoder attention dropout");
            for (const auto& feed_forward : node.second.get_child("feed_forward")) {
                layer.feed_forward.push_back(::Nott::Layer::deserialize_layer_descriptor(
                    feed_forward.second, context + " encoder feed-forward"));
            }
            layer.feed_forward_dropout = ::Nott::Layer::deserialize_layer_descriptor(
                node.second.get_child("feed_forward_dropout"), context + " encoder feed-forward dropout");
            descriptor.layers.push_back(std::move(layer));
        }
        return descriptor;
    }

    constexpr std::string_view descriptor_type_name(Tag<DecoderDescriptor>) { return "transformer_decoder"; }

    inline PropertyTree serialize_descriptor(const DecoderDescriptor& descriptor)
    {
        const auto& options = descriptor.options;
        PropertyTree tree;
        tree.add_child("options.self_attention", serialize_attention_options(options.self_attention));
        tree.add_child("options.cross_attention", serialize_attention_options(options.cross_attention));
        tree.add_child("options.feed_forward", serialize_feed_forward_options(options.feed_forward));
        tree.add_child("options.layer_norm", serialize_layer_norm_options(options.layer_norm));
        tree.add_child("options.positional_encoding",
                       serialize_positional_encoding_options(options.positional_encoding));
        tree.put("options.layers", static_cast<std::uint64_t>(options.layers));
        tree.put("options.embed_dim", options.embed_dim);
        tree.put("options.dropout", options.dropout);
        PropertyTree layers;
        for (const auto& layer : descriptor.layers) {
            PropertyTree entry;
            entry.add_child("self_attention", ::Nott::Attention::serialize_attention(layer.self_attention));
            entry.add_child("self_attention_dropout",
                            ::Nott::Layer::serialize_layer_descriptor(layer.self_attention_dropout));
            entry.add_child("cross_attention", ::Nott::Attention::serialize_attention(layer.cross_attention));
            entry.add_child("cross_attention_dropout",
                            ::Nott::Layer::serialize_layer_descriptor(layer.cross_attention_dropout));
            PropertyTree feed_forward_layers;
            for (const auto& feed_forward : layer.feed_forward) {
                feed_forward_layers.push_back({"", ::Nott::Layer::serialize_layer_descriptor(feed_forward)});
            }
            entry.add_child("feed_forward", feed_forward_layers);
            entry.add_child("feed_forward_dropout",
                            ::Nott::Layer::serialize_layer_descriptor(layer.feed_forward_dropout));
            layers.push_back({"", entry});
        }
        tree.add_child("layers", layers);
        return tree;
    }

    inline DecoderDescriptor deserialize_descriptor(Tag<DecoderDescriptor>, const PropertyTree& tree,
                                                    const std::string& context)
    {
        namespace S = ::Nott::Serialize;
        DecoderDescriptor descriptor;
        auto& options = descriptor.options;
        options.self_attention = deserialize_attention_options(tree.get_child("options.self_attention"), context);
        options.cross_attention = deserialize_attention_options(tree.get_child("options.cross_attention"), context);
        options.feed_forward = deserialize_feed_forward_options(tree.get_child("options.feed_forward"), context);
        options.layer_norm = deserialize_layer_norm_options(tree.get_child("options.layer_norm"), context);
        options.positional_encoding =
            deserialize_positional_encoding_options(tree.get_child("options.positional_encoding"), context);
        options.layers = static_cast<std::size_t>(S::get_numeric<std::uint64_t>(tree, "options.layers", context));
        options.embed_dim = S::get_numeric<std::int64_t>(tree, "options.embed_dim", context);
        options.dropout = S::get_numeric<double>(tree, "options.dropout", context);
        for (const auto& node : tree.get_child("layers")) {
            DecoderLayerDescriptor layer;
            layer.self_attention = ::Nott::Attention::deserialize_attention(
                node.second.get_child("self_attention"), context + " decoder self-attention");
            layer.self_attention_dropout = ::Nott::Layer::deserialize_layer_descriptor(
                node.second.get_child("self_attention_dropout"), context + " decoder self-attention dropout");
            layer.cross_attention = ::Nott::Attention::deserialize_attention(
                node.second.get_child("cross_attention"), context + " decoder cross-attention");
            layer.cross_attention_dropout = ::Nott::Layer::deserialize_layer_descriptor(
                node.second.get_child("cross_attention_dropout"), context + " decoder cross-attention dropout");
            for (const auto& feed_forward : node.second.get_child("feed_forward")) {
                layer.feed_forward.push_back(::Nott::Layer::deserialize_layer_descriptor(
                    feed_forward.second, context + " decoder feed-forward"));
            }
            layer.feed_forward_dropout = ::Nott::Layer::deserialize_layer_descriptor(
                node.second.get_child("feed_forward_dropout"), context + " decoder feed-forward dropout");
            descriptor.layers.push_back(std::move(layer));
        }
        return descriptor;
    }
}

#endif // Nott_BLOCK_TRANSFORMERS_SERIALIZE_CLASSIC_HPP
