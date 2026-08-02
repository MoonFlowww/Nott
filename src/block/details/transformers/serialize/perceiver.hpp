#ifndef Nott_BLOCK_TRANSFORMERS_SERIALIZE_PERCEIVER_HPP
#define Nott_BLOCK_TRANSFORMERS_SERIALIZE_PERCEIVER_HPP
/// Serialization for the perceiver encoder.
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

#include "../../../../activation/serialize.hpp"
#include "../../../../common/serialize.hpp"
#include "../perceiver.hpp"

namespace Nott::Block::Details::Transformer::Perceiver {
    using ::Nott::Serialize::PropertyTree;
    using ::Nott::Serialize::Tag;

    inline PropertyTree serialize_attention_options(const AttentionOptions& options)
    {
        PropertyTree tree;
        tree.put("query_dim", options.query_dim);
        tree.put("key_dim", options.key_dim);
        tree.put("num_heads", options.num_heads);
        tree.put("dropout", options.dropout);
        tree.put("bias", options.bias);
        tree.put("batch_first", options.batch_first);
        return tree;
    }

    inline AttentionOptions deserialize_attention_options(const PropertyTree& tree, const std::string& context)
    {
        namespace S = ::Nott::Serialize;
        AttentionOptions options;
        options.query_dim = S::get_numeric<std::int64_t>(tree, "query_dim", context);
        options.key_dim = S::get_numeric<std::int64_t>(tree, "key_dim", context);
        options.num_heads = S::get_numeric<std::int64_t>(tree, "num_heads", context);
        options.dropout = S::get_numeric<double>(tree, "dropout", context);
        options.bias = S::get_boolean(tree, "bias", context);
        options.batch_first = S::get_boolean(tree, "batch_first", context);
        return options;
    }

    inline PropertyTree serialize_feed_forward_options(const FeedForwardOptions& options)
    {
        PropertyTree tree;
        tree.put("embed_dim", options.embed_dim);
        tree.put("mlp_ratio", options.mlp_ratio);
        tree.add_child("activation", ::Nott::Activation::serialize_activation_descriptor(options.activation));
        tree.put("bias", options.bias);
        tree.put("dropout", options.dropout);
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
        options.dropout = S::get_numeric<double>(tree, "dropout", context);
        return options;
    }

    inline PropertyTree serialize_encoder_layer_descriptor(const EncoderLayerDescriptor& layer)
    {
        PropertyTree tree;
        tree.add_child("feed_forward", serialize_feed_forward_options(layer.feed_forward));
        return tree;
    }

    inline EncoderLayerDescriptor deserialize_encoder_layer_descriptor(const PropertyTree& tree,
                                                                       const std::string& context)
    {
        EncoderLayerDescriptor layer;
        layer.feed_forward =
            deserialize_feed_forward_options(tree.get_child("feed_forward"), context + " feed_forward");
        return layer;
    }

    constexpr std::string_view descriptor_type_name(Tag<EncoderDescriptor>) { return "perceiver_encoder"; }

    inline PropertyTree serialize_descriptor(const EncoderDescriptor& descriptor)
    {
        const auto& options = descriptor.options;
        PropertyTree tree;
        tree.put("options.layers", static_cast<std::uint64_t>(options.layers));
        tree.put("options.self_layers", static_cast<std::uint64_t>(options.self_layers));
        tree.put("options.repeats", static_cast<std::uint64_t>(options.repeats));
        tree.put("options.latent_dim", options.latent_dim);
        tree.put("options.input_dim", options.input_dim);
        tree.put("options.latent_slots", static_cast<std::uint64_t>(options.latent_slots));
        tree.add_child("options.cross_attention", serialize_attention_options(options.cross_attention));
        tree.add_child("options.self_attention", serialize_attention_options(options.self_attention));
        tree.add_child("options.feed_forward", serialize_feed_forward_options(options.feed_forward));
        tree.put("options.residual_dropout", options.residual_dropout);
        tree.put("options.attention_dropout", options.attention_dropout);
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
        options.self_layers =
            static_cast<std::size_t>(S::get_numeric<std::uint64_t>(tree, "options.self_layers", context));
        options.repeats = static_cast<std::size_t>(S::get_numeric<std::uint64_t>(tree, "options.repeats", context));
        options.latent_dim = S::get_numeric<std::int64_t>(tree, "options.latent_dim", context);
        options.input_dim = S::get_numeric<std::int64_t>(tree, "options.input_dim", context);
        options.latent_slots =
            static_cast<std::size_t>(S::get_numeric<std::uint64_t>(tree, "options.latent_slots", context));
        options.cross_attention = deserialize_attention_options(tree.get_child("options.cross_attention"), context);
        options.self_attention = deserialize_attention_options(tree.get_child("options.self_attention"), context);
        options.feed_forward = deserialize_feed_forward_options(tree.get_child("options.feed_forward"), context);
        options.residual_dropout = S::get_numeric<double>(tree, "options.residual_dropout", context);
        options.attention_dropout = S::get_numeric<double>(tree, "options.attention_dropout", context);
        for (const auto& node : tree.get_child("layers")) {
            descriptor.layers.push_back(
                deserialize_encoder_layer_descriptor(node.second, context + " perceiver encoder layer"));
        }
        return descriptor;
    }
}

#endif // Nott_BLOCK_TRANSFORMERS_SERIALIZE_PERCEIVER_HPP
