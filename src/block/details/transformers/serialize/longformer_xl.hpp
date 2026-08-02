#ifndef Nott_BLOCK_TRANSFORMERS_SERIALIZE_LONGFORMER_XL_HPP
#define Nott_BLOCK_TRANSFORMERS_SERIALIZE_LONGFORMER_XL_HPP
/// Serialization for the longformer XL encoder.
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

#include "../../../../activation/serialize.hpp"
#include "../../../../common/serialize.hpp"
#include "../longformer_xl.hpp"

namespace Nott::Block::Details::Transformer::LongformerXL {
    using ::Nott::Serialize::PropertyTree;
    using ::Nott::Serialize::Tag;

    inline PropertyTree serialize_attention_options(const AttentionOptions& options)
    {
        PropertyTree tree;
        tree.put("embed_dim", options.embed_dim);
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
        options.embed_dim = S::get_numeric<std::int64_t>(tree, "embed_dim", context);
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

    inline PropertyTree serialize_encoder_layer_descriptor(const EncoderLayerDescriptor& layer)
    {
        PropertyTree tree;
        tree.add_child("attention", serialize_attention_options(layer.attention));
        tree.add_child("feed_forward", serialize_feed_forward_options(layer.feed_forward));
        return tree;
    }

    inline EncoderLayerDescriptor deserialize_encoder_layer_descriptor(const PropertyTree& tree,
                                                                       const std::string& context)
    {
        EncoderLayerDescriptor layer;
        layer.attention = deserialize_attention_options(tree.get_child("attention"), context + " attention");
        layer.feed_forward =
            deserialize_feed_forward_options(tree.get_child("feed_forward"), context + " feed_forward");
        return layer;
    }

    constexpr std::string_view descriptor_type_name(Tag<EncoderDescriptor>) { return "longformer_xl_encoder"; }

    inline PropertyTree serialize_descriptor(const EncoderDescriptor& descriptor)
    {
        const auto& options = descriptor.options;
        PropertyTree tree;
        tree.put("options.layers", static_cast<std::uint64_t>(options.layers));
        tree.put("options.embed_dim", options.embed_dim);
        tree.add_child("options.attention", serialize_attention_options(options.attention));
        tree.add_child("options.feed_forward", serialize_feed_forward_options(options.feed_forward));
        tree.add_child("options.layer_norm", serialize_layer_norm_options(options.layer_norm));
        tree.put("options.window_size", options.window_size);
        tree.put("options.global_tokens", static_cast<std::uint64_t>(options.global_tokens));
        tree.put("options.causal", options.causal);
        tree.put("options.use_memory", options.use_memory);
        tree.put("options.memory_size", static_cast<std::uint64_t>(options.memory_size));
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
        options.attention = deserialize_attention_options(tree.get_child("options.attention"), context);
        options.feed_forward = deserialize_feed_forward_options(tree.get_child("options.feed_forward"), context);
        options.layer_norm = deserialize_layer_norm_options(tree.get_child("options.layer_norm"), context);
        options.window_size = S::get_numeric<std::int64_t>(tree, "options.window_size", context);
        options.global_tokens =
            static_cast<std::size_t>(S::get_numeric<std::uint64_t>(tree, "options.global_tokens", context));
        options.causal = S::get_boolean(tree, "options.causal", context);
        options.use_memory = S::get_boolean(tree, "options.use_memory", context);
        options.memory_size =
            static_cast<std::size_t>(S::get_numeric<std::uint64_t>(tree, "options.memory_size", context));
        options.residual_dropout = S::get_numeric<double>(tree, "options.residual_dropout", context);
        options.attention_dropout = S::get_numeric<double>(tree, "options.attention_dropout", context);
        options.feed_forward_dropout = S::get_numeric<double>(tree, "options.feed_forward_dropout", context);
        options.pre_norm = S::get_boolean(tree, "options.pre_norm", context);
        options.final_layer_norm = S::get_boolean(tree, "options.final_layer_norm", context);
        for (const auto& node : tree.get_child("layers")) {
            descriptor.layers.push_back(
                deserialize_encoder_layer_descriptor(node.second, context + " longformer encoder layer"));
        }
        return descriptor;
    }
}

#endif // Nott_BLOCK_TRANSFORMERS_SERIALIZE_LONGFORMER_XL_HPP
