#ifndef Nott_BLOCK_TRANSFORMERS_SERIALIZE_PLUSPLUS_HPP
#define Nott_BLOCK_TRANSFORMERS_SERIALIZE_PLUSPLUS_HPP
/// Serialization for the transformer++ encoder/decoder.
#include <cstdint>
#include <string>
#include <string_view>

#include "../../../../activation/serialize.hpp"
#include "../../../../attention/serialize.hpp"
#include "../../../../common/serialize.hpp"
#include "../../../../initialization/serialize.hpp"
#include "../../../../layer/serialize.hpp"
#include "../plusplus.hpp"
#include "classic.hpp"

namespace Nott::Block::Details::Transformer::PlusPlus {
    using ::Nott::Serialize::PropertyTree;
    using ::Nott::Serialize::Tag;

    inline PropertyTree serialize_auxiliary_head_options(const AuxiliaryHeadOptions& options)
    {
        PropertyTree tree;
        tree.put("enabled", options.enabled);
        tree.put("num_classes", options.num_classes);
        tree.put("dropout", options.dropout);
        return tree;
    }

    inline AuxiliaryHeadOptions deserialize_auxiliary_head_options(const PropertyTree& tree,
                                                                   const std::string& context)
    {
        namespace S = ::Nott::Serialize;
        AuxiliaryHeadOptions options;
        options.enabled = S::get_boolean(tree, "enabled", context);
        options.num_classes = S::get_numeric<std::int64_t>(tree, "num_classes", context);
        options.dropout = S::get_numeric<double>(tree, "dropout", context);
        return options;
    }

    inline PropertyTree serialize_hybrid_attention_options(const HybridAttentionOptions& options)
    {
        PropertyTree tree;
        tree.put("embed_dim", options.embed_dim);
        tree.put("num_heads", options.num_heads);
        tree.put("dropout", options.dropout);
        tree.put("bias", options.bias);
        tree.put("batch_first", options.batch_first);
        tree.put("variant", ::Nott::Attention::attention_variant_to_string(options.variant));
        tree.put("use_convolution", options.use_convolution);
        tree.put("convolution_kernel_size", options.convolution_kernel_size);
        tree.put("convolution_groups", options.convolution_groups);
        tree.put("convolution_dropout", options.convolution_dropout);
        return tree;
    }

    inline HybridAttentionOptions deserialize_hybrid_attention_options(const PropertyTree& tree,
                                                                       const std::string& context)
    {
        namespace S = ::Nott::Serialize;
        HybridAttentionOptions options;
        options.embed_dim = S::get_numeric<std::int64_t>(tree, "embed_dim", context);
        options.num_heads = S::get_numeric<std::int64_t>(tree, "num_heads", context);
        options.dropout = S::get_numeric<double>(tree, "dropout", context);
        options.bias = S::get_boolean(tree, "bias", context);
        options.batch_first = S::get_boolean(tree, "batch_first", context);
        options.variant = ::Nott::Attention::attention_variant_from_string(S::get_string(tree, "variant", context));
        options.use_convolution = S::get_boolean(tree, "use_convolution", context);
        options.convolution_kernel_size = S::get_numeric<std::int64_t>(tree, "convolution_kernel_size", context);
        options.convolution_groups = S::get_numeric<std::int64_t>(tree, "convolution_groups", context);
        options.convolution_dropout = S::get_numeric<double>(tree, "convolution_dropout", context);
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

    inline PropertyTree serialize_hybrid_attention_descriptor(const HybridAttentionDescriptor& descriptor)
    {
        PropertyTree tree;
        tree.add_child("attention", ::Nott::Attention::serialize_attention(descriptor.attention));
        tree.put("use_convolution", descriptor.use_convolution);
        tree.put("convolution_kernel_size", descriptor.convolution_kernel_size);
        tree.put("convolution_groups", descriptor.convolution_groups);
        tree.put("convolution_dropout", descriptor.convolution_dropout);
        return tree;
    }

    inline HybridAttentionDescriptor deserialize_hybrid_attention_descriptor(const PropertyTree& tree,
                                                                             const std::string& context)
    {
        namespace S = ::Nott::Serialize;
        HybridAttentionDescriptor descriptor;
        descriptor.attention = ::Nott::Attention::deserialize_attention(tree.get_child("attention"), context);
        descriptor.use_convolution = S::get_boolean(tree, "use_convolution", context);
        descriptor.convolution_kernel_size = S::get_numeric<std::int64_t>(tree, "convolution_kernel_size", context);
        descriptor.convolution_groups = S::get_numeric<std::int64_t>(tree, "convolution_groups", context);
        descriptor.convolution_dropout = S::get_numeric<double>(tree, "convolution_dropout", context);
        return descriptor;
    }

    inline PropertyTree serialize_encoder_layer_descriptor(const EncoderLayerDescriptor& layer)
    {
        PropertyTree tree;
        tree.add_child("hybrid_attention", serialize_hybrid_attention_descriptor(layer.hybrid_attention));
        tree.add_child("attention_dropout", ::Nott::Layer::serialize_layer_descriptor(layer.attention_dropout));
        PropertyTree feed_forward_layers;
        for (const auto& feed_forward : layer.feed_forward) {
            feed_forward_layers.push_back({"", ::Nott::Layer::serialize_layer_descriptor(feed_forward)});
        }
        tree.add_child("feed_forward", feed_forward_layers);
        tree.add_child("feed_forward_dropout",
                       ::Nott::Layer::serialize_layer_descriptor(layer.feed_forward_dropout));
        return tree;
    }

    inline EncoderLayerDescriptor deserialize_encoder_layer_descriptor(const PropertyTree& tree,
                                                                       const std::string& context)
    {
        EncoderLayerDescriptor layer;
        layer.hybrid_attention =
            deserialize_hybrid_attention_descriptor(tree.get_child("hybrid_attention"), context);
        layer.attention_dropout =
            ::Nott::Layer::deserialize_layer_descriptor(tree.get_child("attention_dropout"), context);
        for (const auto& node : tree.get_child("feed_forward")) {
            layer.feed_forward.push_back(::Nott::Layer::deserialize_layer_descriptor(node.second, context));
        }
        layer.feed_forward_dropout =
            ::Nott::Layer::deserialize_layer_descriptor(tree.get_child("feed_forward_dropout"), context);
        return layer;
    }

    inline PropertyTree serialize_decoder_layer_descriptor(const DecoderLayerDescriptor& layer)
    {
        PropertyTree tree;
        tree.add_child("self_attention", serialize_hybrid_attention_descriptor(layer.self_attention));
        tree.add_child("self_attention_dropout",
                       ::Nott::Layer::serialize_layer_descriptor(layer.self_attention_dropout));
        tree.add_child("cross_attention", ::Nott::Attention::serialize_attention(layer.cross_attention));
        tree.add_child("cross_attention_dropout",
                       ::Nott::Layer::serialize_layer_descriptor(layer.cross_attention_dropout));
        PropertyTree feed_forward_layers;
        for (const auto& feed_forward : layer.feed_forward) {
            feed_forward_layers.push_back({"", ::Nott::Layer::serialize_layer_descriptor(feed_forward)});
        }
        tree.add_child("feed_forward", feed_forward_layers);
        tree.add_child("feed_forward_dropout",
                       ::Nott::Layer::serialize_layer_descriptor(layer.feed_forward_dropout));
        return tree;
    }

    inline DecoderLayerDescriptor deserialize_decoder_layer_descriptor(const PropertyTree& tree,
                                                                       const std::string& context)
    {
        DecoderLayerDescriptor layer;
        layer.self_attention = deserialize_hybrid_attention_descriptor(tree.get_child("self_attention"), context);
        layer.self_attention_dropout =
            ::Nott::Layer::deserialize_layer_descriptor(tree.get_child("self_attention_dropout"), context);
        layer.cross_attention =
            ::Nott::Attention::deserialize_attention(tree.get_child("cross_attention"), context);
        layer.cross_attention_dropout =
            ::Nott::Layer::deserialize_layer_descriptor(tree.get_child("cross_attention_dropout"), context);
        for (const auto& node : tree.get_child("feed_forward")) {
            layer.feed_forward.push_back(::Nott::Layer::deserialize_layer_descriptor(node.second, context));
        }
        layer.feed_forward_dropout =
            ::Nott::Layer::deserialize_layer_descriptor(tree.get_child("feed_forward_dropout"), context);
        return layer;
    }

    constexpr std::string_view descriptor_type_name(Tag<EncoderDescriptor>) { return "transformer_pp_encoder"; }

    inline PropertyTree serialize_descriptor(const EncoderDescriptor& descriptor)
    {
        const auto& options = descriptor.options;
        PropertyTree tree;
        tree.put("options.layers", static_cast<std::uint64_t>(options.layers));
        tree.put("options.embed_dim", options.embed_dim);
        tree.add_child("options.hybrid_attention", serialize_hybrid_attention_options(options.hybrid_attention));
        tree.add_child("options.feed_forward", serialize_feed_forward_options(options.feed_forward));
        tree.add_child("options.layer_norm", serialize_layer_norm_options(options.layer_norm));
        tree.add_child("options.positional_encoding",
                       Classic::serialize_positional_encoding_options(options.positional_encoding));
        tree.put("options.dropout", options.dropout);
        tree.add_child("options.pos_head", serialize_auxiliary_head_options(options.pos_head));
        tree.add_child("options.ner_head", serialize_auxiliary_head_options(options.ner_head));
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
        options.hybrid_attention =
            deserialize_hybrid_attention_options(tree.get_child("options.hybrid_attention"), context);
        options.feed_forward = deserialize_feed_forward_options(tree.get_child("options.feed_forward"), context);
        options.layer_norm = deserialize_layer_norm_options(tree.get_child("options.layer_norm"), context);
        options.positional_encoding = Classic::deserialize_positional_encoding_options(
            tree.get_child("options.positional_encoding"), context);
        options.dropout = S::get_numeric<double>(tree, "options.dropout", context);
        options.pos_head = deserialize_auxiliary_head_options(tree.get_child("options.pos_head"), context);
        options.ner_head = deserialize_auxiliary_head_options(tree.get_child("options.ner_head"), context);
        for (const auto& node : tree.get_child("layers")) {
            descriptor.layers.push_back(
                deserialize_encoder_layer_descriptor(node.second, context + " transformer++ encoder layer"));
        }
        return descriptor;
    }

    constexpr std::string_view descriptor_type_name(Tag<DecoderDescriptor>) { return "transformer_pp_decoder"; }

    inline PropertyTree serialize_descriptor(const DecoderDescriptor& descriptor)
    {
        const auto& options = descriptor.options;
        PropertyTree tree;
        tree.put("options.layers", static_cast<std::uint64_t>(options.layers));
        tree.put("options.embed_dim", options.embed_dim);
        tree.add_child("options.self_attention", serialize_hybrid_attention_options(options.self_attention));
        tree.add_child("options.cross_attention", serialize_hybrid_attention_options(options.cross_attention));
        tree.add_child("options.feed_forward", serialize_feed_forward_options(options.feed_forward));
        tree.add_child("options.layer_norm", serialize_layer_norm_options(options.layer_norm));
        tree.add_child("options.positional_encoding",
                       Classic::serialize_positional_encoding_options(options.positional_encoding));
        tree.put("options.dropout", options.dropout);
        tree.add_child("options.pos_head", serialize_auxiliary_head_options(options.pos_head));
        tree.add_child("options.ner_head", serialize_auxiliary_head_options(options.ner_head));
        PropertyTree layers;
        for (const auto& layer : descriptor.layers) {
            layers.push_back({"", serialize_decoder_layer_descriptor(layer)});
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
        options.layers = static_cast<std::size_t>(S::get_numeric<std::uint64_t>(tree, "options.layers", context));
        options.embed_dim = S::get_numeric<std::int64_t>(tree, "options.embed_dim", context);
        options.self_attention =
            deserialize_hybrid_attention_options(tree.get_child("options.self_attention"), context);
        options.cross_attention =
            deserialize_hybrid_attention_options(tree.get_child("options.cross_attention"), context);
        options.feed_forward = deserialize_feed_forward_options(tree.get_child("options.feed_forward"), context);
        options.layer_norm = deserialize_layer_norm_options(tree.get_child("options.layer_norm"), context);
        options.positional_encoding = Classic::deserialize_positional_encoding_options(
            tree.get_child("options.positional_encoding"), context);
        options.dropout = S::get_numeric<double>(tree, "options.dropout", context);
        options.pos_head = deserialize_auxiliary_head_options(tree.get_child("options.pos_head"), context);
        options.ner_head = deserialize_auxiliary_head_options(tree.get_child("options.ner_head"), context);
        for (const auto& node : tree.get_child("layers")) {
            descriptor.layers.push_back(
                deserialize_decoder_layer_descriptor(node.second, context + " transformer++ decoder layer"));
        }
        return descriptor;
    }
}

#endif // Nott_BLOCK_TRANSFORMERS_SERIALIZE_PLUSPLUS_HPP
