#ifndef Nott_BLOCK_TRANSFORMERS_SERIALIZE_BERT_HPP
#define Nott_BLOCK_TRANSFORMERS_SERIALIZE_BERT_HPP
/// Serialization for the bert encoder.
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

#include "../../../../activation/serialize.hpp"
#include "../../../../attention/serialize.hpp"
#include "../../../../common/serialize.hpp"
#include "../bert.hpp"

namespace Nott::Block::Details::Transformer::Bert {
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
        /// Named form. Older files stored a bare enum index here, which the
        /// reader still accepts.
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

    inline PropertyTree serialize_embedding_options(const EmbeddingOptions& options)
    {
        PropertyTree tree;
        tree.put("vocab_size", options.vocab_size);
        tree.put("type_vocab_size", options.type_vocab_size);
        tree.put("max_position_embeddings", options.max_position_embeddings);
        tree.put("dropout", options.dropout);
        tree.put("use_token_type", options.use_token_type);
        tree.put("use_position_embeddings", options.use_position_embeddings);
        return tree;
    }

    inline EmbeddingOptions deserialize_embedding_options(const PropertyTree& tree, const std::string& context)
    {
        namespace S = ::Nott::Serialize;
        EmbeddingOptions options;
        options.vocab_size = S::get_numeric<std::int64_t>(tree, "vocab_size", context);
        options.type_vocab_size = S::get_numeric<std::int64_t>(tree, "type_vocab_size", context);
        options.max_position_embeddings = S::get_numeric<std::int64_t>(tree, "max_position_embeddings", context);
        options.dropout = S::get_numeric<double>(tree, "dropout", context);
        options.use_token_type = S::get_boolean(tree, "use_token_type", context);
        options.use_position_embeddings = S::get_boolean(tree, "use_position_embeddings", context);
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
        EncoderLayerDescriptor descriptor;
        descriptor.attention = deserialize_attention_options(tree.get_child("attention"), context + " attention");
        descriptor.feed_forward =
            deserialize_feed_forward_options(tree.get_child("feed_forward"), context + " feed_forward");
        return descriptor;
    }

    inline PropertyTree serialize_encoder_options(const EncoderOptions& options)
    {
        PropertyTree tree;
        tree.put("layers", static_cast<std::uint64_t>(options.layers));
        tree.put("embed_dim", options.embed_dim);
        tree.add_child("attention", serialize_attention_options(options.attention));
        tree.add_child("feed_forward", serialize_feed_forward_options(options.feed_forward));
        tree.add_child("layer_norm", serialize_layer_norm_options(options.layer_norm));
        tree.add_child("embedding", serialize_embedding_options(options.embedding));
        tree.put("residual_dropout", options.residual_dropout);
        tree.put("attention_dropout", options.attention_dropout);
        tree.put("feed_forward_dropout", options.feed_forward_dropout);
        tree.put("pre_norm", options.pre_norm);
        tree.put("final_layer_norm", options.final_layer_norm);
        return tree;
    }

    inline EncoderOptions deserialize_encoder_options(const PropertyTree& tree, const std::string& context)
    {
        namespace S = ::Nott::Serialize;
        EncoderOptions options;
        options.layers = static_cast<std::size_t>(S::get_numeric<std::uint64_t>(tree, "layers", context));
        options.embed_dim = S::get_numeric<std::int64_t>(tree, "embed_dim", context);
        options.attention = deserialize_attention_options(tree.get_child("attention"), context + " attention");
        options.feed_forward =
            deserialize_feed_forward_options(tree.get_child("feed_forward"), context + " feed_forward");
        options.layer_norm = deserialize_layer_norm_options(tree.get_child("layer_norm"), context + " layer_norm");
        options.embedding = deserialize_embedding_options(tree.get_child("embedding"), context + " embedding");
        options.residual_dropout = S::get_numeric<double>(tree, "residual_dropout", context);
        options.attention_dropout = S::get_numeric<double>(tree, "attention_dropout", context);
        options.feed_forward_dropout = S::get_numeric<double>(tree, "feed_forward_dropout", context);
        options.pre_norm = S::get_boolean(tree, "pre_norm", context);
        options.final_layer_norm = S::get_boolean(tree, "final_layer_norm", context);
        return options;
    }

    constexpr std::string_view descriptor_type_name(Tag<EncoderDescriptor>) { return "bert_encoder"; }

    inline PropertyTree serialize_descriptor(const EncoderDescriptor& descriptor)
    {
        PropertyTree tree;
        tree.add_child("options", serialize_encoder_options(descriptor.options));
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
        EncoderDescriptor descriptor;
        descriptor.options =
            deserialize_encoder_options(tree.get_child("options"), context + " bert encoder options");
        for (const auto& node : tree.get_child("layers")) {
            descriptor.layers.push_back(
                deserialize_encoder_layer_descriptor(node.second, context + " bert encoder layer"));
        }
        return descriptor;
    }
}

#endif // Nott_BLOCK_TRANSFORMERS_SERIALIZE_BERT_HPP
