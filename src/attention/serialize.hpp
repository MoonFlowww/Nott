#ifndef Nott_ATTENTION_SERIALIZE_HPP
#define Nott_ATTENTION_SERIALIZE_HPP
/// Serialization for Attention::Descriptor. One name/write/read triple per
/// alternative; the variant in attention.hpp is what enrolls them.
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>

#include "../common/serialize.hpp"
#include "attention.hpp"

namespace Nott::Attention {
    using ::Nott::Serialize::PropertyTree;
    using ::Nott::Serialize::Tag;

    inline std::string attention_variant_to_string(Variant variant)
    {
        switch (variant) {
            case Variant::Full: return "full";
            case Variant::Causal: return "causal";
        }
        throw std::runtime_error("Unsupported attention variant during serialisation.");
    }

    inline Variant attention_variant_from_string(const std::string& value)
    {
        const auto lowered = ::Nott::Serialize::to_lower(value);
        if (lowered == "full") return Variant::Full;
        if (lowered == "causal") return Variant::Causal;
        std::ostringstream message;
        message << "Unknown attention variant '" << value << "'.";
        throw std::runtime_error(message.str());
    }

    constexpr std::string_view descriptor_type_name(Tag<MultiHeadDescriptor>) { return "multi_head"; }

    inline PropertyTree serialize_descriptor(const MultiHeadDescriptor& descriptor)
    {
        const auto& options = descriptor.options;
        PropertyTree tree;
        tree.put("options.embed_dim", options.embed_dim);
        tree.put("options.num_heads", options.num_heads);
        tree.put("options.dropout", options.dropout);
        tree.put("options.bias", options.bias);
        tree.put("options.add_bias_kv", options.add_bias_kv);
        tree.put("options.add_zero_attn", options.add_zero_attn);
        tree.put("options.batch_first", options.batch_first);
        tree.put("options.variant", attention_variant_to_string(options.variant));
        return tree;
    }

    inline MultiHeadDescriptor deserialize_descriptor(Tag<MultiHeadDescriptor>,
                                                      const PropertyTree& tree,
                                                      const std::string& context)
    {
        namespace S = ::Nott::Serialize;
        MultiHeadOptions options;
        options.embed_dim = S::get_numeric<std::int64_t>(tree, "options.embed_dim", context);
        options.num_heads = S::get_numeric<std::int64_t>(tree, "options.num_heads", context);
        options.dropout = S::get_numeric<double>(tree, "options.dropout", context);
        options.bias = S::get_boolean(tree, "options.bias", context);
        options.add_bias_kv = S::get_boolean(tree, "options.add_bias_kv", context);
        options.add_zero_attn = S::get_boolean(tree, "options.add_zero_attn", context);
        options.batch_first = S::get_boolean(tree, "options.batch_first", context);
        options.variant = attention_variant_from_string(S::get_string(tree, "options.variant", context));
        return MultiHeadDescriptor{options};
    }

    constexpr std::string_view descriptor_type_name(Tag<MultiHeadLatentDescriptor>) { return "multi_head_latent"; }

    inline PropertyTree serialize_descriptor(const MultiHeadLatentDescriptor& descriptor)
    {
        const auto& options = descriptor.options;
        PropertyTree tree;
        tree.put("options.embed_dim", options.embed_dim);
        tree.put("options.num_heads", options.num_heads);
        tree.put("options.latent_dim", options.latent_dim);
        tree.put("options.dropout", options.dropout);
        tree.put("options.bias", options.bias);
        tree.put("options.batch_first", options.batch_first);
        tree.put("options.variant", attention_variant_to_string(options.variant));
        return tree;
    }

    inline MultiHeadLatentDescriptor deserialize_descriptor(Tag<MultiHeadLatentDescriptor>,
                                                            const PropertyTree& tree,
                                                            const std::string& context)
    {
        namespace S = ::Nott::Serialize;
        MultiHeadLatentOptions options;
        options.embed_dim = S::get_numeric<std::int64_t>(tree, "options.embed_dim", context);
        options.num_heads = S::get_numeric<std::int64_t>(tree, "options.num_heads", context);
        options.latent_dim = S::get_numeric<std::int64_t>(tree, "options.latent_dim", context);
        options.dropout = S::get_numeric<double>(tree, "options.dropout", context);
        options.bias = S::get_boolean(tree, "options.bias", context);
        options.batch_first = S::get_boolean(tree, "options.batch_first", context);
        options.variant = attention_variant_from_string(S::get_string(tree, "options.variant", context));
        return MultiHeadLatentDescriptor{options};
    }

    inline PropertyTree serialize_attention(const Descriptor& descriptor)
    {
        return ::Nott::Serialize::serialize_descriptor(descriptor);
    }

    inline Descriptor deserialize_attention(const PropertyTree& tree, const std::string& context)
    {
        return ::Nott::Serialize::deserialize_variant<Descriptor>(tree, context);
    }
}

#endif // Nott_ATTENTION_SERIALIZE_HPP
