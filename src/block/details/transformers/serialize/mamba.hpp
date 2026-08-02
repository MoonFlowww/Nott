#ifndef Nott_BLOCK_TRANSFORMERS_SERIALIZE_MAMBA_HPP
#define Nott_BLOCK_TRANSFORMERS_SERIALIZE_MAMBA_HPP
/// Serialization for the mamba encoder.
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include "../../../../common/serialize.hpp"
#include "../mamba.hpp"

namespace Nott::Block::Details::Transformer::Mamba {
    using ::Nott::Serialize::PropertyTree;
    using ::Nott::Serialize::Tag;

    inline PropertyTree serialize_rms_norm_options(const RMSNormOptions& options)
    {
        PropertyTree tree;
        tree.put("eps", options.eps);
        tree.put("learnable", options.learnable);
        return tree;
    }

    inline RMSNormOptions deserialize_rms_norm_options(const PropertyTree& tree, const std::string& context)
    {
        RMSNormOptions options;
        options.eps = ::Nott::Serialize::get_numeric<double>(tree, "eps", context + " rms_norm");
        options.learnable = ::Nott::Serialize::get_boolean(tree, "learnable", context + " rms_norm");
        return options;
    }

    inline PropertyTree serialize_selective_state_options(const SelectiveStateSpaceOptions& options)
    {
        PropertyTree tree;
        tree.put("embed_dim", options.embed_dim);
        tree.put("state_expansion", options.state_expansion);
        tree.put("ssm_layers", static_cast<std::uint64_t>(options.ssm_layers));
        tree.put("conv_kernel_size", static_cast<std::uint64_t>(options.conv_kernel_size));
        tree.put("dropout", options.dropout);
        tree.put("batch_first", options.batch_first);
        return tree;
    }

    inline SelectiveStateSpaceOptions deserialize_selective_state_options(const PropertyTree& tree,
                                                                          const std::string& context)
    {
        namespace S = ::Nott::Serialize;
        SelectiveStateSpaceOptions options;
        options.embed_dim = S::get_numeric<std::int64_t>(tree, "embed_dim", context + " selective_state");
        options.state_expansion = S::get_numeric<double>(tree, "state_expansion", context + " selective_state");
        options.ssm_layers = S::get_numeric<std::int64_t>(tree, "ssm_layers", context + " selective_state");
        options.conv_kernel_size = S::get_numeric<std::int64_t>(tree, "conv_kernel_size", context + " selective_state");
        options.dropout = S::get_numeric<double>(tree, "dropout", context + " selective_state");
        options.batch_first = S::get_boolean(tree, "batch_first", context + " selective_state");
        return options;
    }

    inline PropertyTree serialize_feed_forward_options(const FeedForwardOptions& options)
    {
        PropertyTree tree;
        tree.put("embed_dim", options.embed_dim);
        tree.put("expansion_ratio", options.expansion_ratio);
        tree.put("dropout", options.dropout);
        tree.put("gated", options.gated);
        return tree;
    }

    inline FeedForwardOptions deserialize_feed_forward_options(const PropertyTree& tree, const std::string& context)
    {
        namespace S = ::Nott::Serialize;
        FeedForwardOptions options;
        options.embed_dim = S::get_numeric<std::int64_t>(tree, "embed_dim", context + " feed_forward");
        options.expansion_ratio = S::get_numeric<double>(tree, "expansion_ratio", context + " feed_forward");
        options.dropout = S::get_numeric<double>(tree, "dropout", context + " feed_forward");
        options.gated = S::get_boolean(tree, "gated", context + " feed_forward");
        return options;
    }

    inline std::string serialize_normalization_order(NormalizationOrder order)
    {
        return order == NormalizationOrder::Pre ? "pre" : "post";
    }

    inline NormalizationOrder deserialize_normalization_order(const std::string& value, const std::string& context)
    {
        const auto lowered = ::Nott::Serialize::to_lower(value);
        if (lowered == "pre") return NormalizationOrder::Pre;
        if (lowered == "post") return NormalizationOrder::Post;
        std::ostringstream message;
        message << "Unknown normalization order '" << value << "' in " << context;
        throw std::runtime_error(message.str());
    }

    constexpr std::string_view descriptor_type_name(Tag<EncoderDescriptor>) { return "mamba_encoder"; }

    inline PropertyTree serialize_descriptor(const EncoderDescriptor& descriptor)
    {
        const auto& options = descriptor.options;
        PropertyTree tree;
        tree.put("options.layers", static_cast<std::uint64_t>(options.layers));
        tree.put("options.embed_dim", options.embed_dim);
        tree.add_child("options.rms_norm", serialize_rms_norm_options(options.rms_norm));
        tree.put("options.normalization", serialize_normalization_order(options.normalization));
        tree.put("options.residual_dropout", options.residual_dropout);
        tree.put("options.feed_forward_dropout", options.feed_forward_dropout);
        tree.put("options.residual_gating", options.residual_gating);
        tree.put("options.feed_forward_gating", options.feed_forward_gating);
        tree.put("options.batch_first", options.batch_first);
        tree.put("options.final_layer_norm", options.final_layer_norm);
        tree.add_child("options.selective_state", serialize_selective_state_options(options.selective_state));
        tree.add_child("options.feed_forward", serialize_feed_forward_options(options.feed_forward));
        PropertyTree layers;
        for (const auto& layer : descriptor.layers) {
            PropertyTree entry;
            entry.add_child("selective_state", serialize_selective_state_options(layer.selective_state));
            entry.add_child("feed_forward", serialize_feed_forward_options(layer.feed_forward));
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
        options.layers = static_cast<std::size_t>(S::get_numeric<std::uint64_t>(tree, "options.layers", context));
        options.embed_dim = S::get_numeric<std::int64_t>(tree, "options.embed_dim", context);
        options.rms_norm = deserialize_rms_norm_options(tree.get_child("options.rms_norm"), context);
        options.normalization =
            deserialize_normalization_order(S::get_string(tree, "options.normalization", context), context);
        options.residual_dropout = S::get_numeric<double>(tree, "options.residual_dropout", context);
        options.feed_forward_dropout = S::get_numeric<double>(tree, "options.feed_forward_dropout", context);
        options.residual_gating = S::get_boolean(tree, "options.residual_gating", context);
        options.feed_forward_gating = S::get_boolean(tree, "options.feed_forward_gating", context);
        options.batch_first = S::get_boolean(tree, "options.batch_first", context);
        options.final_layer_norm = S::get_boolean(tree, "options.final_layer_norm", context);
        options.selective_state = deserialize_selective_state_options(tree.get_child("options.selective_state"), context);
        options.feed_forward = deserialize_feed_forward_options(tree.get_child("options.feed_forward"), context);
        for (const auto& node : tree.get_child("layers")) {
            EncoderLayerDescriptor layer;
            layer.selective_state = deserialize_selective_state_options(node.second.get_child("selective_state"), context);
            layer.feed_forward = deserialize_feed_forward_options(node.second.get_child("feed_forward"), context);
            descriptor.layers.push_back(std::move(layer));
        }
        return descriptor;
    }
}

#endif // Nott_BLOCK_TRANSFORMERS_SERIALIZE_MAMBA_HPP
