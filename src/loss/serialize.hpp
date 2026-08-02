#ifndef Nott_LOSS_SERIALIZE_HPP
#define Nott_LOSS_SERIALIZE_HPP
/// Serialization for Loss::Descriptor.
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "../common/serialize.hpp"
#include "loss.hpp"

/// Overloads live in the namespace the descriptors are declared in, which is
/// what ADL searches from the generic dispatch in common/serialize.hpp.
namespace Nott::Loss::Details {
    using ::Nott::Serialize::PropertyTree;
    using ::Nott::Serialize::Tag;

    inline std::string loss_reduction_to_string(Reduction reduction)
    {
        switch (reduction) {
            case Reduction::None: return "none";
            case Reduction::Sum: return "sum";
            case Reduction::Mean:
            default: return "mean";
        }
    }

    inline Reduction loss_reduction_from_string(const std::string& value, const std::string& context)
    {
        const auto lowered = ::Nott::Serialize::to_lower(value);
        if (lowered == "mean") return Reduction::Mean;
        if (lowered == "sum") return Reduction::Sum;
        if (lowered == "none") return Reduction::None;
        std::ostringstream message;
        message << "Unknown loss reduction '" << value << "' in " << context;
        throw std::runtime_error(message.str());
    }

    /// Every loss carries a reduction, most carry a weight vector. Factored so a
    /// new loss writes only the fields that are actually its own.
    inline void put_reduction(PropertyTree& tree, Reduction reduction)
    {
        tree.put("options.reduction", loss_reduction_to_string(reduction));
    }

    inline Reduction read_reduction(const PropertyTree& tree, const std::string& context)
    {
        return loss_reduction_from_string(::Nott::Serialize::get_string(tree, "options.reduction", context), context);
    }

    inline void put_weight(PropertyTree& tree, const std::string& key, const std::vector<double>& weight)
    {
        tree.put_child(key, ::Nott::Serialize::write_array(weight));
    }

    inline void read_weight(const PropertyTree& tree, const std::string& key,
                            std::vector<double>& weight, const std::string& context)
    {
        if (const auto node = tree.get_child_optional(key)) {
            weight = ::Nott::Serialize::read_array<double>(*node, context + "." + key);
        }
    }

    constexpr std::string_view descriptor_type_name(Tag<Details::MSEDescriptor>) { return "mse"; }

    inline PropertyTree serialize_descriptor(const Details::MSEDescriptor& descriptor)
    {
        PropertyTree tree;
        put_reduction(tree, descriptor.options.reduction);
        put_weight(tree, "options.weight", descriptor.options.weight);
        return tree;
    }

    inline Details::MSEDescriptor deserialize_descriptor(Tag<Details::MSEDescriptor>,
                                                         const PropertyTree& tree, const std::string& context)
    {
        Details::MSEOptions options;
        options.reduction = read_reduction(tree, context);
        read_weight(tree, "options.weight", options.weight, context);
        return Details::MSEDescriptor{options};
    }

    constexpr std::string_view descriptor_type_name(Tag<Details::CrossEntropyDescriptor>) { return "cross_entropy"; }

    inline PropertyTree serialize_descriptor(const Details::CrossEntropyDescriptor& descriptor)
    {
        PropertyTree tree;
        put_reduction(tree, descriptor.options.reduction);
        put_weight(tree, "options.weight", descriptor.options.weight);
        tree.put("options.label_smoothing", descriptor.options.label_smoothing);
        return tree;
    }

    inline Details::CrossEntropyDescriptor deserialize_descriptor(Tag<Details::CrossEntropyDescriptor>,
                                                                   const PropertyTree& tree, const std::string& context)
    {
        Details::CrossEntropyOptions options;
        options.reduction = read_reduction(tree, context);
        read_weight(tree, "options.weight", options.weight, context);
        options.label_smoothing = ::Nott::Serialize::get_numeric<double>(tree, "options.label_smoothing", context);
        return Details::CrossEntropyDescriptor{options};
    }

    constexpr std::string_view descriptor_type_name(Tag<Details::BCEWithLogitsDescriptor>) { return "bce_with_logits"; }

    inline PropertyTree serialize_descriptor(const Details::BCEWithLogitsDescriptor& descriptor)
    {
        PropertyTree tree;
        put_reduction(tree, descriptor.options.reduction);
        put_weight(tree, "options.weight", descriptor.options.weight);
        put_weight(tree, "options.pos_weight", descriptor.options.pos_weight);
        return tree;
    }

    inline Details::BCEWithLogitsDescriptor deserialize_descriptor(Tag<Details::BCEWithLogitsDescriptor>,
                                                                    const PropertyTree& tree, const std::string& context)
    {
        Details::BCEWithLogitsOptions options;
        options.reduction = read_reduction(tree, context);
        read_weight(tree, "options.weight", options.weight, context);
        read_weight(tree, "options.pos_weight", options.pos_weight, context);
        return Details::BCEWithLogitsDescriptor{options};
    }

    constexpr std::string_view descriptor_type_name(Tag<Details::MAEDescriptor>) { return "mae"; }

    inline PropertyTree serialize_descriptor(const Details::MAEDescriptor& descriptor)
    {
        PropertyTree tree;
        put_reduction(tree, descriptor.options.reduction);
        put_weight(tree, "options.weight", descriptor.options.weight);
        return tree;
    }

    inline Details::MAEDescriptor deserialize_descriptor(Tag<Details::MAEDescriptor>,
                                                         const PropertyTree& tree, const std::string& context)
    {
        Details::MAEOptions options;
        options.reduction = read_reduction(tree, context);
        read_weight(tree, "options.weight", options.weight, context);
        return Details::MAEDescriptor{options};
    }

    constexpr std::string_view descriptor_type_name(Tag<Details::NegativeLogLikelihoodDescriptor>) { return "nll"; }

    inline PropertyTree serialize_descriptor(const Details::NegativeLogLikelihoodDescriptor& descriptor)
    {
        PropertyTree tree;
        put_reduction(tree, descriptor.options.reduction);
        put_weight(tree, "options.weight", descriptor.options.weight);
        if (descriptor.options.ignore_index.has_value()) {
            tree.put("options.ignore_index", descriptor.options.ignore_index.value());
        }
        return tree;
    }

    inline Details::NegativeLogLikelihoodDescriptor deserialize_descriptor(
        Tag<Details::NegativeLogLikelihoodDescriptor>, const PropertyTree& tree, const std::string& context)
    {
        Details::NegativeLogLikelihoodOptions options;
        options.reduction = read_reduction(tree, context);
        read_weight(tree, "options.weight", options.weight, context);
        if (const auto ignore_index = tree.get_optional<std::int64_t>("options.ignore_index")) {
            options.ignore_index = *ignore_index;
        }
        return Details::NegativeLogLikelihoodDescriptor{options};
    }

    constexpr std::string_view descriptor_type_name(Tag<Details::SmoothL1Descriptor>) { return "smooth_l1"; }

    inline PropertyTree serialize_descriptor(const Details::SmoothL1Descriptor& descriptor)
    {
        PropertyTree tree;
        put_reduction(tree, descriptor.options.reduction);
        put_weight(tree, "options.weight", descriptor.options.weight);
        tree.put("options.beta", descriptor.options.beta);
        return tree;
    }

    inline Details::SmoothL1Descriptor deserialize_descriptor(Tag<Details::SmoothL1Descriptor>,
                                                              const PropertyTree& tree, const std::string& context)
    {
        Details::SmoothL1Options options;
        options.reduction = read_reduction(tree, context);
        read_weight(tree, "options.weight", options.weight, context);
        options.beta = ::Nott::Serialize::get_numeric<double>(tree, "options.beta", context);
        return Details::SmoothL1Descriptor{options};
    }

    constexpr std::string_view descriptor_type_name(Tag<Details::KLDivDescriptor>) { return "kl"; }

    inline PropertyTree serialize_descriptor(const Details::KLDivDescriptor& descriptor)
    {
        const auto& options = descriptor.options;
        PropertyTree tree;
        put_reduction(tree, options.reduction);
        tree.put("options.log_target", options.log_target);
        tree.put("options.use_batch_mean", options.use_batch_mean);
        tree.put("options.log_softmax_dim", options.log_softmax_dim);
        tree.put("options.prediction_is_log", options.prediction_is_log);
        return tree;
    }

    inline Details::KLDivDescriptor deserialize_descriptor(Tag<Details::KLDivDescriptor>,
                                                           const PropertyTree& tree, const std::string& context)
    {
        namespace S = ::Nott::Serialize;
        Details::KLDivOptions options;
        options.reduction = read_reduction(tree, context);
        options.log_target = S::get_boolean(tree, "options.log_target", context);
        options.use_batch_mean = S::get_boolean(tree, "options.use_batch_mean", context);
        options.log_softmax_dim = S::get_numeric<std::int64_t>(tree, "options.log_softmax_dim", context);
        options.prediction_is_log = S::get_boolean(tree, "options.prediction_is_log", context);
        return Details::KLDivDescriptor{options};
    }

    constexpr std::string_view descriptor_type_name(Tag<Details::MarginRankingDescriptor>) { return "margin_ranking"; }

    inline PropertyTree serialize_descriptor(const Details::MarginRankingDescriptor& descriptor)
    {
        PropertyTree tree;
        put_reduction(tree, descriptor.options.reduction);
        tree.put("options.margin", descriptor.options.margin);
        return tree;
    }

    inline Details::MarginRankingDescriptor deserialize_descriptor(Tag<Details::MarginRankingDescriptor>,
                                                                    const PropertyTree& tree, const std::string& context)
    {
        Details::MarginRankingOptions options;
        options.reduction = read_reduction(tree, context);
        options.margin = ::Nott::Serialize::get_numeric<double>(tree, "options.margin", context);
        return Details::MarginRankingDescriptor{options};
    }

    constexpr std::string_view descriptor_type_name(Tag<Details::CosineEmbeddingDescriptor>) { return "cosine_embedding"; }

    inline PropertyTree serialize_descriptor(const Details::CosineEmbeddingDescriptor& descriptor)
    {
        PropertyTree tree;
        put_reduction(tree, descriptor.options.reduction);
        tree.put("options.margin", descriptor.options.margin);
        return tree;
    }

    inline Details::CosineEmbeddingDescriptor deserialize_descriptor(Tag<Details::CosineEmbeddingDescriptor>,
                                                                      const PropertyTree& tree, const std::string& context)
    {
        Details::CosineEmbeddingOptions options;
        options.reduction = read_reduction(tree, context);
        options.margin = ::Nott::Serialize::get_numeric<double>(tree, "options.margin", context);
        return Details::CosineEmbeddingDescriptor{options};
    }

    constexpr std::string_view descriptor_type_name(Tag<Details::DiceDescriptor>) { return "dice"; }

    inline PropertyTree serialize_descriptor(const Details::DiceDescriptor& descriptor)
    {
        const auto& options = descriptor.options;
        PropertyTree tree;
        put_reduction(tree, options.reduction);
        tree.put("options.smooth", options.smooth);
        tree.put("options.exponent", options.exponent);
        tree.put("options.clamp_predictions", options.clamp_predictions);
        return tree;
    }

    inline Details::DiceDescriptor deserialize_descriptor(Tag<Details::DiceDescriptor>,
                                                          const PropertyTree& tree, const std::string& context)
    {
        namespace S = ::Nott::Serialize;
        Details::DiceOptions options;
        options.reduction = read_reduction(tree, context);
        options.smooth = S::get_numeric<double>(tree, "options.smooth", context);
        options.exponent = S::get_numeric<double>(tree, "options.exponent", context);
        options.clamp_predictions = S::get_boolean(tree, "options.clamp_predictions", context);
        return Details::DiceDescriptor{options};
    }

    constexpr std::string_view descriptor_type_name(Tag<Details::LovaszSoftmaxDescriptor>) { return "lovasz_softmax"; }

    inline PropertyTree serialize_descriptor(const Details::LovaszSoftmaxDescriptor& descriptor)
    {
        const auto& options = descriptor.options;
        PropertyTree tree;
        put_reduction(tree, options.reduction);
        tree.put("options.per_image", options.per_image);
        tree.put("options.ignore_index", options.ignore_index);
        tree.put("options.apply_softmax", options.apply_softmax);
        tree.put("options.include_background", options.include_background);
        tree.put("options.only_present_classes", options.only_present_classes);
        return tree;
    }

    inline Details::LovaszSoftmaxDescriptor deserialize_descriptor(Tag<Details::LovaszSoftmaxDescriptor>,
                                                                    const PropertyTree& tree, const std::string& context)
    {
        namespace S = ::Nott::Serialize;
        Details::LovaszSoftmaxOptions options;
        options.reduction = read_reduction(tree, context);
        options.per_image = S::get_boolean(tree, "options.per_image", context);
        options.ignore_index = S::get_numeric<std::int64_t>(tree, "options.ignore_index", context);
        options.apply_softmax = S::get_boolean(tree, "options.apply_softmax", context);
        options.include_background = S::get_boolean(tree, "options.include_background", context);
        options.only_present_classes = S::get_boolean(tree, "options.only_present_classes", context);
        return Details::LovaszSoftmaxDescriptor{options};
    }

    constexpr std::string_view descriptor_type_name(Tag<Details::TverskyDescriptor>) { return "tversky"; }

    inline PropertyTree serialize_descriptor(const Details::TverskyDescriptor& descriptor)
    {
        const auto& options = descriptor.options;
        PropertyTree tree;
        put_reduction(tree, options.reduction);
        tree.put("options.alpha", options.alpha);
        tree.put("options.beta", options.beta);
        tree.put("options.smooth", options.smooth);
        return tree;
    }

    inline Details::TverskyDescriptor deserialize_descriptor(Tag<Details::TverskyDescriptor>,
                                                             const PropertyTree& tree, const std::string& context)
    {
        namespace S = ::Nott::Serialize;
        Details::TverskyOptions options;
        options.reduction = read_reduction(tree, context);
        options.alpha = S::get_numeric<double>(tree, "options.alpha", context);
        options.beta = S::get_numeric<double>(tree, "options.beta", context);
        options.smooth = S::get_numeric<double>(tree, "options.smooth", context);
        return Details::TverskyDescriptor{options};
    }

}

namespace Nott::Loss {
    using Details::loss_reduction_from_string;
    using Details::loss_reduction_to_string;

    inline ::Nott::Serialize::PropertyTree serialize_loss(const Descriptor& descriptor)
    {
        return ::Nott::Serialize::serialize_descriptor(descriptor);
    }

    inline Descriptor deserialize_loss(const ::Nott::Serialize::PropertyTree& tree, const std::string& context)
    {
        return ::Nott::Serialize::deserialize_variant<Descriptor>(tree, context);
    }
}

#endif // Nott_LOSS_SERIALIZE_HPP
