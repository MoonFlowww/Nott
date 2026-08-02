#ifndef Nott_BLOCK_TRANSFORMERS_SERIALIZE_EBT_HPP
#define Nott_BLOCK_TRANSFORMERS_SERIALIZE_EBT_HPP
/// Serialization for the EBT encoder/decoder.
#include <cstddef>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>

#include "../../../../common/serialize.hpp"
#include "../ebt.hpp"

namespace Nott::Block::Details::Transformer::EBT {
    using ::Nott::Serialize::PropertyTree;
    using ::Nott::Serialize::Tag;

    inline std::string modality_type_to_string(ModalityType type)
    {
        switch (type) {
            case ModalityType::Discrete: return "discrete";
            case ModalityType::Continuous: return "continuous";
        }
        throw std::runtime_error("Unsupported EBT modality type during serialisation.");
    }

    inline ModalityType modality_type_from_string(const std::string& value)
    {
        const auto lowered = ::Nott::Serialize::to_lower(value);
        if (lowered == "discrete") return ModalityType::Discrete;
        if (lowered == "continuous") return ModalityType::Continuous;
        std::ostringstream message;
        message << "Unknown EBT modality type '" << value << "'.";
        throw std::runtime_error(message.str());
    }

    inline PropertyTree serialize_modality_options(const ModalityOptions& options)
    {
        PropertyTree tree;
        tree.put("type", modality_type_to_string(options.type));
        tree.put("vocab_size", static_cast<std::int64_t>(options.vocab_size));
        tree.put("input_dim", static_cast<std::int64_t>(options.input_dim));
        tree.put("embed_dim", static_cast<std::int64_t>(options.embed_dim));
        return tree;
    }

    inline ModalityOptions deserialize_modality_options(const PropertyTree& tree, const std::string& context)
    {
        namespace S = ::Nott::Serialize;
        ModalityOptions options;
        options.type = modality_type_from_string(S::get_string(tree, "type", context));
        options.vocab_size = S::get_numeric<std::int64_t>(tree, "vocab_size", context);
        options.input_dim = S::get_numeric<std::int64_t>(tree, "input_dim", context);
        options.embed_dim = S::get_numeric<std::int64_t>(tree, "embed_dim", context);
        return options;
    }

    inline PropertyTree serialize_energy_options(const EnergyScorerOptions& options)
    {
        PropertyTree tree;
        tree.put("depth", static_cast<std::uint64_t>(options.depth));
        tree.put("hidden_size", static_cast<std::int64_t>(options.hidden_size));
        tree.put("modality_heads", static_cast<std::int64_t>(options.modality_heads));
        return tree;
    }

    inline EnergyScorerOptions deserialize_energy_options(const PropertyTree& tree, const std::string& context)
    {
        namespace S = ::Nott::Serialize;
        EnergyScorerOptions options;
        options.depth = static_cast<std::size_t>(S::get_numeric<std::uint64_t>(tree, "depth", context));
        options.hidden_size = S::get_numeric<std::int64_t>(tree, "hidden_size", context);
        options.modality_heads = S::get_numeric<std::int64_t>(tree, "modality_heads", context);
        return options;
    }

    inline PropertyTree serialize_optimizer_options(const OptimizerOptions& options)
    {
        PropertyTree tree;
        tree.put("learning_rate", options.learning_rate);
        tree.put("momentum", options.momentum);
        tree.put("gradient_clip_norm", options.gradient_clip_norm);
        return tree;
    }

    inline OptimizerOptions deserialize_optimizer_options(const PropertyTree& tree, const std::string& context)
    {
        namespace S = ::Nott::Serialize;
        OptimizerOptions options;
        options.learning_rate = S::get_numeric<double>(tree, "learning_rate", context);
        options.momentum = S::get_numeric<double>(tree, "momentum", context);
        options.gradient_clip_norm = S::get_numeric<double>(tree, "gradient_clip_norm", context);
        return options;
    }

    inline PropertyTree serialize_refinement_options(const RefinementOptions& options)
    {
        PropertyTree tree;
        tree.put("max_steps", static_cast<std::uint64_t>(options.max_steps));
        tree.put("tolerance", options.tolerance);
        tree.put("stop_on_plateau", options.stop_on_plateau);
        return tree;
    }

    inline RefinementOptions deserialize_refinement_options(const PropertyTree& tree, const std::string& context)
    {
        namespace S = ::Nott::Serialize;
        RefinementOptions options;
        options.max_steps = static_cast<std::size_t>(S::get_numeric<std::uint64_t>(tree, "max_steps", context));
        options.tolerance = S::get_numeric<double>(tree, "tolerance", context);
        options.stop_on_plateau = S::get_boolean(tree, "stop_on_plateau", context);
        return options;
    }

    constexpr std::string_view descriptor_type_name(Tag<EncoderDescriptor>) { return "ebt_encoder"; }

    inline PropertyTree serialize_descriptor(const EncoderDescriptor& descriptor)
    {
        const auto& options = descriptor.options;
        PropertyTree tree;
        tree.add_child("options.modality", serialize_modality_options(options.modality));
        tree.add_child("options.energy", serialize_energy_options(options.energy));
        tree.add_child("options.optimizer", serialize_optimizer_options(options.optimizer));
        tree.add_child("options.refinement", serialize_refinement_options(options.refinement));
        return tree;
    }

    inline EncoderDescriptor deserialize_descriptor(Tag<EncoderDescriptor>, const PropertyTree& tree,
                                                    const std::string& context)
    {
        EncoderDescriptor descriptor;
        auto& options = descriptor.options;
        options.modality = deserialize_modality_options(tree.get_child("options.modality"), context);
        options.energy = deserialize_energy_options(tree.get_child("options.energy"), context);
        options.optimizer = deserialize_optimizer_options(tree.get_child("options.optimizer"), context);
        options.refinement = deserialize_refinement_options(tree.get_child("options.refinement"), context);
        return descriptor;
    }

    constexpr std::string_view descriptor_type_name(Tag<DecoderDescriptor>) { return "ebt_decoder"; }

    inline PropertyTree serialize_descriptor(const DecoderDescriptor& descriptor)
    {
        const auto& options = descriptor.options;
        PropertyTree tree;
        tree.add_child("options.target", serialize_modality_options(options.target));
        if (options.context.has_value()) {
            tree.add_child("options.context", serialize_modality_options(*options.context));
        }
        tree.add_child("options.energy", serialize_energy_options(options.energy));
        tree.add_child("options.optimizer", serialize_optimizer_options(options.optimizer));
        tree.add_child("options.refinement", serialize_refinement_options(options.refinement));
        return tree;
    }

    inline DecoderDescriptor deserialize_descriptor(Tag<DecoderDescriptor>, const PropertyTree& tree,
                                                    const std::string& context)
    {
        DecoderDescriptor descriptor;
        auto& options = descriptor.options;
        options.target = deserialize_modality_options(tree.get_child("options.target"), context);
        if (const auto context_node = tree.get_child_optional("options.context")) {
            options.context = deserialize_modality_options(*context_node, context);
        }
        options.energy = deserialize_energy_options(tree.get_child("options.energy"), context);
        options.optimizer = deserialize_optimizer_options(tree.get_child("options.optimizer"), context);
        options.refinement = deserialize_refinement_options(tree.get_child("options.refinement"), context);
        return descriptor;
    }
}

#endif // Nott_BLOCK_TRANSFORMERS_SERIALIZE_EBT_HPP
