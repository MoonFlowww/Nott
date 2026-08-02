#ifndef Nott_BLOCK_SERIALIZE_HPP
#define Nott_BLOCK_SERIALIZE_HPP
/// Serialization for Block::Descriptor. Each transformer family owns its own
/// file under details/transformers/serialize; this one holds the plain blocks
/// and pulls the families in.
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

#include "../activation/serialize.hpp"
#include "../common/local_serialize.hpp"
#include "../common/serialize.hpp"
#include "../layer/serialize.hpp"
#include "block.hpp"
#include "details/transformers/serialize/bert.hpp"
#include "details/transformers/serialize/classic.hpp"
#include "details/transformers/serialize/ebt.hpp"
#include "details/transformers/serialize/longformer_xl.hpp"
#include "details/transformers/serialize/mamba.hpp"
#include "details/transformers/serialize/perceiver.hpp"
#include "details/transformers/serialize/plusplus.hpp"
#include "details/transformers/serialize/vision.hpp"

namespace Nott::Block::Details {
    using ::Nott::Serialize::PropertyTree;
    using ::Nott::Serialize::Tag;

    constexpr std::string_view descriptor_type_name(Tag<SequentialDescriptor>) { return "sequential"; }

    inline PropertyTree serialize_descriptor(const SequentialDescriptor& descriptor)
    {
        PropertyTree tree;
        PropertyTree layers;
        for (const auto& layer : descriptor.layers) {
            layers.push_back({"", ::Nott::Layer::serialize_layer_descriptor(layer)});
        }
        tree.add_child("layers", layers);
        tree.add_child("local", ::Nott::serialize_local_config(descriptor.local));
        return tree;
    }

    inline SequentialDescriptor deserialize_descriptor(Tag<SequentialDescriptor>, const PropertyTree& tree,
                                                        const std::string& context)
    {
        SequentialDescriptor descriptor;
        for (const auto& node : tree.get_child("layers")) {
            descriptor.layers.push_back(
                ::Nott::Layer::deserialize_layer_descriptor(node.second, context + " sequential layer"));
        }
        descriptor.local = ::Nott::deserialize_local_config(tree.get_child("local"), context + " sequential local");
        return descriptor;
    }

    constexpr std::string_view descriptor_type_name(Tag<ResidualDescriptor>) { return "residual"; }

    inline PropertyTree serialize_descriptor(const ResidualDescriptor& descriptor)
    {
        PropertyTree tree;
        PropertyTree layers;
        for (const auto& layer : descriptor.layers) {
            layers.push_back({"", ::Nott::Layer::serialize_layer_descriptor(layer)});
        }
        tree.add_child("layers", layers);
        tree.put("repeats", static_cast<std::uint64_t>(descriptor.repeats));
        if (descriptor.skip.projection) {
            tree.add_child("skip.projection",
                           ::Nott::Layer::serialize_layer_descriptor(*descriptor.skip.projection));
        }
        tree.add_child("output.final_activation",
                       ::Nott::Activation::serialize_activation_descriptor(descriptor.output.final_activation));
        tree.put("output.dropout", descriptor.output.dropout);
        tree.add_child("local", ::Nott::serialize_local_config(descriptor.local));
        return tree;
    }

    inline ResidualDescriptor deserialize_descriptor(Tag<ResidualDescriptor>, const PropertyTree& tree,
                                                      const std::string& context)
    {
        namespace S = ::Nott::Serialize;
        ResidualDescriptor descriptor;
        for (const auto& node : tree.get_child("layers")) {
            descriptor.layers.push_back(
                ::Nott::Layer::deserialize_layer_descriptor(node.second, context + " residual layer"));
        }
        descriptor.repeats = static_cast<std::size_t>(S::get_numeric<std::uint64_t>(tree, "repeats", context));
        if (const auto projection = tree.get_child_optional("skip.projection")) {
            descriptor.skip.projection =
                ::Nott::Layer::deserialize_layer_descriptor(*projection, context + " residual projection");
        }
        descriptor.output.final_activation = ::Nott::Activation::deserialize_activation_descriptor(
            tree.get_child("output.final_activation"), context);
        descriptor.output.dropout = S::get_numeric<double>(tree, "output.dropout", context);
        descriptor.local = ::Nott::deserialize_local_config(tree.get_child("local"), context + " residual local");
        return descriptor;
    }
}

namespace Nott::Block {
    inline ::Nott::Serialize::PropertyTree serialize_block_descriptor(const Descriptor& descriptor)
    {
        return ::Nott::Serialize::serialize_descriptor(descriptor);
    }

    inline Descriptor deserialize_block_descriptor(const ::Nott::Serialize::PropertyTree& tree,
                                                   const std::string& context)
    {
        return ::Nott::Serialize::deserialize_variant<Descriptor>(tree, context);
    }
}

#endif // Nott_BLOCK_SERIALIZE_HPP
