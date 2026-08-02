#ifndef Nott_COMMON_SAVE_LOAD_HPP
#define Nott_COMMON_SAVE_LOAD_HPP
/// Model level save/load. The per descriptor read/write code lives next to each
/// descriptor (layer/serialize.hpp, loss/serialize.hpp, block/serialize.hpp and
/// so on); this file only walks the module list and touches the file.
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <boost/property_tree/json_parser.hpp>

#include "../block/serialize.hpp"
#include "../layer/serialize.hpp"
#include "local_serialize.hpp"
#include "serialize.hpp"

namespace Nott::Common::SaveLoad {
    using PropertyTree = ::Nott::Serialize::PropertyTree;
    using ModuleDescriptor = std::variant<Layer::Descriptor, Block::Descriptor>;

    /// Names kept so existing call sites and saved files are unaffected.
    using ::Nott::deserialize_local_config;
    using ::Nott::serialize_local_config;
    using ::Nott::Attention::deserialize_attention;
    using ::Nott::Attention::serialize_attention;
    using ::Nott::Block::deserialize_block_descriptor;
    using ::Nott::Block::serialize_block_descriptor;
    using ::Nott::Layer::deserialize_layer_descriptor;
    using ::Nott::Layer::serialize_layer_descriptor;
    using ::Nott::Loss::deserialize_loss;
    using ::Nott::Loss::serialize_loss;
    using ::Nott::Optimizer::deserialize_optimizer;
    using ::Nott::Optimizer::serialize_optimizer;
    using ::Nott::Regularization::deserialize_regularization;
    using ::Nott::Regularization::serialize_regularization;

    struct NamedModuleDescriptor {
        ModuleDescriptor descriptor{};
        std::string name{};

        NamedModuleDescriptor() = default;

        NamedModuleDescriptor(ModuleDescriptor descriptor, std::string name = {})
            : descriptor(std::move(descriptor)), name(std::move(name))
        {}
    };

    inline PropertyTree serialize_module_descriptor_payload(const ModuleDescriptor& descriptor)
    {
        PropertyTree tree;
        if (std::holds_alternative<Layer::Descriptor>(descriptor)) {
            tree.put("kind", "layer");
            tree.add_child("descriptor", serialize_layer_descriptor(std::get<Layer::Descriptor>(descriptor)));
        } else {
            tree.put("kind", "block");
            tree.add_child("descriptor", serialize_block_descriptor(std::get<Block::Descriptor>(descriptor)));
        }
        return tree;
    }

    inline PropertyTree serialize_module_descriptor(const NamedModuleDescriptor& descriptor)
    {
        auto tree = serialize_module_descriptor_payload(descriptor.descriptor);
        if (!descriptor.name.empty()) {
            tree.put("name", descriptor.name);
        }
        return tree;
    }

    inline ModuleDescriptor deserialize_module_descriptor(const PropertyTree& tree, const std::string& context)
    {
        const auto kind =
            ::Nott::Serialize::to_lower(::Nott::Serialize::get_string(tree, "kind", context));
        if (kind == "layer") {
            return ModuleDescriptor{deserialize_layer_descriptor(tree.get_child("descriptor"), context + " layer")};
        }
        if (kind == "block") {
            return ModuleDescriptor{deserialize_block_descriptor(tree.get_child("descriptor"), context + " block")};
        }
        std::ostringstream message;
        message << "Unknown module kind '" << kind << "' in " << context;
        throw std::runtime_error(message.str());
    }

    inline NamedModuleDescriptor deserialize_named_module_descriptor(const PropertyTree& tree,
                                                                     const std::string& context)
    {
        NamedModuleDescriptor descriptor{};
        descriptor.descriptor = deserialize_module_descriptor(tree, context);
        if (const auto name_value = tree.get_optional<std::string>("name")) {
            descriptor.name = *name_value;
        }
        return descriptor;
    }

    inline PropertyTree serialize_module_list(const std::vector<NamedModuleDescriptor>& descriptors)
    {
        PropertyTree tree;
        for (const auto& descriptor : descriptors) {
            tree.push_back({"", serialize_module_descriptor(descriptor)});
        }
        return tree;
    }

    inline std::vector<NamedModuleDescriptor> deserialize_module_list(const PropertyTree& tree,
                                                                       const std::string& context)
    {
        std::vector<NamedModuleDescriptor> descriptors;
        descriptors.reserve(tree.size());
        for (const auto& node : tree) {
            descriptors.push_back(deserialize_named_module_descriptor(node.second, context));
        }
        return descriptors;
    }

    inline void write_json_file(const std::filesystem::path& path, const PropertyTree& tree)
    {
        std::ofstream stream(path);
        if (!stream) {
            std::ostringstream message;
            message << "Failed to open '" << path.string() << "' for writing.";
            throw std::runtime_error(message.str());
        }
        boost::property_tree::write_json(stream, tree, true);
    }

    inline PropertyTree read_json_file(const std::filesystem::path& path)
    {
        PropertyTree tree;
        boost::property_tree::read_json(path.string(), tree);
        return tree;
    }
}
#endif // Nott_COMMON_SAVE_LOAD_HPP
