#ifndef Nott_ACTIVATION_SERIALIZE_HPP
#define Nott_ACTIVATION_SERIALIZE_HPP
/// Serialization for Activation::Descriptor. Lives next to the type so adding an
/// activation touches this folder only.
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>

#include "../common/serialize.hpp"
#include "activation.hpp"

namespace Nott::Activation {
    using ::Nott::Serialize::PropertyTree;

    inline std::string activation_type_to_string(Type type)
    {
        switch (type) {
            case Type::Identity: return "identity";
            case Type::ReLU: return "relu";
            case Type::Sigmoid: return "sigmoid";
            case Type::Tanh: return "tanh";
            case Type::LeakyReLU: return "leaky_relu";
            case Type::Softmax: return "softmax";
            case Type::SiLU: return "silu";
            case Type::GeLU: return "gelu";
            case Type::GLU: return "glu";
            case Type::SwiGLU: return "swiglu";
            case Type::dSiLU: return "dsilu";
            case Type::PSiLU: return "psilu";
            case Type::Mish: return "mish";
            case Type::Swish: return "swish";
        }
        throw std::runtime_error("Unsupported activation type during serialisation.");
    }

    inline Type activation_type_from_string(const std::string& value)
    {
        const auto lowered = ::Nott::Serialize::to_lower(value);
        if (lowered == "identity") return Type::Identity;
        if (lowered == "relu") return Type::ReLU;
        if (lowered == "sigmoid") return Type::Sigmoid;
        if (lowered == "tanh") return Type::Tanh;
        if (lowered == "leaky_relu") return Type::LeakyReLU;
        if (lowered == "softmax") return Type::Softmax;
        if (lowered == "silu") return Type::SiLU;
        if (lowered == "gelu") return Type::GeLU;
        if (lowered == "glu") return Type::GLU;
        if (lowered == "swiglu") return Type::SwiGLU;
        if (lowered == "dsilu") return Type::dSiLU;
        if (lowered == "psilu") return Type::PSiLU;
        if (lowered == "mish") return Type::Mish;
        if (lowered == "swish") return Type::Swish;
        std::ostringstream message;
        message << "Unknown activation type '" << value << "'.";
        throw std::runtime_error(message.str());
    }

    /// Reads the legacy encoding that stored the enum by index. Kept for files
    /// written before activations were named, and only reachable from that path.
    inline Type activation_type_from_index(std::uint64_t value, const std::string& context)
    {
        switch (value) {
            case static_cast<std::uint64_t>(Type::Identity): return Type::Identity;
            case static_cast<std::uint64_t>(Type::ReLU): return Type::ReLU;
            case static_cast<std::uint64_t>(Type::Sigmoid): return Type::Sigmoid;
            case static_cast<std::uint64_t>(Type::Tanh): return Type::Tanh;
            case static_cast<std::uint64_t>(Type::LeakyReLU): return Type::LeakyReLU;
            case static_cast<std::uint64_t>(Type::Softmax): return Type::Softmax;
            case static_cast<std::uint64_t>(Type::SiLU): return Type::SiLU;
            case static_cast<std::uint64_t>(Type::GeLU): return Type::GeLU;
            case static_cast<std::uint64_t>(Type::GLU): return Type::GLU;
            case static_cast<std::uint64_t>(Type::SwiGLU): return Type::SwiGLU;
            case static_cast<std::uint64_t>(Type::dSiLU): return Type::dSiLU;
            case static_cast<std::uint64_t>(Type::PSiLU): return Type::PSiLU;
            case static_cast<std::uint64_t>(Type::Mish): return Type::Mish;
            case static_cast<std::uint64_t>(Type::Swish): return Type::Swish;
            default: break;
        }
        std::ostringstream message;
        message << "Unknown activation type index '" << value << "' in " << context;
        throw std::runtime_error(message.str());
    }

    inline PropertyTree serialize_activation_descriptor(const Descriptor& descriptor)
    {
        PropertyTree tree;
        tree.put("type", activation_type_to_string(descriptor.type));
        return tree;
    }

    inline Descriptor deserialize_activation_descriptor(const PropertyTree& tree, const std::string& context)
    {
        Descriptor descriptor;
        descriptor.type =
            activation_type_from_string(::Nott::Serialize::get_string(tree, "type", context));
        return descriptor;
    }

    /// Reads an activation stored under key, accepting both the named form and
    /// the older bare enum index some transformer families used to write. New
    /// files always get the name, which survives reordering the enum.
    inline Descriptor deserialize_activation_field(const PropertyTree& tree, const std::string& key,
                                                   const std::string& context)
    {
        Descriptor descriptor;
        const auto node = tree.get_child_optional(key);
        if (!node) {
            return descriptor;
        }
        const auto value = ::Nott::Serialize::get_string(*node, "type", context + " " + key);
        const bool is_index = !value.empty() &&
                              value.find_first_not_of("0123456789") == std::string::npos;
        descriptor.type = is_index ? activation_type_from_index(std::stoull(value), context)
                                   : activation_type_from_string(value);
        return descriptor;
    }
}

#endif // Nott_ACTIVATION_SERIALIZE_HPP
