#ifndef Nott_INITIALIZATION_SERIALIZE_HPP
#define Nott_INITIALIZATION_SERIALIZE_HPP
/// Serialization for Initialization::Descriptor.
#include <sstream>
#include <stdexcept>
#include <string>

#include "../common/serialize.hpp"
#include "initialization.hpp"

namespace Nott::Initialization {
    using ::Nott::Serialize::PropertyTree;

    inline std::string initialization_type_to_string(Type type)
    {
        switch (type) {
            case Type::Default: return "default";
            case Type::XavierNormal: return "xavier_normal";
            case Type::XavierUniform: return "xavier_uniform";
            case Type::HeNormal: return "he_normal";
            case Type::HeUniform: return "he_uniform";
            case Type::ZeroBias: return "zero_bias";
            case Type::Dirac: return "dirac";
            case Type::Lyapunov: return "lyapunov";
        }
        throw std::runtime_error("Unsupported initialisation type during serialisation.");
    }

    inline Type initialization_type_from_string(const std::string& value)
    {
        const auto lowered = ::Nott::Serialize::to_lower(value);
        if (lowered == "default") return Type::Default;
        if (lowered == "xavier_normal") return Type::XavierNormal;
        if (lowered == "xavier_uniform") return Type::XavierUniform;
        if (lowered == "he_normal") return Type::HeNormal;
        if (lowered == "he_uniform") return Type::HeUniform;
        if (lowered == "zero_bias") return Type::ZeroBias;
        if (lowered == "dirac") return Type::Dirac;
        if (lowered == "lyapunov") return Type::Lyapunov;
        std::ostringstream message;
        message << "Unknown initialisation type '" << value << "'.";
        throw std::runtime_error(message.str());
    }

    inline PropertyTree serialize_initialization_descriptor(const Descriptor& descriptor)
    {
        PropertyTree tree;
        tree.put("type", initialization_type_to_string(descriptor.type));
        return tree;
    }

    inline Descriptor deserialize_initialization_descriptor(const PropertyTree& tree, const std::string& context)
    {
        Descriptor descriptor;
        descriptor.type =
            initialization_type_from_string(::Nott::Serialize::get_string(tree, "type", context));
        return descriptor;
    }
}

#endif // Nott_INITIALIZATION_SERIALIZE_HPP
