#ifndef Nott_COMMON_SERIALIZE_HPP
#define Nott_COMMON_SERIALIZE_HPP
/// Generic serialization plumbing. Mirrors layer/registry.hpp: a descriptor is
/// serialized by an overload declared next to the type it describes, never by a
/// central switch. Adding a descriptor to a variant without writing that
/// overload is a compile error, not a silent gap at runtime.
#include <algorithm>
#include <cctype>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include <boost/property_tree/ptree.hpp>

namespace Nott::Serialize {
    using PropertyTree = boost::property_tree::ptree;

    /// Carries a type through overload resolution so deserialization, which has
    /// no argument to dispatch on, can use the same ADL mechanism as the rest.
    template <class T>
    struct Tag {
        using type = T;
    };

    inline std::string to_lower(std::string value)
    {
        std::transform(value.begin(), value.end(), value.begin(), [](unsigned char character) {
            return static_cast<char>(std::tolower(character));
        });
        return value;
    }

    template <class Numeric>
    Numeric get_numeric(const PropertyTree& tree, const std::string& key, const std::string& context)
    {
        static_assert(std::is_arithmetic_v<Numeric>, "Numeric type required for property tree extraction.");
        const auto value = tree.get_optional<Numeric>(key);
        if (!value) {
            std::ostringstream message;
            message << "Missing numeric field '" << key << "' in " << context;
            throw std::runtime_error(message.str());
        }
        return *value;
    }

    inline bool get_boolean(const PropertyTree& tree, const std::string& key, const std::string& context)
    {
        const auto value = tree.get_optional<bool>(key);
        if (!value) {
            std::ostringstream message;
            message << "Missing boolean field '" << key << "' in " << context;
            throw std::runtime_error(message.str());
        }
        return *value;
    }

    inline std::string get_string(const PropertyTree& tree, const std::string& key, const std::string& context)
    {
        const auto value = tree.get_optional<std::string>(key);
        if (!value) {
            std::ostringstream message;
            message << "Missing string field '" << key << "' in " << context;
            throw std::runtime_error(message.str());
        }
        return *value;
    }

    template <class T>
    std::vector<T> read_array(const PropertyTree& tree, const std::string& context)
    {
        std::vector<T> values;
        values.reserve(tree.size());
        for (const auto& child : tree) {
            try {
                if constexpr (std::is_same_v<T, bool>) {
                    values.push_back(static_cast<bool>(child.second.get_value<int>()));
                } else {
                    values.push_back(child.second.get_value<T>());
                }
            } catch (const boost::property_tree::ptree_bad_data&) {
                std::ostringstream message;
                message << "Invalid array element in " << context;
                throw std::runtime_error(message.str());
            }
        }
        return values;
    }

    template <class T>
    PropertyTree write_array(const std::vector<T>& values)
    {
        PropertyTree array;
        for (const auto& value : values) {
            PropertyTree element;
            element.put("", value);
            array.push_back({"", element});
        }
        return array;
    }

    /// Fallbacks. Instantiated only when a descriptor reaches the generic
    /// dispatch without its own overload, and then names what is missing.
    template <class Descriptor>
    constexpr std::string_view descriptor_type_name(Tag<Descriptor>)
    {
        static_assert(sizeof(Descriptor) == 0,
                      "Descriptor has no descriptor_type_name overload. Declare one next to the descriptor.");
        return {};
    }

    template <class Descriptor>
    PropertyTree serialize_descriptor(const Descriptor&)
    {
        static_assert(sizeof(Descriptor) == 0,
                      "Descriptor has no serialize_descriptor overload. Declare one next to the descriptor.");
        return {};
    }

    template <class Descriptor>
    Descriptor deserialize_descriptor(Tag<Descriptor>, const PropertyTree&, const std::string&)
    {
        static_assert(sizeof(Descriptor) == 0,
                      "Descriptor has no deserialize_descriptor overload. Declare one next to the descriptor.");
        return {};
    }

    /// The variant is the registry. Its alternative list is the single place a
    /// descriptor is enrolled, and it drives both directions.
    template <class... Descriptors>
    PropertyTree serialize_descriptor(const std::variant<Descriptors...>& descriptor)
    {
        return std::visit(
            [](const auto& concrete) {
                using Concrete = std::decay_t<decltype(concrete)>;
                auto tree = serialize_descriptor(concrete);
                tree.put("type", std::string{descriptor_type_name(Tag<Concrete>{})});
                return tree;
            },
            descriptor);
    }

    template <class Variant, std::size_t... Indices>
    std::optional<Variant> deserialize_alternative(const std::string& name,
                                                   const PropertyTree& tree,
                                                   const std::string& context,
                                                   std::index_sequence<Indices...>)
    {
        std::optional<Variant> result;
        const auto attempt = [&](auto tag) {
            if (result || descriptor_type_name(tag) != name) {
                return;
            }
            result = Variant{deserialize_descriptor(tag, tree, context)};
        };
        (attempt(Tag<std::variant_alternative_t<Indices, Variant>>{}), ...);
        return result;
    }

    template <class Variant>
    Variant deserialize_variant(const PropertyTree& tree, const std::string& context)
    {
        const auto name = to_lower(get_string(tree, "type", context));
        auto result = deserialize_alternative<Variant>(
            name, tree, context, std::make_index_sequence<std::variant_size_v<Variant>>{});
        if (!result) {
            std::ostringstream message;
            message << "Unknown type '" << name << "' in " << context;
            throw std::runtime_error(message.str());
        }
        return std::move(*result);
    }
}

#endif // Nott_COMMON_SERIALIZE_HPP
