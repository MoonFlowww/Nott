#ifndef Nott_REGULARIZATION_SERIALIZE_HPP
#define Nott_REGULARIZATION_SERIALIZE_HPP
/// Serialization for Regularization::Descriptor.
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

#include "../common/serialize.hpp"
#include "regularization.hpp"

namespace Nott::Regularization::Details {
    using ::Nott::Serialize::PropertyTree;
    using ::Nott::Serialize::Tag;

    /// Most regularizers are a single coefficient. The shared pair keeps a new
    /// one down to a name plus whatever extra fields it actually has.
    inline PropertyTree serialize_coefficient(double coefficient)
    {
        PropertyTree tree;
        tree.put("options.coefficient", coefficient);
        return tree;
    }

    inline double read_coefficient(const PropertyTree& tree, const std::string& context)
    {
        return ::Nott::Serialize::get_numeric<double>(tree, "options.coefficient", context);
    }

    inline PropertyTree serialize_strength(double strength)
    {
        PropertyTree tree;
        tree.put("options.strength", strength);
        return tree;
    }

    inline double read_strength(const PropertyTree& tree, const std::string& context)
    {
        return ::Nott::Serialize::get_numeric<double>(tree, "options.strength", context);
    }

/// Declares a regularizer whose only serialized field is options.coefficient.
#define Nott_REGULARIZATION_COEFFICIENT_ONLY(DescriptorType, OptionsType, name)                       \
    constexpr std::string_view descriptor_type_name(Tag<DescriptorType>) { return name; }             \
    inline PropertyTree serialize_descriptor(const DescriptorType& descriptor)                        \
    {                                                                                                 \
        return serialize_coefficient(descriptor.options.coefficient);                                 \
    }                                                                                                 \
    inline DescriptorType deserialize_descriptor(Tag<DescriptorType>, const PropertyTree& tree,       \
                                                 const std::string& context)                          \
    {                                                                                                 \
        OptionsType options;                                                                          \
        options.coefficient = read_coefficient(tree, context);                                        \
        return DescriptorType{options};                                                               \
    }

/// Same, for the ones that spell the field strength instead.
#define Nott_REGULARIZATION_STRENGTH_ONLY(DescriptorType, OptionsType, name)                          \
    constexpr std::string_view descriptor_type_name(Tag<DescriptorType>) { return name; }             \
    inline PropertyTree serialize_descriptor(const DescriptorType& descriptor)                        \
    {                                                                                                 \
        return serialize_strength(descriptor.options.strength);                                       \
    }                                                                                                 \
    inline DescriptorType deserialize_descriptor(Tag<DescriptorType>, const PropertyTree& tree,       \
                                                 const std::string& context)                          \
    {                                                                                                 \
        OptionsType options;                                                                          \
        options.strength = read_strength(tree, context);                                              \
        return DescriptorType{options};                                                               \
    }

    Nott_REGULARIZATION_COEFFICIENT_ONLY(L1Descriptor, L1Options, "l1")
    Nott_REGULARIZATION_COEFFICIENT_ONLY(L2Descriptor, L2Options, "l2")
    Nott_REGULARIZATION_COEFFICIENT_ONLY(OrthogonalityDescriptor, OrthogonalityOptions, "orthogonality")
    Nott_REGULARIZATION_COEFFICIENT_ONLY(JacobianNormDescriptor, JacobianNormOptions, "jacobian_norm")
    Nott_REGULARIZATION_COEFFICIENT_ONLY(R1Descriptor, R1Options, "r1")
    Nott_REGULARIZATION_COEFFICIENT_ONLY(R2Descriptor, R2Options, "r2")
    Nott_REGULARIZATION_COEFFICIENT_ONLY(TRADESDescriptor, TRADESOptions, "trades")
    Nott_REGULARIZATION_COEFFICIENT_ONLY(VATDescriptor, VATOptions, "vat")
    Nott_REGULARIZATION_COEFFICIENT_ONLY(SWADescriptor, SWAOptions, "swa")
    Nott_REGULARIZATION_COEFFICIENT_ONLY(FGEDescriptor, FGEOptions, "fge")
    Nott_REGULARIZATION_COEFFICIENT_ONLY(SFGEDescriptor, SFGEOptions, "sfge")

    Nott_REGULARIZATION_STRENGTH_ONLY(EWCDescriptor, EWCOptions, "ewc")
    Nott_REGULARIZATION_STRENGTH_ONLY(MASDescriptor, MASOptions, "mas")
    Nott_REGULARIZATION_STRENGTH_ONLY(NuclearNormDescriptor, NuclearNormOptions, "nuclear_norm")

#undef Nott_REGULARIZATION_COEFFICIENT_ONLY
#undef Nott_REGULARIZATION_STRENGTH_ONLY

    constexpr std::string_view descriptor_type_name(Tag<ElasticNetDescriptor>) { return "elastic_net"; }

    inline PropertyTree serialize_descriptor(const ElasticNetDescriptor& descriptor)
    {
        PropertyTree tree;
        tree.put("options.l1_coefficient", descriptor.options.l1_coefficient);
        tree.put("options.l2_coefficient", descriptor.options.l2_coefficient);
        return tree;
    }

    inline ElasticNetDescriptor deserialize_descriptor(Tag<ElasticNetDescriptor>, const PropertyTree& tree,
                                                       const std::string& context)
    {
        namespace S = ::Nott::Serialize;
        ElasticNetOptions options;
        options.l1_coefficient = S::get_numeric<double>(tree, "options.l1_coefficient", context);
        options.l2_coefficient = S::get_numeric<double>(tree, "options.l2_coefficient", context);
        return ElasticNetDescriptor{options};
    }

    constexpr std::string_view descriptor_type_name(Tag<GroupLassoDescriptor>) { return "group_lasso"; }

    inline PropertyTree serialize_descriptor(const GroupLassoDescriptor& descriptor)
    {
        const auto& options = descriptor.options;
        PropertyTree tree;
        tree.put("options.coefficient", options.coefficient);
        tree.put("options.group_dim", static_cast<std::int64_t>(options.group_dim));
        tree.put("options.epsilon", options.epsilon);
        return tree;
    }

    inline GroupLassoDescriptor deserialize_descriptor(Tag<GroupLassoDescriptor>, const PropertyTree& tree,
                                                        const std::string& context)
    {
        namespace S = ::Nott::Serialize;
        GroupLassoOptions options;
        options.coefficient = read_coefficient(tree, context);
        options.group_dim = S::get_numeric<std::int64_t>(tree, "options.group_dim", context);
        options.epsilon = S::get_numeric<double>(tree, "options.epsilon", context);
        return GroupLassoDescriptor{options};
    }

    constexpr std::string_view descriptor_type_name(Tag<StructuredL2Descriptor>) { return "structured_l2"; }

    inline PropertyTree serialize_descriptor(const StructuredL2Descriptor& descriptor)
    {
        const auto& options = descriptor.options;
        PropertyTree tree;
        tree.put("options.coefficient", options.coefficient);
        tree.put("options.group_dim", static_cast<std::int64_t>(options.group_dim));
        return tree;
    }

    inline StructuredL2Descriptor deserialize_descriptor(Tag<StructuredL2Descriptor>, const PropertyTree& tree,
                                                          const std::string& context)
    {
        StructuredL2Options options;
        options.coefficient = read_coefficient(tree, context);
        options.group_dim = ::Nott::Serialize::get_numeric<std::int64_t>(tree, "options.group_dim", context);
        return StructuredL2Descriptor{options};
    }

    constexpr std::string_view descriptor_type_name(Tag<L0HardConcreteDescriptor>) { return "l0_hard_concrete"; }

    inline PropertyTree serialize_descriptor(const L0HardConcreteDescriptor& descriptor)
    {
        const auto& options = descriptor.options;
        PropertyTree tree;
        tree.put("options.coefficient", options.coefficient);
        tree.put("options.beta", options.beta);
        tree.put("options.gamma", options.gamma);
        tree.put("options.zeta", options.zeta);
        return tree;
    }

    inline L0HardConcreteDescriptor deserialize_descriptor(Tag<L0HardConcreteDescriptor>, const PropertyTree& tree,
                                                            const std::string& context)
    {
        namespace S = ::Nott::Serialize;
        L0HardConcreteOptions options;
        options.coefficient = read_coefficient(tree, context);
        options.beta = S::get_numeric<double>(tree, "options.beta", context);
        options.gamma = S::get_numeric<double>(tree, "options.gamma", context);
        options.zeta = S::get_numeric<double>(tree, "options.zeta", context);
        return L0HardConcreteDescriptor{options};
    }

    constexpr std::string_view descriptor_type_name(Tag<SpectralNormDescriptor>) { return "spectral_norm"; }

    inline PropertyTree serialize_descriptor(const SpectralNormDescriptor& descriptor)
    {
        PropertyTree tree;
        tree.put("options.coefficient", descriptor.options.coefficient);
        tree.put("options.target", descriptor.options.target);
        return tree;
    }

    inline SpectralNormDescriptor deserialize_descriptor(Tag<SpectralNormDescriptor>, const PropertyTree& tree,
                                                          const std::string& context)
    {
        SpectralNormOptions options;
        options.coefficient = read_coefficient(tree, context);
        options.target = ::Nott::Serialize::get_numeric<double>(tree, "options.target", context);
        return SpectralNormDescriptor{options};
    }

    constexpr std::string_view descriptor_type_name(Tag<MaxNormDescriptor>) { return "max_norm"; }

    inline PropertyTree serialize_descriptor(const MaxNormDescriptor& descriptor)
    {
        const auto& options = descriptor.options;
        PropertyTree tree;
        tree.put("options.coefficient", options.coefficient);
        tree.put("options.max_norm", options.max_norm);
        tree.put("options.dim", static_cast<std::int64_t>(options.dim));
        return tree;
    }

    inline MaxNormDescriptor deserialize_descriptor(Tag<MaxNormDescriptor>, const PropertyTree& tree,
                                                     const std::string& context)
    {
        namespace S = ::Nott::Serialize;
        MaxNormOptions options;
        options.coefficient = read_coefficient(tree, context);
        options.max_norm = S::get_numeric<double>(tree, "options.max_norm", context);
        options.dim = S::get_numeric<std::int64_t>(tree, "options.dim", context);
        return MaxNormDescriptor{options};
    }

    constexpr std::string_view descriptor_type_name(Tag<KLSparsityDescriptor>) { return "kl_sparsity"; }

    inline PropertyTree serialize_descriptor(const KLSparsityDescriptor& descriptor)
    {
        const auto& options = descriptor.options;
        PropertyTree tree;
        tree.put("options.coefficient", options.coefficient);
        tree.put("options.target", options.target);
        tree.put("options.epsilon", options.epsilon);
        return tree;
    }

    inline KLSparsityDescriptor deserialize_descriptor(Tag<KLSparsityDescriptor>, const PropertyTree& tree,
                                                        const std::string& context)
    {
        namespace S = ::Nott::Serialize;
        KLSparsityOptions options;
        options.coefficient = read_coefficient(tree, context);
        options.target = S::get_numeric<double>(tree, "options.target", context);
        options.epsilon = S::get_numeric<double>(tree, "options.epsilon", context);
        return KLSparsityDescriptor{options};
    }

    constexpr std::string_view descriptor_type_name(Tag<DeCovDescriptor>) { return "decov"; }

    inline PropertyTree serialize_descriptor(const DeCovDescriptor& descriptor)
    {
        PropertyTree tree;
        tree.put("options.coefficient", descriptor.options.coefficient);
        tree.put("options.epsilon", descriptor.options.epsilon);
        return tree;
    }

    inline DeCovDescriptor deserialize_descriptor(Tag<DeCovDescriptor>, const PropertyTree& tree,
                                                   const std::string& context)
    {
        DeCovOptions options;
        options.coefficient = read_coefficient(tree, context);
        options.epsilon = ::Nott::Serialize::get_numeric<double>(tree, "options.epsilon", context);
        return DeCovDescriptor{options};
    }

    constexpr std::string_view descriptor_type_name(Tag<CenteringVarianceDescriptor>) { return "centering_variance"; }

    inline PropertyTree serialize_descriptor(const CenteringVarianceDescriptor& descriptor)
    {
        PropertyTree tree;
        tree.put("options.coefficient", descriptor.options.coefficient);
        tree.put("options.target_std", descriptor.options.target_std);
        return tree;
    }

    inline CenteringVarianceDescriptor deserialize_descriptor(Tag<CenteringVarianceDescriptor>,
                                                               const PropertyTree& tree, const std::string& context)
    {
        CenteringVarianceOptions options;
        options.coefficient = read_coefficient(tree, context);
        options.target_std = ::Nott::Serialize::get_numeric<double>(tree, "options.target_std", context);
        return CenteringVarianceDescriptor{options};
    }

    constexpr std::string_view descriptor_type_name(Tag<WGANGPDescriptor>) { return "wgan_gp"; }

    inline PropertyTree serialize_descriptor(const WGANGPDescriptor& descriptor)
    {
        PropertyTree tree;
        tree.put("options.coefficient", descriptor.options.coefficient);
        tree.put("options.target", descriptor.options.target);
        return tree;
    }

    inline WGANGPDescriptor deserialize_descriptor(Tag<WGANGPDescriptor>, const PropertyTree& tree,
                                                    const std::string& context)
    {
        WGANGPOptions options;
        options.coefficient = read_coefficient(tree, context);
        options.target = ::Nott::Serialize::get_numeric<double>(tree, "options.target", context);
        return WGANGPDescriptor{options};
    }

    constexpr std::string_view descriptor_type_name(Tag<SIDescriptor>) { return "si"; }

    inline PropertyTree serialize_descriptor(const SIDescriptor& descriptor)
    {
        PropertyTree tree;
        tree.put("options.strength", descriptor.options.strength);
        tree.put("options.damping", descriptor.options.damping);
        return tree;
    }

    inline SIDescriptor deserialize_descriptor(Tag<SIDescriptor>, const PropertyTree& tree,
                                                const std::string& context)
    {
        SIOptions options;
        options.strength = read_strength(tree, context);
        options.damping = ::Nott::Serialize::get_numeric<double>(tree, "options.damping", context);
        return SIDescriptor{options};
    }

    constexpr std::string_view descriptor_type_name(Tag<SWAGDescriptor>) { return "swag"; }

    inline PropertyTree serialize_descriptor(const SWAGDescriptor& descriptor)
    {
        const auto& options = descriptor.options;
        PropertyTree tree;
        tree.put("options.coefficient", options.coefficient);
        tree.put("options.variance_epsilon", options.variance_epsilon);
        tree.put("options.start_step", static_cast<std::uint64_t>(options.start_step));
        tree.put("options.accumulation_stride", static_cast<std::uint64_t>(options.accumulation_stride));
        tree.put("options.max_snapshots", static_cast<std::uint64_t>(options.max_snapshots));
        return tree;
    }

    inline SWAGDescriptor deserialize_descriptor(Tag<SWAGDescriptor>, const PropertyTree& tree,
                                                  const std::string& context)
    {
        namespace S = ::Nott::Serialize;
        SWAGOptions options;
        options.coefficient = read_coefficient(tree, context);
        options.variance_epsilon = S::get_numeric<double>(tree, "options.variance_epsilon", context);
        options.start_step = static_cast<std::size_t>(S::get_numeric<std::uint64_t>(tree, "options.start_step", context));
        options.accumulation_stride =
            static_cast<std::size_t>(S::get_numeric<std::uint64_t>(tree, "options.accumulation_stride", context));
        options.max_snapshots =
            static_cast<std::size_t>(S::get_numeric<std::uint64_t>(tree, "options.max_snapshots", context));
        return SWAGDescriptor{options};
    }
}

namespace Nott::Regularization {
    inline ::Nott::Serialize::PropertyTree serialize_regularization(const Descriptor& descriptor)
    {
        return ::Nott::Serialize::serialize_descriptor(descriptor);
    }

    inline Descriptor deserialize_regularization(const ::Nott::Serialize::PropertyTree& tree,
                                                 const std::string& context)
    {
        return ::Nott::Serialize::deserialize_variant<Descriptor>(tree, context);
    }
}

#endif // Nott_REGULARIZATION_SERIALIZE_HPP
