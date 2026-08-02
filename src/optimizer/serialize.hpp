#ifndef Nott_OPTIMIZER_SERIALIZE_HPP
#define Nott_OPTIMIZER_SERIALIZE_HPP
/// Serialization for Optimizer::Descriptor.
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>

#include "../common/serialize.hpp"
#include "optimizer.hpp"

/// Overloads live in the namespace the descriptors are declared in, which is
/// what ADL searches from the generic dispatch in common/serialize.hpp.
namespace Nott::Optimizer::Details {
    using ::Nott::Serialize::PropertyTree;
    using ::Nott::Serialize::Tag;

    inline std::string manifold_kind_to_string(Details::ManifoldKind kind)
    {
        switch (kind) {
            case Details::ManifoldKind::Euclidean: return "euclidean";
            case Details::ManifoldKind::UnitSphere: return "unit_sphere";
            case Details::ManifoldKind::Stiefel: return "stiefel";
        }
        throw std::logic_error("Unknown Muon manifold kind encountered during serialization.");
    }

    inline Details::ManifoldKind manifold_kind_from_string(const std::string& value, const std::string& context)
    {
        const auto lowered = ::Nott::Serialize::to_lower(value);
        if (lowered == "euclidean") return Details::ManifoldKind::Euclidean;
        if (lowered == "unit_sphere") return Details::ManifoldKind::UnitSphere;
        if (lowered == "stiefel") return Details::ManifoldKind::Stiefel;
        std::ostringstream message;
        message << "Unknown Muon manifold kind '" << value << "' in " << context;
        throw std::runtime_error(message.str());
    }

    constexpr std::string_view descriptor_type_name(Tag<Details::SGDDescriptor>) { return "sgd"; }

    inline PropertyTree serialize_descriptor(const Details::SGDDescriptor& descriptor)
    {
        const auto& options = descriptor.options;
        PropertyTree tree;
        tree.put("options.learning_rate", options.learning_rate);
        tree.put("options.momentum", options.momentum);
        tree.put("options.dampening", options.dampening);
        tree.put("options.weight_decay", options.weight_decay);
        tree.put("options.nesterov", options.nesterov);
        tree.put("options.maximize", options.maximize);
        return tree;
    }

    inline Details::SGDDescriptor deserialize_descriptor(Tag<Details::SGDDescriptor>,
                                                         const PropertyTree& tree, const std::string& context)
    {
        namespace S = ::Nott::Serialize;
        Details::SGDOptions options;
        options.learning_rate = S::get_numeric<double>(tree, "options.learning_rate", context);
        options.momentum = S::get_numeric<double>(tree, "options.momentum", context);
        options.dampening = S::get_numeric<double>(tree, "options.dampening", context);
        options.weight_decay = S::get_numeric<double>(tree, "options.weight_decay", context);
        options.nesterov = S::get_boolean(tree, "options.nesterov", context);
        options.maximize = S::get_boolean(tree, "options.maximize", context);
        return Details::SGDDescriptor{options};
    }

    constexpr std::string_view descriptor_type_name(Tag<Details::AdamDescriptor>) { return "adam"; }

    inline PropertyTree serialize_descriptor(const Details::AdamDescriptor& descriptor)
    {
        const auto& options = descriptor.options;
        PropertyTree tree;
        tree.put("options.learning_rate", options.learning_rate);
        tree.put("options.beta1", options.beta1);
        tree.put("options.beta2", options.beta2);
        tree.put("options.eps", options.eps);
        tree.put("options.weight_decay", options.weight_decay);
        tree.put("options.amsgrad", options.amsgrad);
        return tree;
    }

    inline Details::AdamDescriptor deserialize_descriptor(Tag<Details::AdamDescriptor>,
                                                          const PropertyTree& tree, const std::string& context)
    {
        namespace S = ::Nott::Serialize;
        Details::AdamOptions options;
        options.learning_rate = S::get_numeric<double>(tree, "options.learning_rate", context);
        options.beta1 = S::get_numeric<double>(tree, "options.beta1", context);
        options.beta2 = S::get_numeric<double>(tree, "options.beta2", context);
        options.eps = S::get_numeric<double>(tree, "options.eps", context);
        options.weight_decay = S::get_numeric<double>(tree, "options.weight_decay", context);
        options.amsgrad = S::get_boolean(tree, "options.amsgrad", context);
        return Details::AdamDescriptor{options};
    }

    constexpr std::string_view descriptor_type_name(Tag<Details::AdamWDescriptor>) { return "adamw"; }

    inline PropertyTree serialize_descriptor(const Details::AdamWDescriptor& descriptor)
    {
        const auto& options = descriptor.options;
        PropertyTree tree;
        tree.put("options.learning_rate", options.learning_rate);
        tree.put("options.beta1", options.beta1);
        tree.put("options.beta2", options.beta2);
        tree.put("options.eps", options.eps);
        tree.put("options.weight_decay", options.weight_decay);
        tree.put("options.amsgrad", options.amsgrad);
        return tree;
    }

    inline Details::AdamWDescriptor deserialize_descriptor(Tag<Details::AdamWDescriptor>,
                                                           const PropertyTree& tree, const std::string& context)
    {
        namespace S = ::Nott::Serialize;
        Details::AdamWOptions options;
        options.learning_rate = S::get_numeric<double>(tree, "options.learning_rate", context);
        options.beta1 = S::get_numeric<double>(tree, "options.beta1", context);
        options.beta2 = S::get_numeric<double>(tree, "options.beta2", context);
        options.eps = S::get_numeric<double>(tree, "options.eps", context);
        options.weight_decay = S::get_numeric<double>(tree, "options.weight_decay", context);
        options.amsgrad = S::get_boolean(tree, "options.amsgrad", context);
        return Details::AdamWDescriptor{options};
    }

    constexpr std::string_view descriptor_type_name(Tag<Details::SophiaGDescriptor>) { return "sophia_g"; }

    inline PropertyTree serialize_descriptor(const Details::SophiaGDescriptor& descriptor)
    {
        const auto& options = descriptor.options;
        PropertyTree tree;
        tree.put("options.learning_rate", options.lr());
        tree.put("options.beta1", options.beta1());
        tree.put("options.beta2", options.beta2());
        tree.put("options.rho", options.rho());
        tree.put("options.eps", options.eps());
        tree.put("options.weight_decay", options.weight_decay());
        tree.put("options.clip", options.clip());
        tree.put("options.hessian_update_interval", options.hessian_update_interval());
        return tree;
    }

    inline Details::SophiaGDescriptor deserialize_descriptor(Tag<Details::SophiaGDescriptor>,
                                                             const PropertyTree& tree, const std::string& context)
    {
        namespace S = ::Nott::Serialize;
        Details::SophiaGOptions options;
        options.lr(S::get_numeric<double>(tree, "options.learning_rate", context));
        options.beta1(S::get_numeric<double>(tree, "options.beta1", context));
        options.beta2(S::get_numeric<double>(tree, "options.beta2", context));
        options.rho(S::get_numeric<double>(tree, "options.rho", context));
        options.eps(S::get_numeric<double>(tree, "options.eps", context));
        options.weight_decay(S::get_numeric<double>(tree, "options.weight_decay", context));
        options.clip(S::get_numeric<double>(tree, "options.clip", context));
        options.hessian_update_interval(
            S::get_numeric<std::int64_t>(tree, "options.hessian_update_interval", context));
        return Details::SophiaGDescriptor{options};
    }

    constexpr std::string_view descriptor_type_name(Tag<Details::SophiaHDescriptor>) { return "sophia_h"; }

    inline PropertyTree serialize_descriptor(const Details::SophiaHDescriptor& descriptor)
    {
        const auto& options = descriptor.options;
        PropertyTree tree;
        tree.put("options.learning_rate", options.lr());
        tree.put("options.beta1", options.beta1());
        tree.put("options.beta2", options.beta2());
        tree.put("options.rho", options.rho());
        tree.put("options.eps", options.eps());
        tree.put("options.weight_decay", options.weight_decay());
        tree.put("options.clip", options.clip());
        tree.put("options.hessian_update_interval", options.hessian_update_interval());
        return tree;
    }

    inline Details::SophiaHDescriptor deserialize_descriptor(Tag<Details::SophiaHDescriptor>,
                                                             const PropertyTree& tree, const std::string& context)
    {
        namespace S = ::Nott::Serialize;
        Details::SophiaHOptions options;
        options.lr(S::get_numeric<double>(tree, "options.learning_rate", context));
        options.beta1(S::get_numeric<double>(tree, "options.beta1", context));
        options.beta2(S::get_numeric<double>(tree, "options.beta2", context));
        options.rho(S::get_numeric<double>(tree, "options.rho", context));
        options.eps(S::get_numeric<double>(tree, "options.eps", context));
        options.weight_decay(S::get_numeric<double>(tree, "options.weight_decay", context));
        options.clip(S::get_numeric<double>(tree, "options.clip", context));
        options.hessian_update_interval(
            S::get_numeric<std::int64_t>(tree, "options.hessian_update_interval", context));
        return Details::SophiaHDescriptor{options};
    }

    constexpr std::string_view descriptor_type_name(Tag<Details::MuonDescriptor>) { return "muon"; }

    inline PropertyTree serialize_descriptor(const Details::MuonDescriptor& descriptor)
    {
        const auto& options = descriptor.options;
        PropertyTree tree;
        tree.put("options.learning_rate", options.lr());
        tree.put("options.beta", options.beta());
        tree.put("options.weight_decay", options.weight_decay());
        tree.put("options.eps", options.eps());
        tree.put("options.max_update_norm", options.max_update_norm());
        return tree;
    }

    inline Details::MuonDescriptor deserialize_descriptor(Tag<Details::MuonDescriptor>,
                                                          const PropertyTree& tree, const std::string& context)
    {
        namespace S = ::Nott::Serialize;
        Details::MuonOptions options;
        options.lr(S::get_numeric<double>(tree, "options.learning_rate", context));
        options.beta(S::get_numeric<double>(tree, "options.beta", context));
        options.weight_decay(S::get_numeric<double>(tree, "options.weight_decay", context));
        options.eps(S::get_numeric<double>(tree, "options.eps", context));
        options.max_update_norm(S::get_numeric<double>(tree, "options.max_update_norm", context));
        return Details::MuonDescriptor{options};
    }

    constexpr std::string_view descriptor_type_name(Tag<Details::AdaMuonDescriptor>) { return "ada_muon"; }

    inline PropertyTree serialize_descriptor(const Details::AdaMuonDescriptor& descriptor)
    {
        const auto& options = descriptor.options;
        PropertyTree tree;
        tree.put("options.learning_rate", options.lr());
        tree.put("options.beta", options.beta());
        tree.put("options.beta2", options.beta2());
        tree.put("options.weight_decay", options.weight_decay());
        tree.put("options.eps", options.eps());
        tree.put("options.max_update_norm", options.max_update_norm());
        return tree;
    }

    inline Details::AdaMuonDescriptor deserialize_descriptor(Tag<Details::AdaMuonDescriptor>,
                                                             const PropertyTree& tree, const std::string& context)
    {
        namespace S = ::Nott::Serialize;
        Details::AdaMuonOptions options;
        options.lr(S::get_numeric<double>(tree, "options.learning_rate", context));
        options.beta(S::get_numeric<double>(tree, "options.beta", context));
        options.beta2(S::get_numeric<double>(tree, "options.beta2", context));
        options.weight_decay(S::get_numeric<double>(tree, "options.weight_decay", context));
        options.eps(S::get_numeric<double>(tree, "options.eps", context));
        options.max_update_norm(S::get_numeric<double>(tree, "options.max_update_norm", context));
        return Details::AdaMuonDescriptor{options};
    }

    constexpr std::string_view descriptor_type_name(Tag<Details::MuonManifoldDescriptor>) { return "muon_manifold"; }

    inline PropertyTree serialize_descriptor(const Details::MuonManifoldDescriptor& descriptor)
    {
        const auto& options = descriptor.options;
        PropertyTree tree;
        tree.put("options.learning_rate", options.lr());
        tree.put("options.beta", options.beta());
        tree.put("options.beta2", options.beta2());
        tree.put("options.weight_decay", options.weight_decay());
        tree.put("options.eps", options.eps());
        tree.put("options.max_update_norm", options.max_update_norm());
        tree.put("options.retraction_epsilon", options.retraction_epsilon());
        tree.put("options.renormalize", options.renormalize());
        tree.put("options.manifold", manifold_kind_to_string(options.manifold()));
        return tree;
    }

    inline Details::MuonManifoldDescriptor deserialize_descriptor(Tag<Details::MuonManifoldDescriptor>,
                                                                  const PropertyTree& tree, const std::string& context)
    {
        namespace S = ::Nott::Serialize;
        Details::MuonManifoldOptions options;
        options.lr(S::get_numeric<double>(tree, "options.learning_rate", context));
        options.beta(S::get_numeric<double>(tree, "options.beta", context));
        options.beta2(S::get_numeric<double>(tree, "options.beta2", context));
        options.weight_decay(S::get_numeric<double>(tree, "options.weight_decay", context));
        options.eps(S::get_numeric<double>(tree, "options.eps", context));
        options.max_update_norm(S::get_numeric<double>(tree, "options.max_update_norm", context));
        options.retraction_epsilon(S::get_numeric<double>(tree, "options.retraction_epsilon", context));
        options.renormalize(S::get_boolean(tree, "options.renormalize", context));
        options.manifold(manifold_kind_from_string(S::get_string(tree, "options.manifold", context), context));
        return Details::MuonManifoldDescriptor{options};
    }

    constexpr std::string_view descriptor_type_name(Tag<Details::AdafactorDescriptor>) { return "adafactor"; }

    inline PropertyTree serialize_descriptor(const Details::AdafactorDescriptor& descriptor)
    {
        const auto& options = descriptor.options;
        PropertyTree tree;
        tree.put("options.learning_rate", options.lr());
        tree.put("options.eps1", options.eps1());
        tree.put("options.eps2", options.eps2());
        tree.put("options.clip_threshold", options.clip_threshold());
        tree.put("options.decay_rate", options.decay_rate());
        tree.put("options.beta1", options.beta1());
        tree.put("options.use_first_moment", options.use_first_moment());
        tree.put("options.weight_decay", options.weight_decay());
        tree.put("options.scale_parameter", options.scale_parameter());
        tree.put("options.relative_step", options.relative_step());
        tree.put("options.warmup_init", options.warmup_init());
        return tree;
    }

    inline Details::AdafactorDescriptor deserialize_descriptor(Tag<Details::AdafactorDescriptor>,
                                                               const PropertyTree& tree, const std::string& context)
    {
        namespace S = ::Nott::Serialize;
        Details::AdafactorOptions options;
        options.lr(S::get_numeric<double>(tree, "options.learning_rate", context));
        options.eps1(S::get_numeric<double>(tree, "options.eps1", context));
        options.eps2(S::get_numeric<double>(tree, "options.eps2", context));
        options.clip_threshold(S::get_numeric<double>(tree, "options.clip_threshold", context));
        options.decay_rate(S::get_numeric<double>(tree, "options.decay_rate", context));
        options.beta1(S::get_numeric<double>(tree, "options.beta1", context));
        /// use_first_moment is bool and weight_decay is double. The two were read
        /// with each other's types before, so any adafactor load threw.
        options.use_first_moment(S::get_boolean(tree, "options.use_first_moment", context));
        options.weight_decay(S::get_numeric<double>(tree, "options.weight_decay", context));
        options.scale_parameter(S::get_boolean(tree, "options.scale_parameter", context));
        options.relative_step(S::get_boolean(tree, "options.relative_step", context));
        options.warmup_init(S::get_boolean(tree, "options.warmup_init", context));
        return Details::AdafactorDescriptor{options};
    }

    constexpr std::string_view descriptor_type_name(Tag<Details::AdagradDescriptor>) { return "adagrad"; }

    inline PropertyTree serialize_descriptor(const Details::AdagradDescriptor& descriptor)
    {
        const auto& options = descriptor.options;
        PropertyTree tree;
        tree.put("options.learning_rate", options.learning_rate);
        tree.put("options.lr_decay", options.lr_decay);
        tree.put("options.weight_decay", options.weight_decay);
        tree.put("options.initial_accumulator_value", options.initial_accumulator_value);
        tree.put("options.eps", options.eps);
        return tree;
    }

    inline Details::AdagradDescriptor deserialize_descriptor(Tag<Details::AdagradDescriptor>,
                                                             const PropertyTree& tree, const std::string& context)
    {
        namespace S = ::Nott::Serialize;
        Details::AdagradOptions options;
        options.learning_rate = S::get_numeric<double>(tree, "options.learning_rate", context);
        options.lr_decay = S::get_numeric<double>(tree, "options.lr_decay", context);
        options.weight_decay = S::get_numeric<double>(tree, "options.weight_decay", context);
        options.initial_accumulator_value =
            S::get_numeric<double>(tree, "options.initial_accumulator_value", context);
        options.eps = S::get_numeric<double>(tree, "options.eps", context);
        return Details::AdagradDescriptor{options};
    }

    constexpr std::string_view descriptor_type_name(Tag<Details::LAMBDescriptor>) { return "lamb"; }

    inline PropertyTree serialize_descriptor(const Details::LAMBDescriptor& descriptor)
    {
        const auto& options = descriptor.options;
        PropertyTree tree;
        tree.put("options.learning_rate", options.lr());
        tree.put("options.beta1", options.beta1());
        tree.put("options.beta2", options.beta2());
        tree.put("options.eps", options.eps());
        tree.put("options.weight_decay", options.weight_decay());
        tree.put("options.adam", options.adam());
        return tree;
    }

    inline Details::LAMBDescriptor deserialize_descriptor(Tag<Details::LAMBDescriptor>,
                                                          const PropertyTree& tree, const std::string& context)
    {
        namespace S = ::Nott::Serialize;
        Details::LAMBOptions options;
        /// Reads options.learning_rate, which is the key that gets written. The
        /// old reader looked for options.lr and threw on every lamb load.
        options.lr(S::get_numeric<double>(tree, "options.learning_rate", context));
        options.beta1(S::get_numeric<double>(tree, "options.beta1", context));
        options.beta2(S::get_numeric<double>(tree, "options.beta2", context));
        options.eps(S::get_numeric<double>(tree, "options.eps", context));
        options.weight_decay(S::get_numeric<double>(tree, "options.weight_decay", context));
        options.adam(S::get_boolean(tree, "options.adam", context));
        return Details::LAMBDescriptor{options};
    }

    constexpr std::string_view descriptor_type_name(Tag<Details::LionDescriptor>) { return "lion"; }

    inline PropertyTree serialize_descriptor(const Details::LionDescriptor& descriptor)
    {
        const auto& options = descriptor.options;
        PropertyTree tree;
        tree.put("options.learning_rate", options.lr());
        tree.put("options.beta1", options.beta1());
        tree.put("options.beta2", options.beta2());
        tree.put("options.weight_decay", options.weight_decay());
        return tree;
    }

    inline Details::LionDescriptor deserialize_descriptor(Tag<Details::LionDescriptor>,
                                                          const PropertyTree& tree, const std::string& context)
    {
        namespace S = ::Nott::Serialize;
        Details::LionOptions options;
        /// Same key mismatch as lamb, same fix.
        options.lr(S::get_numeric<double>(tree, "options.learning_rate", context));
        options.beta1(S::get_numeric<double>(tree, "options.beta1", context));
        options.beta2(S::get_numeric<double>(tree, "options.beta2", context));
        options.weight_decay(S::get_numeric<double>(tree, "options.weight_decay", context));
        return Details::LionDescriptor{options};
    }

    constexpr std::string_view descriptor_type_name(Tag<Details::RMSpropDescriptor>) { return "rmsprop"; }

    inline PropertyTree serialize_descriptor(const Details::RMSpropDescriptor& descriptor)
    {
        const auto& options = descriptor.options;
        PropertyTree tree;
        tree.put("options.learning_rate", options.learning_rate);
        tree.put("options.alpha", options.alpha);
        tree.put("options.eps", options.eps);
        tree.put("options.weight_decay", options.weight_decay);
        tree.put("options.momentum", options.momentum);
        tree.put("options.centered", options.centered);
        return tree;
    }

    inline Details::RMSpropDescriptor deserialize_descriptor(Tag<Details::RMSpropDescriptor>,
                                                             const PropertyTree& tree, const std::string& context)
    {
        namespace S = ::Nott::Serialize;
        Details::RMSpropOptions options;
        options.learning_rate = S::get_numeric<double>(tree, "options.learning_rate", context);
        options.alpha = S::get_numeric<double>(tree, "options.alpha", context);
        options.eps = S::get_numeric<double>(tree, "options.eps", context);
        options.weight_decay = S::get_numeric<double>(tree, "options.weight_decay", context);
        options.momentum = S::get_numeric<double>(tree, "options.momentum", context);
        options.centered = S::get_boolean(tree, "options.centered", context);
        return Details::RMSpropDescriptor{options};
    }

}

namespace Nott::Optimizer {
    using Details::manifold_kind_from_string;
    using Details::manifold_kind_to_string;

    inline ::Nott::Serialize::PropertyTree serialize_optimizer(const Descriptor& descriptor)
    {
        return ::Nott::Serialize::serialize_descriptor(descriptor);
    }

    inline Descriptor deserialize_optimizer(const ::Nott::Serialize::PropertyTree& tree, const std::string& context)
    {
        return ::Nott::Serialize::deserialize_variant<Descriptor>(tree, context);
    }
}

#endif // Nott_OPTIMIZER_SERIALIZE_HPP
