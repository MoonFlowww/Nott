#ifndef Nott_COMMON_LOCAL_SERIALIZE_HPP
#define Nott_COMMON_LOCAL_SERIALIZE_HPP
/// Serialization for LocalConfig, the per module optimizer/loss/regularization
/// override. Sits next to local.hpp and pulls in the three modules it embeds.
#include <string>

#include "../loss/serialize.hpp"
#include "../optimizer/serialize.hpp"
#include "../regularization/serialize.hpp"
#include "local.hpp"
#include "serialize.hpp"

namespace Nott {
    inline ::Nott::Serialize::PropertyTree serialize_local_config(const LocalConfig& config)
    {
        ::Nott::Serialize::PropertyTree tree;
        if (config.optimizer) {
            tree.add_child("optimizer", Optimizer::serialize_optimizer(*config.optimizer));
        }
        if (config.loss) {
            tree.add_child("loss", Loss::serialize_loss(*config.loss));
        }
        ::Nott::Serialize::PropertyTree regularization;
        for (const auto& descriptor : config.regularization) {
            regularization.push_back({"", Regularization::serialize_regularization(descriptor)});
        }
        tree.add_child("regularization", regularization);
        return tree;
    }

    inline LocalConfig deserialize_local_config(const ::Nott::Serialize::PropertyTree& tree,
                                                const std::string& context)
    {
        LocalConfig config;
        if (const auto optimizer = tree.get_child_optional("optimizer")) {
            config.optimizer = Optimizer::deserialize_optimizer(*optimizer, context + " optimizer");
        }
        if (const auto loss = tree.get_child_optional("loss")) {
            config.loss = Loss::deserialize_loss(*loss, context + " loss");
        }
        if (const auto regularization = tree.get_child_optional("regularization")) {
            for (const auto& node : *regularization) {
                config.regularization.push_back(
                    Regularization::deserialize_regularization(node.second, context + " regularization"));
            }
        }
        return config;
    }
}

#endif // Nott_COMMON_LOCAL_SERIALIZE_HPP
