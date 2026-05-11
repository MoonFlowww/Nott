#ifndef Nott_BATCHNORM_HPP
#define Nott_BATCHNORM_HPP
/// https://arxiv.org/pdf/1502.03167
#include <cstdint>

#include <stdexcept>
#include <string>
#include <utility>

#include <torch/nn/module.h>
#include <torch/nn/options/batchnorm.h>

#include "../../common/local.hpp"
#include "../../activation/activation.hpp"
#include "../../initialization/initialization.hpp"
#include "../../initialization/apply.hpp"
#include "../../initialization/initialization.hpp"
#include "../registry.hpp"

namespace Nott::Layer::Details {

    struct BatchNorm2dOptions {
        std::int64_t num_features{};
        double eps{1e-5};
        double momentum{0.1};
        bool affine{true};
        bool track_running_stats{true};
    };

    struct BatchNorm2dDescriptor {
        BatchNorm2dOptions options{};
        ::Nott::Activation::Descriptor activation{::Nott::Activation::Identity};
        ::Nott::Initialization::Descriptor initialization{::Nott::Initialization::Default};
        ::Nott::LocalConfig local{};
    };

    template <class Owner>
    RegisteredLayer build_registered_layer(Owner& owner, const BatchNorm2dDescriptor& descriptor, std::size_t index)
    {
        if (descriptor.options.num_features <= 0) {
            throw std::invalid_argument("BatchNorm2d requires a positive number of features.");
        }

        auto options = torch::nn::BatchNorm2dOptions(descriptor.options.num_features)
                            .eps(descriptor.options.eps)
                            .momentum(descriptor.options.momentum)
                            .affine(descriptor.options.affine)
                            .track_running_stats(descriptor.options.track_running_stats);

        auto module = owner.register_module("batchnorm2d_" + std::to_string(index), torch::nn::BatchNorm2d(options));
        ::Nott::Initialization::Details::apply_module_initialization(module, descriptor);

        RegisteredLayer registered_layer{};
        registered_layer.activation = descriptor.activation.type;
        registered_layer.module = to_shared_module_ptr(module);
        registered_layer.local = descriptor.local;
        registered_layer.bind_module_forward(module.get());
        return registered_layer;
    }

}

#endif //Nott_BATCHNORM_HPP