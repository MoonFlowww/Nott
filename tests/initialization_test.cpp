#include "third_party/doctest.h"

#include <torch/torch.h>
#include "../src/initialization/apply.hpp"

using namespace Nott;

namespace {
    constexpr std::array kAllInitializations = {
        Initialization::Type::Default, Initialization::Type::XavierNormal, Initialization::Type::XavierUniform,
        Initialization::Type::HeNormal, Initialization::Type::HeUniform, Initialization::Type::ZeroBias,
        Initialization::Type::Dirac, Initialization::Type::Lyapunov,
    };
}

TEST_CASE("initialization: every registered type leaves weights finite and biases zeroed where applicable") {
    for (auto type : kAllInitializations) {
        CAPTURE(static_cast<int>(type));
        // Dirac requires weight.dim() >= 3 to act (see apply.hpp); use a Conv1d-shaped module.
        auto conv = torch::nn::Conv1d(torch::nn::Conv1dOptions(4, 8, 3).bias(true));
        Initialization::Descriptor descriptor{type};
        Initialization::Details::apply_module_initialization(conv, descriptor);

        CHECK(torch::isfinite(conv->weight).all().item<bool>());
        if (conv->bias.defined()) {
            CHECK(torch::isfinite(conv->bias).all().item<bool>());
        }
        if (type == Initialization::Type::ZeroBias) {
            CHECK(torch::allclose(conv->bias, torch::zeros_like(conv->bias)));
        }
    }
}

TEST_CASE("initialization: Xavier/He actually perturb weights away from their pre-init values") {
    auto make_linear = []() { return torch::nn::Linear(torch::nn::LinearOptions(64, 64).bias(true)); };

    for (auto type : {Initialization::Type::XavierNormal, Initialization::Type::XavierUniform,
                       Initialization::Type::HeNormal, Initialization::Type::HeUniform, Initialization::Type::Lyapunov}) {
        CAPTURE(static_cast<int>(type));
        auto linear = make_linear();
        auto before = linear->weight.clone();
        Initialization::Descriptor descriptor{type};
        Initialization::Details::apply_module_initialization(linear, descriptor);

        CHECK_FALSE(torch::allclose(before, linear->weight));
        CHECK(torch::isfinite(linear->weight).all().item<bool>());
        // A freshly initialized 64x64 layer should not collapse to all-zero.
        CHECK(linear->weight.abs().sum().item<double>() > 0.0);
    }
}
