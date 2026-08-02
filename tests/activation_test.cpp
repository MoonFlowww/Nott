#include "test_prelude.hpp"

#include <torch/torch.h>
#include "../src/activation/apply.hpp"

using namespace Nott;

namespace {
    constexpr std::array kAllActivations = {
        Activation::Type::Identity, Activation::Type::ReLU, Activation::Type::Sigmoid,
        Activation::Type::Tanh, Activation::Type::LeakyReLU, Activation::Type::Softmax,
        Activation::Type::SiLU, Activation::Type::GeLU, Activation::Type::GLU,
        Activation::Type::SwiGLU, Activation::Type::dSiLU, Activation::Type::PSiLU,
        Activation::Type::Mish, Activation::Type::Swish,
    };

    // GLU/SwiGLU gate on the last dimension, halving it.
    bool halves_last_dim(Activation::Type type) {
        return type == Activation::Type::GLU || type == Activation::Type::SwiGLU;
    }
}

TEST_CASE("activation: every registered type produces finite, correctly-shaped output and backprops") {
    for (auto type : kAllActivations) {
        CAPTURE(static_cast<int>(type));
        auto input = torch::randn({4, 8}, torch::requires_grad(true));
        auto output = Activation::Details::apply(type, input);

        REQUIRE(output.defined());
        auto expected_last_dim = halves_last_dim(type) ? 4 : 8;
        CHECK(output.size(0) == 4);
        CHECK(output.size(1) == expected_last_dim);
        CHECK(torch::isfinite(output).all().item<bool>());

        output.sum().backward();
        CHECK(input.grad().defined());
        CHECK(torch::isfinite(input.grad()).all().item<bool>());
    }
}

TEST_CASE("activation: golden values on known inputs") {
    auto x = torch::tensor({-1.0, 0.0, 1.0, 2.0});

    SUBCASE("ReLU zeroes negatives, passes positives through") {
        auto y = Activation::Details::apply(Activation::Type::ReLU, x);
        CHECK(y[0].item<double>() == doctest::Approx(0.0));
        CHECK(y[2].item<double>() == doctest::Approx(1.0));
        CHECK(y[3].item<double>() == doctest::Approx(2.0));
    }

    SUBCASE("Sigmoid(0) == 0.5") {
        auto y = Activation::Details::apply(Activation::Type::Sigmoid, x);
        CHECK(y[1].item<double>() == doctest::Approx(0.5));
    }

    SUBCASE("Tanh(0) == 0") {
        auto y = Activation::Details::apply(Activation::Type::Tanh, x);
        CHECK(y[1].item<double>() == doctest::Approx(0.0));
    }

    SUBCASE("Identity is a no-op") {
        auto y = Activation::Details::apply(Activation::Type::Identity, x);
        CHECK(torch::allclose(x, y));
    }

    SUBCASE("Softmax rows sum to 1") {
        auto batch = torch::randn({3, 5});
        auto y = Activation::Details::apply(Activation::Type::Softmax, batch);
        auto sums = y.sum(-1);
        CHECK(torch::allclose(sums, torch::ones({3}), 1e-5, 1e-5));
    }
}
