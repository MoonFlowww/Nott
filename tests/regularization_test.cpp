#include "third_party/doctest.h"

#include <torch/torch.h>
#include <variant>
#include "../src/regularization/apply.hpp"

using namespace Nott;

namespace {
    // Every simple (single-tensor) regularizer descriptor. EWC/MAS/SI are excluded: their
    // penalty() only accepts an extra task-importance state that only exists mid-training,
    // so they're exercised indirectly, not here.
    // ponytail: no dedicated EWC/MAS/SI unit test; add one if a bug is ever reported there.
    const std::vector<Regularization::Descriptor> kSimpleRegularizers = {
        Regularization::L1(), Regularization::ElasticNet(), Regularization::GroupLasso(),
        Regularization::StructuredL2(), Regularization::L0HardConcrete(), Regularization::Orthogonality(),
        Regularization::SpectralNorm(), Regularization::MaxNorm(), Regularization::KLSparsity(),
        Regularization::DeCov(), Regularization::CenteringVariance(), Regularization::JacobianNorm(),
        Regularization::WGANGP(), Regularization::R1(), Regularization::R2(), Regularization::TRADES(),
        Regularization::VAT(), Regularization::L2(), Regularization::NuclearNorm(), Regularization::SWA(),
        Regularization::SWAG(), Regularization::FGE(), Regularization::SFGE(),
    };

    // Bump `.options.coefficient` to 1.0 where that field exists, so the real penalty math
    // executes instead of the coefficient==0 fast-path every descriptor defaults to.
    template <class Descriptor>
    Descriptor with_active_coefficient(Descriptor descriptor) {
        if constexpr (requires { descriptor.options.coefficient; }) {
            descriptor.options.coefficient = 1.0;
        }
        return descriptor;
    }
}

TEST_CASE("regularization: every simple-signature descriptor produces a finite, defined penalty") {
    auto params = torch::randn({6, 6}, torch::requires_grad(true));

    for (const auto& variant_descriptor : kSimpleRegularizers) {
        std::visit(
            [&](const auto& concrete_descriptor) {
                using DescriptorType = std::decay_t<decltype(concrete_descriptor)>;
                if constexpr (Regularization::detail::SupportsPenalty<DescriptorType>) {
                    auto active = with_active_coefficient(concrete_descriptor);
                    auto penalty = Regularization::apply(active, params);
                    REQUIRE(penalty.defined());
                    CHECK(torch::isfinite(penalty).all().template item<bool>());
                } else {
                    FAIL("Descriptor unexpectedly lacks the simple (descriptor, params) penalty overload; "
                         "update kSimpleRegularizers or this test.");
                }
            },
            variant_descriptor);
    }
}

TEST_CASE("regularization: accumulate sums the penalty across a parameter list") {
    std::vector<torch::Tensor> parameters = {torch::randn({4, 4}), torch::randn({3})};
    auto descriptor = with_active_coefficient(Regularization::L1());
    auto total = Regularization::accumulate(descriptor, parameters);
    REQUIRE(total.defined());
    CHECK(torch::isfinite(total).all().item<bool>());

    auto expected = Regularization::apply(descriptor, parameters[0]) + Regularization::apply(descriptor, parameters[1]);
    CHECK(torch::allclose(total, expected));
}

TEST_CASE("regularization: L1 penalty is zero at coefficient 0 and positive once activated") {
    auto params = torch::randn({5, 5});
    CHECK(Regularization::apply(Regularization::L1(), params).item<double>() == doctest::Approx(0.0));

    auto active = with_active_coefficient(Regularization::L1());
    CHECK(Regularization::apply(active, params).item<double>() > 0.0);
}
