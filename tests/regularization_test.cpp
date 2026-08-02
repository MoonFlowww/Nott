#include "test_prelude.hpp"

#include <torch/torch.h>
#include <variant>
#include "../src/regularization/apply.hpp"

using namespace Nott;

namespace {
    // Every simple (single-tensor) regularizer descriptor. EWC/MAS/SI are excluded: their
    // penalty() only accepts an extra task-importance state that only exists mid-training,
    // so they're exercised indirectly, not here.
    // No dedicated EWC/MAS/SI unit test; add one if a bug is ever reported there.
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

// VAT and TRADES each have two overloads: a single-tensor one that is a
// deliberate zero stub, and a two-tensor one carrying the real KL divergence.
// The suite above only ever reached the stub, so the actual maths was never
// executed. Both compute KL(softmax(logits) || softmax(other)) * coefficient,
// which has closed-form answers we can pin down.
namespace {
    template <class Descriptor>
    void check_kl_penalty_contract(Descriptor descriptor) {
        descriptor.options.coefficient = 1.0;
        auto logits = torch::tensor({{2.0, 0.5, -1.0}, {0.1, 0.2, 0.3}});

        // Identical distributions have zero divergence.
        torch::Tensor identical = Regularization::Details::penalty(descriptor, logits, logits.clone());
        REQUIRE(identical.defined());
        CHECK(identical.item<double>() == doctest::Approx(0.0).epsilon(1e-9));

        // Divergent ones are strictly positive, and the coefficient scales it.
        auto other = torch::tensor({{-1.0, 0.0, 2.0}, {1.5, -0.5, 0.0}});
        torch::Tensor divergence = Regularization::Details::penalty(descriptor, logits, other);
        REQUIRE(divergence.defined());
        CHECK(divergence.item<double>() > 0.0);
        CHECK(std::isfinite(divergence.item<double>()));

        // Against a hand-computed KL over the same inputs.
        auto p = torch::softmax(logits, -1);
        auto expected = (p * (torch::log_softmax(logits, -1) - torch::log_softmax(other, -1)))
                            .sum(-1).mean().item<double>();
        CHECK(divergence.item<double>() == doctest::Approx(expected));

        auto scaled = descriptor;
        scaled.options.coefficient = 2.5;
        torch::Tensor scaled_penalty = Regularization::Details::penalty(scaled, logits, other);
        CHECK(scaled_penalty.item<double>() == doctest::Approx(divergence.item<double>() * 2.5));

        // A zero coefficient must short-circuit to zero, not to the KL value.
        auto disabled = descriptor;
        disabled.options.coefficient = 0.0;
        torch::Tensor disabled_penalty = Regularization::Details::penalty(disabled, logits, other);
        REQUIRE(disabled_penalty.defined());
        CHECK(disabled_penalty.sum().item<double>() == doctest::Approx(0.0));
    }
}

TEST_CASE("regularization: VAT's two-tensor overload computes a real KL divergence") {
    check_kl_penalty_contract(Regularization::Details::VATDescriptor{});
}

TEST_CASE("regularization: TRADES's two-tensor overload computes a real KL divergence") {
    check_kl_penalty_contract(Regularization::Details::TRADESDescriptor{});
}

TEST_CASE("regularization: the single-tensor VAT/TRADES overloads stay a zero stub") {
    auto logits = torch::randn({4, 3});
    Regularization::Details::VATDescriptor vat{};
    vat.options.coefficient = 1.0;
    auto vat_penalty = Regularization::Details::penalty(vat, logits);
    REQUIRE(vat_penalty.defined());
    CHECK(vat_penalty.sum().item<double>() == doctest::Approx(0.0));

    Regularization::Details::TRADESDescriptor trades{};
    trades.options.coefficient = 1.0;
    auto trades_penalty = Regularization::Details::penalty(trades, logits);
    REQUIRE(trades_penalty.defined());
    CHECK(trades_penalty.sum().item<double>() == doctest::Approx(0.0));
}

// The state-carrying regularizers: EWC, MAS, SI (continual-learning importance)
// and SWA, SWAG, FGE, SFGE (weight-averaging snapshots). None had a test,
// because their penalty() needs state that only exists mid-training. That state
// is a plain struct, so it can be built by hand here.
//
// Shared contract, asserted for each: no drift from the reference means no
// penalty, drift means a strictly positive one, and the strength/coefficient
// scales the result linearly.
namespace {
    template <class Penalty>
    void check_drift_penalty_contract(Penalty penalty_of, double& knob)
    {
        auto reference = torch::zeros({4});

        knob = 1.0;
        torch::Tensor at_reference = penalty_of(reference);
        REQUIRE(at_reference.defined());
        CHECK(at_reference.sum().item<double>() == doctest::Approx(0.0));

        auto drifted = torch::full({4}, 0.5);
        torch::Tensor drift = penalty_of(drifted);
        REQUIRE(drift.defined());
        const double base = drift.sum().item<double>();
        CHECK(base > 0.0);
        CHECK(std::isfinite(base));

        knob = 3.0;
        torch::Tensor scaled = penalty_of(drifted);
        CHECK(scaled.sum().item<double>() == doctest::Approx(base * 3.0));

        knob = 0.0;
        torch::Tensor disabled = penalty_of(drifted);
        REQUIRE(disabled.defined());
        CHECK(disabled.sum().item<double>() == doctest::Approx(0.0));
    }
}

TEST_CASE("regularization: EWC penalises drift weighted by Fisher information") {
    Regularization::Details::EWCDescriptor descriptor{};
    Regularization::Details::EWCState state{torch::zeros({4}), torch::ones({4})};
    check_drift_penalty_contract(
        [&](const torch::Tensor& params) {
            return Regularization::Details::penalty(descriptor, params, state);
        },
        descriptor.options.strength);

    // Fisher information weights each coordinate: doubling it doubles the term.
    descriptor.options.strength = 1.0;
    auto params = torch::full({4}, 0.5);
    const double unit = Regularization::Details::penalty(descriptor, params, state).item<double>();
    Regularization::Details::EWCState heavier{torch::zeros({4}), torch::full({4}, 2.0)};
    const double doubled = Regularization::Details::penalty(descriptor, params, heavier).item<double>();
    CHECK(doubled == doctest::Approx(unit * 2.0));
    // sum(fisher * (params - reference)^2) * strength = 4 * 1 * 0.25 * 1
    CHECK(unit == doctest::Approx(1.0));
}

TEST_CASE("regularization: MAS penalises drift weighted by importance") {
    Regularization::Details::MASDescriptor descriptor{};
    Regularization::Details::MASState state{torch::zeros({4}), torch::ones({4})};
    check_drift_penalty_contract(
        [&](const torch::Tensor& params) {
            return Regularization::Details::penalty(descriptor, params, state);
        },
        descriptor.options.strength);
}

TEST_CASE("regularization: SI penalises drift weighted by importance") {
    Regularization::Details::SIDescriptor descriptor{};
    Regularization::Details::SIState state{torch::zeros({4}), torch::ones({4})};
    check_drift_penalty_contract(
        [&](const torch::Tensor& params) {
            return Regularization::Details::penalty(descriptor, params, state);
        },
        descriptor.options.strength);
}

TEST_CASE("regularization: SWA pulls parameters toward the running average") {
    Regularization::Details::SWADescriptor descriptor{};
    Regularization::Details::SWAState state{torch::zeros({4})};
    check_drift_penalty_contract(
        [&](const torch::Tensor& params) {
            return Regularization::Details::penalty(descriptor, params, state);
        },
        descriptor.options.coefficient);

    // mean((params - average)^2) * coefficient = 0.25 * 1
    descriptor.options.coefficient = 1.0;
    CHECK(Regularization::Details::penalty(descriptor, torch::full({4}, 0.5), state).item<double>() ==
          doctest::Approx(0.25));

    // An undefined average is the "no snapshots yet" case and must be inert.
    Regularization::Details::SWAState empty{};
    CHECK(Regularization::Details::penalty(descriptor, torch::full({4}, 0.5), empty).sum().item<double>() ==
          doctest::Approx(0.0));
}

TEST_CASE("regularization: FGE averages the distance to every snapshot") {
    Regularization::Details::FGEDescriptor descriptor{};
    Regularization::Details::FGEState state{};
    state.snapshots = {torch::zeros({4}), torch::zeros({4})};
    check_drift_penalty_contract(
        [&](const torch::Tensor& params) {
            return Regularization::Details::penalty(descriptor, params, state);
        },
        descriptor.options.coefficient);

    // Two identical snapshots must give the same answer as one: the sum over
    // snapshots is divided by their count.
    descriptor.options.coefficient = 1.0;
    auto params = torch::full({4}, 0.5);
    Regularization::Details::FGEState single{};
    single.snapshots = {torch::zeros({4})};
    CHECK(Regularization::Details::penalty(descriptor, params, state).item<double>() ==
          doctest::Approx(Regularization::Details::penalty(descriptor, params, single).item<double>()));

    Regularization::Details::FGEState empty{};
    CHECK(Regularization::Details::penalty(descriptor, params, empty).sum().item<double>() ==
          doctest::Approx(0.0));
}

TEST_CASE("regularization: SFGE weights each snapshot's contribution") {
    Regularization::Details::SFGEDescriptor descriptor{};
    Regularization::Details::SFGEState state{};
    state.snapshots = {torch::zeros({4}), torch::zeros({4})};
    state.weights = {1.0, 1.0};
    check_drift_penalty_contract(
        [&](const torch::Tensor& params) {
            return Regularization::Details::penalty(descriptor, params, state);
        },
        descriptor.options.coefficient);

    // Equal weights reduce to the unweighted average, so a lopsided weighting
    // toward an identical snapshot must land in the same place.
    descriptor.options.coefficient = 1.0;
    auto params = torch::full({4}, 0.5);
    Regularization::Details::SFGEState lopsided{};
    lopsided.snapshots = {torch::zeros({4}), torch::zeros({4})};
    lopsided.weights = {3.0, 1.0};
    CHECK(Regularization::Details::penalty(descriptor, params, state).item<double>() ==
          doctest::Approx(Regularization::Details::penalty(descriptor, params, lopsided).item<double>()));
}

TEST_CASE("regularization: SWAG needs at least two snapshots before it penalises") {
    Regularization::Details::SWAGDescriptor descriptor{};
    descriptor.options.coefficient = 1.0;
    auto params = torch::full({4}, 0.5);

    // Variance is undefined from a single snapshot, so the guard returns zero.
    Regularization::Details::SWAGState too_few{torch::zeros({4}), torch::ones({4}), 1};
    CHECK(Regularization::Details::penalty(descriptor, params, too_few).sum().item<double>() ==
          doctest::Approx(0.0));

    Regularization::Details::SWAGState usable{torch::zeros({4}), torch::ones({4}), 4};
    torch::Tensor penalty = Regularization::Details::penalty(descriptor, params, usable);
    REQUIRE(penalty.defined());
    CHECK(penalty.item<double>() > 0.0);
    CHECK(std::isfinite(penalty.item<double>()));
}
