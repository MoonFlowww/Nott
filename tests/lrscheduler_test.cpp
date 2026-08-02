/// Cosine annealing schedule. The maths is closed form, so these assert the
/// actual curve rather than just that the learning rate moved.
#include "third_party/doctest.h"

#include <cmath>
#include <torch/torch.h>

#include "../src/lrscheduler/lrscheduler.hpp"

namespace {
    constexpr double kBaseLr = 0.1;

    /// A scheduler needs a live optimizer to write into; one parameter is enough.
    struct Fixture {
        torch::Tensor parameter{torch::zeros({1}, torch::requires_grad())};
        torch::optim::SGD optimizer{{parameter}, torch::optim::SGDOptions(kBaseLr)};

        [[nodiscard]] double lr() const
        {
            return optimizer.param_groups().front().options().get_lr();
        }
    };

    double current_lr(const torch::optim::Optimizer& optimizer)
    {
        return const_cast<torch::optim::Optimizer&>(optimizer).param_groups().front().options().get_lr();
    }
}

TEST_CASE("cosine annealing starts at the base learning rate") {
    Fixture fixture;
    Nott::LrScheduler::Details::CosineAnnealingOptions options;
    options.T_max = 10;
    Nott::LrScheduler::Details::CosineAnnealingScheduler scheduler(fixture.optimizer, options);
    CHECK(current_lr(fixture.optimizer) == doctest::Approx(kBaseLr));
}

TEST_CASE("cosine annealing reaches eta_min at T_max and follows the cosine curve") {
    Fixture fixture;
    Nott::LrScheduler::Details::CosineAnnealingOptions options;
    options.T_max = 10;
    options.eta_min = 0.001;
    Nott::LrScheduler::Details::CosineAnnealingScheduler scheduler(fixture.optimizer, options);

    /// Halfway through the cycle the cosine term is zero, so the rate sits at
    /// the midpoint between base and eta_min.
    for (int step = 0; step < 5; ++step) {
        scheduler.step();
    }
    CHECK(current_lr(fixture.optimizer) == doctest::Approx((kBaseLr + options.eta_min) / 2.0));

    for (int step = 5; step < 10; ++step) {
        scheduler.step();
    }
    CHECK(current_lr(fixture.optimizer) == doctest::Approx(options.eta_min));
}

TEST_CASE("cosine annealing decreases monotonically across the cycle") {
    Fixture fixture;
    Nott::LrScheduler::Details::CosineAnnealingOptions options;
    options.T_max = 20;
    Nott::LrScheduler::Details::CosineAnnealingScheduler scheduler(fixture.optimizer, options);

    double previous = current_lr(fixture.optimizer);
    for (int step = 0; step < 20; ++step) {
        scheduler.step();
        const double now = current_lr(fixture.optimizer);
        CHECK(now < previous);
        previous = now;
    }
}

TEST_CASE("stepping past T_max clamps instead of climbing back up") {
    Fixture fixture;
    Nott::LrScheduler::Details::CosineAnnealingOptions options;
    options.T_max = 5;
    options.eta_min = 0.002;
    Nott::LrScheduler::Details::CosineAnnealingScheduler scheduler(fixture.optimizer, options);

    for (int step = 0; step < 5; ++step) {
        scheduler.step();
    }
    const double at_t_max = current_lr(fixture.optimizer);

    /// Without the clamp the cosine would turn back upward past a half period.
    for (int step = 0; step < 25; ++step) {
        scheduler.step();
        CHECK(current_lr(fixture.optimizer) == doctest::Approx(at_t_max));
    }
    CHECK(at_t_max == doctest::Approx(options.eta_min));
}

TEST_CASE("warmup ramps from the start factor then hands over to the cosine") {
    Fixture fixture;
    Nott::LrScheduler::Details::CosineAnnealingOptions options;
    options.T_max = 10;
    options.warmup_steps = 4;
    options.warmup_start_factor = 0.25;
    Nott::LrScheduler::Details::CosineAnnealingScheduler scheduler(fixture.optimizer, options);

    CHECK(current_lr(fixture.optimizer) == doctest::Approx(kBaseLr * 0.25));

    /// Linear ramp: factor goes 0.25 -> 1.0 across the warmup window.
    double previous = current_lr(fixture.optimizer);
    for (std::size_t step = 1; step < options.warmup_steps; ++step) {
        scheduler.step();
        const double now = current_lr(fixture.optimizer);
        CHECK(now > previous);
        previous = now;
    }

    /// The step that ends warmup lands on the cosine at its own step zero,
    /// which is the full base rate.
    scheduler.step();
    CHECK(current_lr(fixture.optimizer) == doctest::Approx(kBaseLr));
}

TEST_CASE("cosine annealing rejects options it cannot honour") {
    Fixture fixture;

    SUBCASE("T_max of zero") {
        Nott::LrScheduler::Details::CosineAnnealingOptions options;
        options.T_max = 0;
        CHECK_THROWS_AS(
            Nott::LrScheduler::Details::CosineAnnealingScheduler(fixture.optimizer, options),
            std::invalid_argument);
    }

    SUBCASE("warmup start factor outside [0, 1]") {
        Nott::LrScheduler::Details::CosineAnnealingOptions options;
        options.T_max = 5;
        options.warmup_start_factor = 1.5;
        CHECK_THROWS_AS(
            Nott::LrScheduler::Details::CosineAnnealingScheduler(fixture.optimizer, options),
            std::invalid_argument);
    }
}

TEST_CASE("each param group anneals from its own base rate") {
    torch::Tensor first = torch::zeros({1}, torch::requires_grad());
    torch::Tensor second = torch::zeros({1}, torch::requires_grad());

    torch::optim::SGD optimizer({torch::optim::OptimizerParamGroup({first})},
                                torch::optim::SGDOptions(0.1));
    optimizer.add_param_group(torch::optim::OptimizerParamGroup({second}));
    optimizer.param_groups()[1].options().set_lr(0.5);

    Nott::LrScheduler::Details::CosineAnnealingOptions options;
    options.T_max = 4;
    Nott::LrScheduler::Details::CosineAnnealingScheduler scheduler(optimizer, options);

    CHECK(optimizer.param_groups()[0].options().get_lr() == doctest::Approx(0.1));
    CHECK(optimizer.param_groups()[1].options().get_lr() == doctest::Approx(0.5));

    for (int step = 0; step < 4; ++step) {
        scheduler.step();
    }
    /// Both land on eta_min (zero here), each having travelled its own range.
    CHECK(optimizer.param_groups()[0].options().get_lr() == doctest::Approx(0.0));
    CHECK(optimizer.param_groups()[1].options().get_lr() == doctest::Approx(0.0));
}

TEST_CASE("the public factory carries options through to the descriptor") {
    Nott::LrScheduler::CosineAnnealingOptions options;
    options.T_max = 7;
    options.eta_min = 0.05;
    const auto descriptor = Nott::LrScheduler::CosineAnnealing(options);
    CHECK(descriptor.options.T_max == 7);
    CHECK(descriptor.options.eta_min == doctest::Approx(0.05));

    const Nott::LrScheduler::Descriptor variant{descriptor};
    CHECK(std::holds_alternative<Nott::LrScheduler::CosineAnnealingDescriptor>(variant));
}
