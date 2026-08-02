/// Calibration: the reliability/AUC helpers have closed-form answers on small
/// inputs, and temperature scaling has a direction we can assert (an
/// overconfident model should get its logits divided down).
#include "third_party/doctest.h"

#include <limits>
#include <sstream>
#include <vector>

#include <torch/torch.h>

#include "../src/calibration/calibration.hpp"

namespace Calibration = Nott::Calibration;

TEST_CASE("auc is 1 for a perfect ranking and 0.5 for an uninformative one") {
    SUBCASE("perfect separation") {
        const std::vector<double> scores{0.1, 0.2, 0.8, 0.9};
        const std::vector<int> labels{0, 0, 1, 1};
        CHECK(Calibration::Details::compute_auc(scores, labels) == doctest::Approx(1.0));
    }

    SUBCASE("inverted ranking") {
        const std::vector<double> scores{0.9, 0.8, 0.2, 0.1};
        const std::vector<int> labels{0, 0, 1, 1};
        CHECK(Calibration::Details::compute_auc(scores, labels) == doctest::Approx(0.0));
    }

    SUBCASE("all scores tied") {
        const std::vector<double> scores{0.5, 0.5, 0.5, 0.5};
        const std::vector<int> labels{0, 1, 0, 1};
        CHECK(Calibration::Details::compute_auc(scores, labels) == doctest::Approx(0.5));
    }
}

TEST_CASE("auc reports NaN when it is undefined rather than a misleading number") {
    SUBCASE("empty input") {
        CHECK(std::isnan(Calibration::Details::compute_auc({}, {})));
    }

    SUBCASE("mismatched lengths") {
        const std::vector<double> scores{0.1, 0.2};
        const std::vector<int> labels{1};
        CHECK(std::isnan(Calibration::Details::compute_auc(scores, labels)));
    }

    SUBCASE("only one class present") {
        const std::vector<double> scores{0.1, 0.7, 0.9};
        const std::vector<int> labels{0, 0, 0};
        CHECK(std::isnan(Calibration::Details::compute_auc(scores, labels)));
    }
}

TEST_CASE("confidence maps into a bin index that stays in range") {
    constexpr std::size_t bins = 10;
    CHECK(Calibration::Details::clamp_bin_index(0.0, bins) == 0);
    CHECK(Calibration::Details::clamp_bin_index(0.55, bins) == 5);
    /// A confidence of exactly 1 must not fall off the end of the histogram.
    CHECK(Calibration::Details::clamp_bin_index(1.0, bins) == bins - 1);
    CHECK(Calibration::Details::clamp_bin_index(2.0, bins) == bins - 1);
}

TEST_CASE("a confidently correct model has near zero calibration error") {
    /// Four samples, all predicted correctly with high confidence.
    const auto logits = torch::tensor({{6.0, 0.0}, {6.0, 0.0}, {0.0, 6.0}, {0.0, 6.0}});
    const auto targets = torch::tensor({0, 0, 1, 1});

    const auto computation = Calibration::Details::compute_reliability(logits, targets, 15);

    std::size_t counted = 0;
    double confidence_sum = 0.0;
    double accuracy_sum = 0.0;
    for (const auto& bin : computation.bins) {
        counted += bin.count;
        confidence_sum += bin.confidence_sum;
        accuracy_sum += bin.accuracy_sum;
    }
    REQUIRE(counted == 4);
    /// Accuracy is 1 everywhere, and confidence is high, so the gap is small.
    CHECK(accuracy_sum == doctest::Approx(4.0));
    CHECK(std::abs(confidence_sum - accuracy_sum) < 0.05);
    CHECK(computation.log_loss < 0.05);
}

TEST_CASE("temperature scaling refuses to fit before it is attached") {
    auto method = Calibration::Details::make_temperature_scaling_method(
        Calibration::TemperatureScalingDescriptor{});
    const auto logits = torch::tensor({{2.0, 0.0}});
    const auto targets = torch::tensor({0});
    CHECK_THROWS_AS(method->fit(logits, targets), std::logic_error);
}

TEST_CASE("temperature scaling starts as a no-op at T=1") {
    auto method = Calibration::Details::make_temperature_scaling_method(
        Calibration::TemperatureScalingDescriptor{});
    torch::nn::Linear host(2, 2);
    torch::nn::Module& host_module = *host;
    method->attach(host_module, torch::kCPU);

    const auto logits = torch::tensor({{2.0, -1.0}, {0.5, 3.0}});
    const auto transformed = method->transform(logits);
    /// log_temperature is initialised to zero, so temperature is exp(0) == 1.
    CHECK(torch::allclose(transformed, logits));
}

TEST_CASE("fitting an overconfident model softens its logits") {
    torch::manual_seed(0);
    auto method = Calibration::Details::make_temperature_scaling_method(
        Calibration::TemperatureScalingDescriptor{});
    torch::nn::Linear host(2, 2);
    torch::nn::Module& host_module = *host;
    method->attach(host_module, torch::kCPU);

    /// Predictions are hugely confident but only 50 percent correct, which is
    /// the textbook case temperature scaling exists to fix: T should rise above
    /// 1 so the calibrated probabilities come down.
    const auto logits = torch::tensor({{12.0, 0.0}, {12.0, 0.0}, {12.0, 0.0}, {12.0, 0.0}});
    const auto targets = torch::tensor({0, 1, 0, 1});
    method->fit(logits, targets);

    const auto calibrated = method->transform(logits);
    const auto raw_confidence =
        std::get<0>(torch::softmax(logits, 1).max(1)).mean().item<double>();
    const auto calibrated_confidence =
        std::get<0>(torch::softmax(calibrated, 1).max(1)).mean().item<double>();
    CHECK(calibrated_confidence < raw_confidence);

    std::ostringstream description;
    method->plot(description);
    CHECK(description.str().find("Temperature scaling") != std::string::npos);
}

TEST_CASE("the calibration descriptor variant carries its options") {
    Calibration::TemperatureScalingDescriptor descriptor;
    descriptor.max_iterations = 25;
    descriptor.learning_rate = 0.05;

    const Calibration::Descriptor variant{descriptor};
    REQUIRE(std::holds_alternative<Calibration::TemperatureScalingDescriptor>(variant));
    CHECK(std::get<Calibration::TemperatureScalingDescriptor>(variant).max_iterations == 25);
    CHECK(std::get<Calibration::TemperatureScalingDescriptor>(variant).learning_rate == doctest::Approx(0.05));
}
