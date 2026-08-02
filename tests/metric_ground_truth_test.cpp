#include "test_prelude.hpp"

#include <torch/torch.h>
#include <algorithm>
#include "../include/Nott.h"

using namespace Nott;

// These tests check metric *values* against hand-computed ground truth, unlike
// training_integration_test.cpp's smoke loop (which only checks "didn't crash, right
// count"). A wrong formula (swapped precision/recall, off-by-one in an integral, wrong
// sign) would sail through a crash-only check but should get caught here.

TEST_CASE("metric: classification metrics match a hand-computed confusion matrix") {
    // Binary, N=10, deliberately imperfect and symmetric: TP=FN=FP=TN=... the same for
    // *both* classes by construction, so macro-averaging collapses to the same numbers
    // as the textbook binary formulas (sidesteps ambiguity about macro vs micro vs
    // per-class-accuracy conventions -- ANY reasonable aggregation gives the same
    // answer here).
    //   actual:    0 0 0 0 0 1 1 1 1 1
    //   predicted: 0 0 0 1 1 1 1 1 0 0
    // confusion (per class, both classes symmetric): TP=3 FN=2 FP=2 TN=3
    //   Accuracy         = (TP+TN)/N            = 6/10  = 0.6
    //   Precision/Recall/Specificity/F1/BalancedAccuracy = TP/(TP+FP) etc. = 3/5 = 0.6
    //   Matthews (MCC)   = (TP*TN-FP*FN)/sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN)) = 5/25 = 0.2
    //   YoudenIndex      = Recall+Specificity-1  = 0.2
    //   Markness         = Precision+NPV-1       = 0.2
    //   CohensKappa      = (po-pe)/(1-pe), po=0.6, pe=0.5                  = 0.2
    //   FalsePositiveRate/FalseNegativeRate/FalseDiscoveryRate/FalseOmissionRate = 2/5 = 0.4
    //   HammingLoss      = 1-Accuracy            = 0.4
    Model model("metric_ground_truth_classification");
    model.add(Layer::FC({2, 2, false}));

    {
        torch::NoGradGuard guard;
        auto parameters = model.parameters();
        REQUIRE(parameters.size() == 1);
        parameters[0].copy_(torch::eye(2)); // forward(x) == x: logits are whatever we pass in
    }

    // Large separation keeps softmax-derived probabilities close to hard 0/1 without
    // affecting the argmax-based metrics (which are exact regardless of separation).
    auto logits = torch::tensor({
        {10.0, -10.0}, {10.0, -10.0}, {10.0, -10.0}, {-10.0, 10.0}, {-10.0, 10.0},
        {-10.0, 10.0}, {-10.0, 10.0}, {-10.0, 10.0}, {10.0, -10.0}, {10.0, -10.0},
    });
    auto targets = torch::tensor({0, 0, 0, 0, 0, 1, 1, 1, 1, 1}, torch::kInt64);

    std::vector<Metric::Classification::Descriptor> metrics = {
        Metric::Classification::Accuracy,       Metric::Classification::Precision,
        Metric::Classification::Recall,         Metric::Classification::Specificity,
        Metric::Classification::F1,             Metric::Classification::BalancedAccuracy,
        Metric::Classification::Matthews,       Metric::Classification::YoudenIndex,
        Metric::Classification::Markness,       Metric::Classification::CohensKappa,
        Metric::Classification::FalsePositiveRate, Metric::Classification::FalseNegativeRate,
        Metric::Classification::FalseDiscoveryRate, Metric::Classification::FalseOmissionRate,
        Metric::Classification::HammingLoss,
    };

    Evaluation::Options eval_options{};
    eval_options.print_summary = false;
    eval_options.print_per_class = false;

    auto report = model.evaluate(logits, targets, Evaluation::Classification, metrics, eval_options);

    auto macro_value = [&](Metric::Classification::Kind kind) -> double {
        auto it = std::find_if(report.summary.begin(), report.summary.end(),
                               [&](const auto& row) { return row.metric == kind; });
        REQUIRE(it != report.summary.end());
        return it->macro;
    };

    using Kind = Metric::Classification::Kind;
    CHECK(macro_value(Kind::Accuracy) == doctest::Approx(0.6));
    CHECK(macro_value(Kind::Precision) == doctest::Approx(0.6));
    CHECK(macro_value(Kind::Recall) == doctest::Approx(0.6));
    CHECK(macro_value(Kind::Specificity) == doctest::Approx(0.6));
    CHECK(macro_value(Kind::F1) == doctest::Approx(0.6));
    CHECK(macro_value(Kind::BalancedAccuracy) == doctest::Approx(0.6));
    CHECK(macro_value(Kind::Matthews) == doctest::Approx(0.2));
    CHECK(macro_value(Kind::YoudenIndex) == doctest::Approx(0.2));
    CHECK(macro_value(Kind::Markness) == doctest::Approx(0.2));
    CHECK(macro_value(Kind::CohensKappa) == doctest::Approx(0.2));
    CHECK(macro_value(Kind::FalsePositiveRate) == doctest::Approx(0.4));
    CHECK(macro_value(Kind::FalseNegativeRate) == doctest::Approx(0.4));
    CHECK(macro_value(Kind::FalseDiscoveryRate) == doctest::Approx(0.4));
    CHECK(macro_value(Kind::FalseOmissionRate) == doctest::Approx(0.4));
    CHECK(macro_value(Kind::HammingLoss) == doctest::Approx(0.4));
}

TEST_CASE("metric: classification metrics are exact at a perfect confusion matrix") {
    // Complements the imperfect case above: catches formulas that only happen to be
    // right when errors exist (e.g. an off-by-one in a denominator that only manifests
    // away from the boundary) by pinning down the degenerate all-correct case too.
    Model model("metric_ground_truth_perfect");
    model.add(Layer::FC({3, 3, false}));

    {
        torch::NoGradGuard guard;
        auto parameters = model.parameters();
        REQUIRE(parameters.size() == 1);
        parameters[0].copy_(torch::eye(3));
    }

    auto logits = torch::tensor({
        {10.0, -10.0, -10.0}, {10.0, -10.0, -10.0}, {-10.0, 10.0, -10.0},
        {-10.0, 10.0, -10.0}, {-10.0, -10.0, 10.0}, {-10.0, -10.0, 10.0},
    });
    auto targets = torch::tensor({0, 0, 1, 1, 2, 2}, torch::kInt64);

    std::vector<Metric::Classification::Descriptor> metrics = {
        Metric::Classification::Accuracy, Metric::Classification::Precision, Metric::Classification::Recall,
        Metric::Classification::F1,       Metric::Classification::Matthews,  Metric::Classification::CohensKappa,
    };
    Evaluation::Options eval_options{};
    eval_options.print_summary = false;
    eval_options.print_per_class = false;

    auto report = model.evaluate(logits, targets, Evaluation::Classification, metrics, eval_options);

    auto macro_value = [&](Metric::Classification::Kind kind) -> double {
        auto it = std::find_if(report.summary.begin(), report.summary.end(),
                               [&](const auto& row) { return row.metric == kind; });
        REQUIRE(it != report.summary.end());
        return it->macro;
    };

    using Kind = Metric::Classification::Kind;
    CHECK(macro_value(Kind::Accuracy) == doctest::Approx(1.0));
    CHECK(macro_value(Kind::Precision) == doctest::Approx(1.0));
    CHECK(macro_value(Kind::Recall) == doctest::Approx(1.0));
    CHECK(macro_value(Kind::F1) == doctest::Approx(1.0));
    CHECK(macro_value(Kind::Matthews) == doctest::Approx(1.0));
    CHECK(macro_value(Kind::CohensKappa) == doctest::Approx(1.0));
}

TEST_CASE("metric: timeseries metrics match hand-computed values on a known series") {
    // predictions = targets + errors [1, 0, 1, 0, 2]:
    //   MAE  = mean(|e|)                       = 4/5   = 0.8
    //   MSE  = mean(e^2)                       = 6/5   = 1.2
    //   RMSE = sqrt(MSE)
    //   R2   = 1 - SS_res/SS_tot, SS_res=6, SS_tot=sum((y-mean(y))^2)=10 -> 1-6/10 = 0.4
    Model model("metric_ground_truth_timeseries");
    model.add(Layer::FC({1, 1, false}));

    {
        torch::NoGradGuard guard;
        auto parameters = model.parameters();
        REQUIRE(parameters.size() == 1);
        parameters[0].copy_(torch::eye(1)); // forward(x) == x
    }

    auto predictions_source = torch::tensor({{2.0}, {2.0}, {4.0}, {4.0}, {7.0}});
    auto targets = torch::tensor({{1.0}, {2.0}, {3.0}, {4.0}, {5.0}});

    std::vector<Metric::Timeseries::Descriptor> metrics = {
        Metric::Timeseries::MeanAbsoluteError,
        Metric::Timeseries::MeanSquaredError,
        Metric::Timeseries::RootMeanSquaredError,
        Metric::Timeseries::R2Score,
    };
    Evaluation::TimeseriesOptions eval_options{};
    eval_options.print_summary = false;

    auto report =
        Evaluation::Evaluate(model, predictions_source, targets, Evaluation::Timeseries, metrics, eval_options);

    auto value_of = [&](Metric::Timeseries::Kind kind) -> double {
        auto it = std::find(report.order.begin(), report.order.end(), kind);
        REQUIRE(it != report.order.end());
        return report.values[static_cast<std::size_t>(std::distance(report.order.begin(), it))];
    };

    using Kind = Metric::Timeseries::Kind;
    CHECK(value_of(Kind::MeanAbsoluteError) == doctest::Approx(0.8));
    CHECK(value_of(Kind::MeanSquaredError) == doctest::Approx(1.2));
    CHECK(value_of(Kind::RootMeanSquaredError) == doctest::Approx(std::sqrt(1.2)));
    CHECK(value_of(Kind::R2Score) == doctest::Approx(0.4));
}
