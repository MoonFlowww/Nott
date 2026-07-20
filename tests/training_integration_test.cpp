#include "third_party/doctest.h"

#include <torch/torch.h>
#include <array>
#include "../include/Nott.h"

using namespace Nott;

namespace {
    constexpr std::array kAllClassificationKinds = {
        // Every Metric::Classification::Kind except HausdorffDistance/BoundaryIoU, which are
        // segmentation-only and expect mask-shaped tensors rather than class scores.
        Metric::Classification::Kind::Accuracy, Metric::Classification::Kind::AUCROC,
        Metric::Classification::Kind::BalancedAccuracy, Metric::Classification::Kind::BalancedErrorRate,
        Metric::Classification::Kind::F1, Metric::Classification::Kind::FBeta0Point5,
        Metric::Classification::Kind::FBeta2, Metric::Classification::Kind::FalseDiscoveryRate,
        Metric::Classification::Kind::FalseNegativeRate, Metric::Classification::Kind::FalseOmissionRate,
        Metric::Classification::Kind::FalsePositiveRate, Metric::Classification::Kind::FowlkesMallows,
        Metric::Classification::Kind::HammingLoss, Metric::Classification::Kind::Informedness,
        Metric::Classification::Kind::JaccardIndexMicro, Metric::Classification::Kind::JaccardIndexMacro,
        Metric::Classification::Kind::Markness, Metric::Classification::Kind::Matthews,
        Metric::Classification::Kind::NegativeLikelihoodRatio, Metric::Classification::Kind::NegativePredictiveValue,
        Metric::Classification::Kind::PositiveLikelihoodRatio, Metric::Classification::Kind::PositivePredictiveValue,
        Metric::Classification::Kind::Precision, Metric::Classification::Kind::Prevalence,
        Metric::Classification::Kind::Recall, Metric::Classification::Kind::Top1Error,
        Metric::Classification::Kind::Top3Error, Metric::Classification::Kind::Top5Error,
        Metric::Classification::Kind::Top1Accuracy, Metric::Classification::Kind::Top3Accuracy,
        Metric::Classification::Kind::Top5Accuracy, Metric::Classification::Kind::Specificity,
        Metric::Classification::Kind::ThreatScore, Metric::Classification::Kind::TrueNegativeRate,
        Metric::Classification::Kind::TruePositiveRate, Metric::Classification::Kind::YoudenIndex,
        Metric::Classification::Kind::LogLoss, Metric::Classification::Kind::BrierScore,
        Metric::Classification::Kind::BrierSkillScore, Metric::Classification::Kind::ExpectedCalibrationError,
        Metric::Classification::Kind::MaximumCalibrationError, Metric::Classification::Kind::CalibrationSlope,
        Metric::Classification::Kind::CalibrationIntercept, Metric::Classification::Kind::HosmerLemeshowPValue,
        Metric::Classification::Kind::KolmogorovSmirnovStatistic, Metric::Classification::Kind::CohensKappa,
        Metric::Classification::Kind::ConfusionEntropy, Metric::Classification::Kind::CoverageError,
        Metric::Classification::Kind::LabelRankingAveragePrecision, Metric::Classification::Kind::SubsetAccuracy,
        Metric::Classification::Kind::AUPRC, Metric::Classification::Kind::AUPRG,
        Metric::Classification::Kind::GiniCoefficient,
    };

    constexpr std::array kAllTimeseriesKinds = {
        Metric::Timeseries::Kind::MeanAbsoluteError, Metric::Timeseries::Kind::MeanAbsolutePercentageError,
        Metric::Timeseries::Kind::MeanBiasError, Metric::Timeseries::Kind::MeanSquaredError,
        Metric::Timeseries::Kind::MedianAbsoluteError, Metric::Timeseries::Kind::R2Score,
        Metric::Timeseries::Kind::RootMeanSquaredError, Metric::Timeseries::Kind::SymmetricMeanAbsolutePercentageError,
        Metric::Timeseries::Kind::WeightedAbsolutePercentageError, Metric::Timeseries::Kind::MeanPercentageError,
        Metric::Timeseries::Kind::ExplainedVariance, Metric::Timeseries::Kind::TheilsU1,
        Metric::Timeseries::Kind::TheilsU2, Metric::Timeseries::Kind::MeanAbsoluteScaledError,
        Metric::Timeseries::Kind::RootMeanSquaredScaledError, Metric::Timeseries::Kind::MedianRelativeAbsoluteError,
        Metric::Timeseries::Kind::GeometricMeanRelativeAbsoluteError, Metric::Timeseries::Kind::OverallWeightedAverage,
        Metric::Timeseries::Kind::DynamicTimeWarpingDistance, Metric::Timeseries::Kind::TimeWarpEditDistance,
        Metric::Timeseries::Kind::SpectralDistance, Metric::Timeseries::Kind::CosineSimilarity,
        Metric::Timeseries::Kind::NegativeLogLikelihood, Metric::Timeseries::Kind::ContinuousRankedProbabilityScore,
        Metric::Timeseries::Kind::EnergyScore, Metric::Timeseries::Kind::PinballLossAverage,
        Metric::Timeseries::Kind::BrierScore, Metric::Timeseries::Kind::PredictionIntervalCoverageProbability,
        Metric::Timeseries::Kind::MeanPredictionIntervalWidth, Metric::Timeseries::Kind::WinklerScore,
        Metric::Timeseries::Kind::ConditionalCoverageError, Metric::Timeseries::Kind::QuantileCrossingRate,
        Metric::Timeseries::Kind::AutocorrelationOfResiduals, Metric::Timeseries::Kind::PartialAutocorrelationOfResiduals,
        Metric::Timeseries::Kind::LjungBoxStatistic, Metric::Timeseries::Kind::BoxPierceStatistic,
        Metric::Timeseries::Kind::DurbinWatsonStatistic, Metric::Timeseries::Kind::JarqueBeraStatistic,
        Metric::Timeseries::Kind::AndersonDarlingStatistic, Metric::Timeseries::Kind::BreuschPaganStatistic,
        Metric::Timeseries::Kind::WhiteStatistic, Metric::Timeseries::Kind::PopulationStabilityIndex,
        Metric::Timeseries::Kind::KullbackLeiblerDivergence, Metric::Timeseries::Kind::JensenShannonDivergence,
        Metric::Timeseries::Kind::WassersteinDistance, Metric::Timeseries::Kind::MaximumMeanDiscrepancy,
        Metric::Timeseries::Kind::LossDriftSlope, Metric::Timeseries::Kind::LossCusumStatistic,
        Metric::Timeseries::Kind::ResidualChangePointScore, Metric::Timeseries::Kind::QLIKE,
        Metric::Timeseries::Kind::LogVarianceMeanSquaredError, Metric::Timeseries::Kind::SqrtVarianceMeanSquaredError,
        Metric::Timeseries::Kind::msIC, Metric::Timeseries::Kind::msIR,
    };
}

namespace {
    // Every class index in [0, kNumClasses) guaranteed to appear at least once, so class
    // count inference (see classification.hpp) has the full picture.
    torch::Tensor make_targets_covering_every_class(int64_t num_classes, int64_t total) {
        auto targets = torch::randint(0, num_classes, {total}, torch::kInt64);
        for (int64_t class_index = 0; class_index < num_classes && class_index < total; ++class_index) {
            targets[class_index] = class_index;
        }
        return targets;
    }
}

TEST_CASE("training: end-to-end classifier trains and evaluates against every classification metric") {
    // Top3/Top5 metrics need at least that many classes to exist, or the underlying
    // topk() call throws -- use 6 classes so the full metric list below is meaningful.
    constexpr int64_t kNumClasses = 6;
    Model model("integration_classifier");
    model.add(Layer::FC({8, 16, true}, Activation::ReLU, Initialization::HeNormal));
    model.add(Layer::FC({16, kNumClasses, true}, Activation::Softmax));
    model.set_optimizer(Optimizer::AdamW({.learning_rate = 1e-2}));
    model.set_loss(Loss::CrossEntropy());

    auto inputs = torch::randn({64, 8});
    auto targets = make_targets_covering_every_class(kNumClasses, 64);

    TrainOptions train_options{};
    train_options.epoch = 5;
    train_options.batch_size = 16;
    train_options.monitor = false;
    model.train(inputs, targets, train_options);

    for (const auto& epoch : model.training_telemetry().epochs()) {
        CHECK(std::isfinite(epoch.train_loss_value()));
    }

    std::vector<Metric::Classification::Descriptor> metrics;
    metrics.reserve(kAllClassificationKinds.size());
    for (auto kind : kAllClassificationKinds) {
        metrics.push_back(Metric::Classification::Make(kind));
    }

    Evaluation::Options eval_options{};
    eval_options.print_summary = false;
    eval_options.print_per_class = false;
    // Deliberately smaller than total_samples: class-count inference is computed once
    // from the full target set (see the classification-batch-inference regression test
    // below), so per-batch class coverage no longer matters here.
    eval_options.batch_size = 8;

    Evaluation::ClassificationReport report;
    REQUIRE_NOTHROW(report = model.evaluate(inputs, targets, Evaluation::Classification, metrics, eval_options));

    CHECK(report.total_samples == 64);
    CHECK(report.summary.size() == kAllClassificationKinds.size());
}

TEST_CASE("training: classification evaluation is stable when one mini-batch misses the top class") {
    // Regression test for a fixed bug: classification.hpp used to infer the class count
    // per *mini-batch* from `max(observed label in that batch) + 1` instead of from the
    // full target set / model output width. Whenever a batch didn't happen to contain a
    // sample of the highest class index, logits got silently sliced down to fewer
    // columns before any metric was computed for that batch -- corrupting every metric
    // on that batch, and outright crashing Top-k metrics that needed more columns than
    // survived the slice ("selected index k out of range"). Fixed by inferring the class
    // count once from the full target tensor, before batching.
    //
    // Here the model has 6 output classes, but the *first* 8-sample batch deliberately
    // only contains classes 0-3 (missing 4 and 5) while the full 64-sample target set
    // does cover all 6. Before the fix this reliably crashed on Top5Accuracy for the
    // first batch even though the dataset as a whole has enough classes.
    constexpr int64_t kNumClasses = 6;
    Model model("classification_batch_inference_regression");
    model.add(Layer::FC({4, kNumClasses, true}, Activation::Softmax));
    model.set_optimizer(Optimizer::SGD());
    model.set_loss(Loss::CrossEntropy());

    auto inputs = torch::randn({64, 4});
    // Every sample defaults to class 0-3 (the first 8-sample batch stays entirely within
    // this range), then samples 8 and 9 -- in the *second* batch -- are forced to classes
    // 4 and 5 so the full dataset covers all 6 classes.
    auto targets = torch::randint(0, kNumClasses - 2, {64}, torch::kInt64);
    targets.index_put_({8}, kNumClasses - 2);
    targets.index_put_({9}, kNumClasses - 1);

    std::vector<Metric::Classification::Descriptor> metrics = {Metric::Classification::Top5Accuracy};
    Evaluation::Options eval_options{};
    eval_options.print_summary = false;
    eval_options.print_per_class = false;
    eval_options.batch_size = 8;

    REQUIRE_NOTHROW(model.evaluate(inputs, targets, Evaluation::Classification, metrics, eval_options));
}

TEST_CASE("training: end-to-end regressor trains and evaluates against every timeseries metric") {
    Model model("integration_regressor");
    model.add(Layer::FC({4, 8, true}, Activation::ReLU, Initialization::HeNormal));
    model.add(Layer::FC({8, 1, true}));
    model.set_optimizer(Optimizer::Adam({.learning_rate = 1e-2}));
    model.set_loss(Loss::MSE());

    auto inputs = torch::randn({64, 4});
    auto targets = torch::randn({64, 1});

    TrainOptions train_options{};
    train_options.epoch = 5;
    train_options.batch_size = 16;
    train_options.monitor = false;
    model.train(inputs, targets, train_options);

    std::vector<Metric::Timeseries::Descriptor> metrics;
    metrics.reserve(kAllTimeseriesKinds.size());
    for (auto kind : kAllTimeseriesKinds) {
        metrics.push_back(Metric::Timeseries::Make(kind));
    }

    Evaluation::TimeseriesOptions eval_options{};
    eval_options.print_summary = false;

    Evaluation::TimeseriesReport report;
    REQUIRE_NOTHROW(report = Evaluation::Evaluate(model, inputs, targets, Evaluation::Timeseries, metrics, eval_options));

    CHECK(report.total_series == 64);
    CHECK(report.order.size() == kAllTimeseriesKinds.size());
}
