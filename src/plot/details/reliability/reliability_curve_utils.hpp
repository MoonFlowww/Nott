#ifndef Nott_RELIABILITY_CURVE_UTILS_HPP
#define Nott_RELIABILITY_CURVE_UTILS_HPP
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <torch/torch.h>

#include "../../../core/core.hpp"

namespace Nott::Plot::Details::Reliability::Curves {
    struct BinarySeries {
        std::vector<double> scores{};
        std::vector<int64_t> targets{};
        std::string label{};
    };

    struct BinaryCurvePoint {
        double threshold{0.0};
        std::size_t truePositives{0};
        std::size_t falsePositives{0};
        std::size_t falseNegatives{0};
        std::size_t trueNegatives{0};
    };

    struct BinaryCurve {
        std::vector<BinaryCurvePoint> points{};
        std::size_t totalPositives{0};
        std::size_t totalNegatives{0};
    };

    /**
     * Converts raw logits or probabilities shaped as (batch, classes) alongside
     * target labels shaped as (batch, 1) or (batch) into a binary series ready
     * for curve construction. The final column is treated as the positive class
     * probability.
     */
    inline auto MakeSeriesFromTensor(torch::Tensor logitsOrProbabilities,
                                     torch::Tensor targets,
                                     std::string label) -> BinarySeries
    {
        if (!logitsOrProbabilities.defined()) {
            throw std::invalid_argument("Reliability curves require defined logits/probabilities.");
        }
        if (!targets.defined()) {
            throw std::invalid_argument("Reliability curves require defined target labels.");
        }

        if (logitsOrProbabilities.dim() != 2) {
            throw std::invalid_argument(
                "Reliability curves expect logits/probabilities shaped as (batch, classes).");
        }

        const auto batch = logitsOrProbabilities.size(0);
        const auto classes = logitsOrProbabilities.size(1);
        if (batch <= 0) {
            throw std::invalid_argument("Reliability curves require at least one prediction.");
        }
        if (classes < 2) {
            throw std::invalid_argument("Reliability curves currently support binary classification (classes = 2).");
        }

        torch::Tensor probabilities = logitsOrProbabilities;
        const bool valuesInRange = logitsOrProbabilities.ge(0.0).all().item<bool>()
            && logitsOrProbabilities.le(1.0).all().item<bool>();
        bool rowsSumToOne = false;
        if (valuesInRange) {
            const auto rowSums = logitsOrProbabilities.sum(1);
            const auto ones = torch::ones_like(rowSums);
            rowsSumToOne = torch::allclose(rowSums, ones, 1e-4, 1e-4);
        }
        if (!valuesInRange || !rowsSumToOne) {
            probabilities = torch::softmax(logitsOrProbabilities, 1);
        }

        probabilities = probabilities.to(torch::kDouble).cpu();
        auto positiveProbabilities = probabilities.select(1, classes - 1).contiguous();

        BinarySeries series{};
        series.scores.resize(static_cast<std::size_t>(batch));
        const auto* probsData = positiveProbabilities.data_ptr<double>();
        std::copy(probsData, probsData + batch, series.scores.begin());
        series.label = std::move(label);

        torch::Tensor flattenedTargets;
        if (targets.dim() == 2) {
            if (targets.size(1) != 1) {
                throw std::invalid_argument("Target tensor must be shaped as (batch, 1) for reliability curves.");
            }
            flattenedTargets = targets.reshape({targets.size(0)});
        } else if (targets.dim() == 1) {
            flattenedTargets = targets;
        } else {
            throw std::invalid_argument("Target tensor must be one- or two-dimensional for reliability curves.");
        }

        if (flattenedTargets.size(0) != batch) {
            throw std::invalid_argument("Logits/probabilities and targets must share the same batch dimension.");
        }

        flattenedTargets = flattenedTargets.to(torch::kLong).cpu().contiguous();
        series.targets.resize(static_cast<std::size_t>(batch));
        const auto* targetsData = flattenedTargets.data_ptr<int64_t>();
        std::copy(targetsData, targetsData + batch, series.targets.begin());
        for (const auto target : series.targets) {
            if (target != 0 && target != 1) {
                throw std::invalid_argument("Reliability curves expect binary targets encoded as {0, 1}.");
            }
        }

        return series;
    }

    /**
     * Normalises probability/target vectors into a binary series for curve
     * construction. Probabilities must lie in [0, 1] and targets must be encoded
     * as {0, 1}.
     */
    inline auto MakeSeriesFromVectors(const std::vector<double>& probabilities,
                                      const std::vector<int64_t>& targets,
                                      std::string label) -> BinarySeries
    {
        if (probabilities.empty()) {
            throw std::invalid_argument("Reliability curves require at least one probability value.");
        }
        if (probabilities.size() != targets.size()) {
            throw std::invalid_argument("Probability and target vectors must have the same length.");
        }

        BinarySeries series{};
        series.scores = probabilities;
        series.targets = targets;
        series.label = std::move(label);

        for (const auto probability : series.scores) {
            if (!(probability >= 0.0 && probability <= 1.0)) {
                throw std::invalid_argument("Probabilities must lie within the range [0, 1].");
            }
        }

        for (const auto target : series.targets) {
            if (target != 0 && target != 1) {
                throw std::invalid_argument("Reliability curves expect binary targets encoded as {0, 1}.");
            }
        }

        return series;
    }

    inline auto BuildCurve(const BinarySeries& series) -> BinaryCurve
    {
        if (series.scores.size() != series.targets.size()) {
            throw std::invalid_argument("Probability scores and targets must share the same length.");
        }
        if (series.scores.empty()) {
            throw std::invalid_argument("Reliability curves require at least one data point.");
        }

        BinaryCurve curve{};
        curve.totalPositives = static_cast<std::size_t>(std::count(series.targets.begin(), series.targets.end(), 1));
        curve.totalNegatives = series.targets.size() - curve.totalPositives;

        if (curve.totalPositives == 0 || curve.totalNegatives == 0) {
            throw std::invalid_argument("Reliability curves require both positive and negative samples.");
        }

        std::vector<std::pair<double, int64_t>> paired(series.scores.size());
        for (std::size_t i = 0; i < series.scores.size(); ++i) {
            paired[i] = std::make_pair(series.scores[i], series.targets[i]);
        }

        std::sort(paired.begin(), paired.end(), [](const auto& lhs, const auto& rhs) {
            if (lhs.first == rhs.first) {
                return lhs.second > rhs.second;
            }
            return lhs.first > rhs.first;
        });

        curve.points.reserve(paired.size() + 2);
        curve.points.push_back(BinaryCurvePoint{std::numeric_limits<double>::infinity(), 0, 0, curve.totalPositives, curve.totalNegatives});

        std::size_t cumulativeTP = 0;
        std::size_t cumulativeFP = 0;
        std::size_t index = 0;
        while (index < paired.size()) {
            const double threshold = paired[index].first;
            while (index < paired.size() && paired[index].first == threshold) {
                if (paired[index].second == 1) {
                    ++cumulativeTP;
                } else {
                    ++cumulativeFP;
                }
                ++index;
            }

            const std::size_t fn = curve.totalPositives - cumulativeTP;
            const std::size_t tn = curve.totalNegatives - cumulativeFP;
            curve.points.push_back(BinaryCurvePoint{threshold, cumulativeTP, cumulativeFP, fn, tn});
        }

        curve.points.push_back(BinaryCurvePoint{std::numeric_limits<double>::lowest(),
                                               curve.totalPositives,
                                               curve.totalNegatives,
                                               0,
                                               0});

        return curve;
    }

    inline auto DefaultLabelForIndex(std::size_t index) -> std::string
    {
        switch (index) {
            case 0:
                return "Primary";
            case 1:
                return "Secondary";
            case 2:
                return "Tertiary";
            default:
                return "Series " + std::to_string(index + 1);
        }
    }

    namespace {
        [[nodiscard]] auto ensure_two_dimensional(torch::Tensor logits) -> torch::Tensor
        {
            if (!logits.defined()) {
                throw std::invalid_argument("Reliability curves require defined model outputs.");
            }
            if (logits.dim() == 1) {
                logits = logits.unsqueeze(1);
            } else if (logits.dim() > 2) {
                logits = logits.reshape({logits.size(0), -1});
            }
            return logits;
        }

        [[nodiscard]] auto flatten_targets(torch::Tensor targets) -> torch::Tensor
        {
            if (!targets.defined()) {
                throw std::invalid_argument("Reliability curves require defined target labels.");
            }

            if (targets.dim() == 2) {
                if (targets.size(1) != 1) {
                    throw std::invalid_argument("Target tensor must be shaped as (batch, 1) for reliability curves.");
                }
                targets = targets.reshape({targets.size(0)});
            } else if (targets.dim() != 1) {
                throw std::invalid_argument("Target tensor must be one- or two-dimensional for reliability curves.");
            }

            return targets.to(torch::kLong);
        }

        [[nodiscard]] auto build_binary_projection(torch::Tensor logits, torch::Tensor targets)
            -> std::pair<torch::Tensor, torch::Tensor>
        {
            logits = ensure_two_dimensional(std::move(logits));
            if (logits.dim() != 2) {
                throw std::invalid_argument("Reliability curves expect model outputs shaped as (batch, classes).");
            }

            const auto batch = logits.size(0);
            const auto classes = logits.size(1);
            if (batch <= 0) {
                throw std::invalid_argument("Reliability curves require at least one prediction.");
            }
            if (classes < 2) {
                throw std::invalid_argument("Reliability curves currently support binary classification (classes = 2).");
            }

            auto probabilities = torch::softmax(logits, 1).to(torch::kDouble).cpu();
            auto positive_probabilities = probabilities.select(1, classes - 1).contiguous();
            auto negative_probabilities = (1.0 - positive_probabilities).contiguous();
            auto binary_probabilities = torch::stack({negative_probabilities, positive_probabilities}, 1).contiguous();

            auto flattened_targets = flatten_targets(std::move(targets)).to(torch::kCPU);
            auto binary_targets = (flattened_targets == (classes - 1)).to(torch::kLong);

            return {std::move(binary_probabilities), std::move(binary_targets)};
        }

        [[nodiscard]] auto make_series_from_binary(torch::Tensor probabilities,
                                                   torch::Tensor targets,
                                                   std::string label) -> BinarySeries
        {
            return MakeSeriesFromTensor(std::move(probabilities), std::move(targets), std::move(label));
        }

        [[nodiscard]] auto run_inference(Model& model, torch::Tensor inputs) -> torch::Tensor
        {
            torch::NoGradGuard no_grad{};
            const bool was_training = model.is_training();
            model.eval();

            torch::Tensor outputs;
            try {
                ForwardOptions options{};
                options.max_chunk_size = 256;
                outputs = model.forward(std::move(inputs), options);
            } catch (...) {
                if (was_training) {
                    model.train();
                }
                throw;
            }

            if (was_training) {
                model.train();
            }

            return outputs.detach().to(torch::kCPU);
        }
    }

    inline auto MakeSeriesFromSamples(Model& model,
                                      torch::Tensor inputs,
                                      torch::Tensor targets,
                                      std::string label) -> BinarySeries
    {
        if (!inputs.defined()) {
            throw std::invalid_argument("Reliability curves require defined input samples.");
        }
        if (!targets.defined()) {
            throw std::invalid_argument("Reliability curves require defined target labels.");
        }

        auto logits = run_inference(model, std::move(inputs));
        auto [binary_probabilities, binary_targets] = build_binary_projection(std::move(logits), std::move(targets));
        return make_series_from_binary(std::move(binary_probabilities), std::move(binary_targets), std::move(label));
    }

    inline auto MakeSeriesFromSamples(Model& model,
                                      torch::Tensor firstInputs,
                                      torch::Tensor firstTargets,
                                      torch::Tensor secondInputs,
                                      torch::Tensor secondTargets,
                                      std::string firstLabel,
                                      std::string secondLabel) -> std::vector<BinarySeries> {
        std::vector<BinarySeries> series;
        series.reserve(2);
        series.emplace_back(MakeSeriesFromSamples(model,
                                                  std::move(firstInputs),
                                                  std::move(firstTargets),
                                                  std::move(firstLabel)));
        series.emplace_back(MakeSeriesFromSamples(model,
                                                  std::move(secondInputs),
                                                  std::move(secondTargets),
                                                  std::move(secondLabel)));
        return series;
    }
}

namespace Nott::Plot::Details::Reliability::Interpretability {
    inline auto SelectIndices(std::size_t total, std::size_t requested, bool random)
        -> std::vector<std::size_t>
    {
        if (total == 0) {
            return {};
        }

        std::size_t count = requested == 0 ? total : std::min(requested, total);
        count = std::max<std::size_t>(1, count);

        std::vector<std::size_t> all(total);
        std::iota(all.begin(), all.end(), 0);

        if (!random || count == total) {
            return {all.begin(), all.begin() + static_cast<std::ptrdiff_t>(count)};
        }

        std::vector<std::size_t> selected;
        selected.reserve(count);
        std::sample(all.begin(), all.end(), std::back_inserter(selected), count, std::mt19937{std::random_device{}()});
        std::sort(selected.begin(), selected.end());
        return selected;
    }

    inline auto FlattenTargets(torch::Tensor targets, std::size_t expected) -> torch::Tensor
    {
        if (!targets.defined()) {
            throw std::invalid_argument("Targets must be defined for interpretability plots.");
        }

        if (targets.dim() == 2) {
            if (targets.size(1) != 1) {
                throw std::invalid_argument("Target tensor must have shape (batch) or (batch, 1).");
            }
            targets = targets.reshape({targets.size(0)});
        } else if (targets.dim() != 1) {
            throw std::invalid_argument("Target tensor must be one- or two-dimensional.");
        }

        if (static_cast<std::size_t>(targets.size(0)) != expected) {
            throw std::invalid_argument("Input batch and targets must share the same leading dimension.");
        }

        targets = targets.to(torch::kLong).cpu().contiguous();
        return targets;
    }
}
#endif //Nott_RELIABILITY_CURVE_UTILS_HPP