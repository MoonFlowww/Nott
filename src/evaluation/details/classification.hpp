#ifndef Nott_CLASSIFICATION_HPP
#define Nott_CLASSIFICATION_HPP

#include <algorithm>
#include <cmath>
#include <deque>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <tuple>
#include <type_traits>

#include <torch/torch.h>

#include "../../common/streaming.hpp"
#include "../../metric/metric.hpp"
#include "../../plot/details/statistics.hpp"
#include "../../utils/terminal.hpp"

namespace Nott::Evaluation::Details::Classification {
    struct Descriptor { };
    struct MultiDescriptor { };
    struct SegmentationDescriptor { };

    struct Options {
        std::size_t batch_size{8};
        std::size_t buffer_vram{0};
        std::size_t calibration_bins{15};
        bool print_summary{true};
        bool print_per_class{true};
        std::ostream* stream{&std::cout};
        Utils::Terminal::FrameStyle frame_style{Utils::Terminal::FrameStyle::Box};
    };

    struct Report {
        struct SummaryRow {
            Metric::Classification::Kind metric{};
            double macro{0.0};
            double weighted{0.0};
        };

        std::vector<Metric::Classification::Kind> order{};
        std::vector<SummaryRow> summary{};
        std::vector<std::int64_t> labels{};
        std::vector<std::size_t> support{};
        std::vector<std::vector<double>> per_class{};
        std::size_t total_samples{0};
        double overall_accuracy{std::numeric_limits<double>::quiet_NaN()};
    };

    namespace detail {
        inline double safe_div(double num, double den) {
            constexpr double kEps = 1e-12;
            return (std::abs(den) < kEps) ? 0.0 : (num / den);
        }

        inline std::size_t clamp_bin(double value, std::size_t bins) {
            if (bins == 0) {
                return 0;
            }
            const double clamped = std::clamp(value, 0.0, 1.0);
            if (clamped >= 1.0) {
                return bins - 1;
            }
            const double scaled = clamped * static_cast<double>(bins);
            std::size_t idx = static_cast<std::size_t>(scaled);
            if (idx >= bins) {
                idx = bins - 1;
            }
            return idx;
        }

        inline std::string_view metric_name(Metric::Classification::Kind kind) {
            using MetricKind = Metric::Classification::Kind;
            switch (kind) {
                case MetricKind::Accuracy: return "Accuracy";
                case MetricKind::AUCROC: return "AUC ROC";
                case MetricKind::BalancedAccuracy: return "Balanced accuracy";
                case MetricKind::BalancedErrorRate: return "Balanced error rate";
                case MetricKind::F1: return "F1 score";
                case MetricKind::FBeta0Point5: return "F0.5 score";
                case MetricKind::FBeta2: return "F2 score";
                case MetricKind::FalseDiscoveryRate: return "False discovery rate";
                case MetricKind::FalseNegativeRate: return "False negative rate";
                case MetricKind::FalseOmissionRate: return "False omission rate";
                case MetricKind::FalsePositiveRate: return "False positive rate";
                case MetricKind::FowlkesMallows: return "Fowlkes-Mallows";
                case MetricKind::HammingLoss: return "Hamming loss";
                case MetricKind::Informedness: return "Informedness";
                case MetricKind::JaccardIndexMicro: return "Jaccard index (micro)";
                case MetricKind::JaccardIndexMacro: return "Jaccard index (macro)";
                case MetricKind::Markness: return "Markness";
                case MetricKind::Matthews: return "Matthews correlation";
                case MetricKind::NegativeLikelihoodRatio: return "Negative likelihood ratio";
                case MetricKind::NegativePredictiveValue: return "Negative predictive value";
                case MetricKind::PositiveLikelihoodRatio: return "Positive likelihood ratio";
                case MetricKind::PositivePredictiveValue: return "Positive predictive value";
                case MetricKind::Precision: return "Precision";
                case MetricKind::Prevalence: return "Prevalence";
                case MetricKind::Recall: return "Recall";
                case MetricKind::Top1Error: return "Top-1 error";
                case MetricKind::Top3Error: return "Top-3 error";
                case MetricKind::Top5Error: return "Top-5 error";
                case MetricKind::Top1Accuracy: return "Top-1 accuracy";
                case MetricKind::Top3Accuracy: return "Top-3 accuracy";
                case MetricKind::Top5Accuracy: return "Top-5 accuracy";
                case MetricKind::Specificity: return "Specificity";
                case MetricKind::ThreatScore: return "Threat score";
                case MetricKind::TrueNegativeRate: return "True negative rate";
                case MetricKind::TruePositiveRate: return "True positive rate";
                case MetricKind::YoudenIndex: return "Youden index";
                case MetricKind::LogLoss: return "Log loss";
                case MetricKind::BrierScore: return "Brier score";
                case MetricKind::BrierSkillScore: return "Brier skill score";
                case MetricKind::ExpectedCalibrationError: return "Expected calibration error";
                case MetricKind::MaximumCalibrationError: return "Maximum calibration error";
                case MetricKind::CalibrationSlope: return "Calibration slope";
                case MetricKind::CalibrationIntercept: return "Calibration intercept";
                case MetricKind::HosmerLemeshowPValue: return "Hosmer-Lemeshow p-value";
                case MetricKind::KolmogorovSmirnovStatistic: return "Kolmogorov-Smirnov statistic";
                case MetricKind::CohensKappa: return "Cohen's kappa";
                case MetricKind::ConfusionEntropy: return "Confusion entropy";
                case MetricKind::CoverageError: return "Coverage error";
                case MetricKind::LabelRankingAveragePrecision: return "Label ranking average precision";
                case MetricKind::SubsetAccuracy: return "Subset accuracy";
                case MetricKind::AUPRC: return "AUPRC";
                case MetricKind::AUPRG: return "AUPRG";
                case MetricKind::GiniCoefficient: return "Gini coefficient";
                case MetricKind::HausdorffDistance: return "Hausdorff distance";
                case MetricKind::BoundaryIoU: return "Boundary IoU";
            }
            return "Metric";
        }

        inline std::string format_double(double value) {
            if (!std::isfinite(value)) {
                return "nan";
            }
            std::ostringstream out;
            out << std::fixed << std::setprecision(6) << value;
            return out.str();
        }

        inline std::string format_size(std::size_t value) {
            return std::to_string(value);
        }

        struct ClassCounts {
            std::size_t support{0};
            std::size_t predicted{0};
            std::size_t tp{0};
            std::size_t fp{0};
            std::size_t fn{0};
            std::size_t tn{0};
            double log_loss_sum{0.0};
            double brier_sum{0.0};
        };

        struct ProbabilityCurveData {
            std::vector<double> scores{};
            std::vector<int> labels{};
        };

        inline double compute_auc(const ProbabilityCurveData& data) {
            const std::size_t total = data.labels.size();
            if (total == 0) {
                return std::numeric_limits<double>::quiet_NaN();
            }
            std::size_t positives = 0;
            for (int label : data.labels) {
                positives += (label == 1) ? 1 : 0;
            }
            const std::size_t negatives = total - positives;
            if (positives == 0 || negatives == 0) {
                return std::numeric_limits<double>::quiet_NaN();
            }

            std::vector<std::pair<double, int>> pairs(total);
            for (std::size_t i = 0; i < total; ++i) {
                pairs[i] = {data.scores[i], data.labels[i]};
            }
            std::sort(pairs.begin(), pairs.end(), [](const auto& a, const auto& b) {
                if (a.first == b.first) {
                    return a.second > b.second;
                }
                return a.first > b.first;
            });

            double auc = 0.0;
            double tp = 0.0;
            double fp = 0.0;
            double prev_tpr = 0.0;
            double prev_fpr = 0.0;
            double prev_score = std::numeric_limits<double>::infinity();
            for (const auto& [score, label] : pairs) {
                if (score != prev_score) {
                    const double tpr = tp / static_cast<double>(positives);
                    const double fpr = fp / static_cast<double>(negatives);
                    auc += (fpr - prev_fpr) * (tpr + prev_tpr) * 0.5;
                    prev_tpr = tpr;
                    prev_fpr = fpr;
                    prev_score = score;
                }
                if (label == 1) {
                    tp += 1.0;
                } else {
                    fp += 1.0;
                }
            }
            const double tpr = tp / static_cast<double>(positives);
            const double fpr = fp / static_cast<double>(negatives);
            auc += (fpr - prev_fpr) * (tpr + prev_tpr) * 0.5;
            return std::clamp(auc, 0.0, 1.0);
        }


        inline double compute_average_precision(const ProbabilityCurveData& data) {
            const std::size_t total = data.labels.size();
            if (total == 0) {
                return std::numeric_limits<double>::quiet_NaN();
            }
            std::size_t positives = 0;
            for (int label : data.labels) {
                positives += (label == 1) ? 1 : 0;
            }
            if (positives == 0) {
                return std::numeric_limits<double>::quiet_NaN();
            }

            std::vector<std::pair<double, int>> pairs(total);
            for (std::size_t i = 0; i < total; ++i) {
                pairs[i] = {data.scores[i], data.labels[i]};
            }
            std::sort(pairs.begin(), pairs.end(), [](const auto& a, const auto& b) {
                if (a.first == b.first) {
                    return a.second > b.second;
                }
                return a.first > b.first;
            });

            double tp = 0.0;
            double fp = 0.0;
            double sum_precision = 0.0;
            for (const auto& [_, label] : pairs) {
                if (label == 1) {
                    tp += 1.0;
                    sum_precision += tp / (tp + fp);
                } else {
                    fp += 1.0;

                }
            }
            return sum_precision / static_cast<double>(positives);
        }

        void Print(const Report& report, const Options& options);

        inline double compute_gini(double auc) {
            if (!std::isfinite(auc)) {
                return std::numeric_limits<double>::quiet_NaN();
            }
            return (2.0 * auc) - 1.0;
        }

        inline double compute_mcc(const std::vector<std::vector<std::size_t>>& confusion,
                                   std::size_t total_samples) {
            if (total_samples == 0) {
                return 0.0;
            }
            const std::size_t num_classes = confusion.size();
            double diag_sum = 0.0;
            std::vector<double> row_sums(num_classes, 0.0);
            std::vector<double> col_sums(num_classes, 0.0);
            for (std::size_t i = 0; i < num_classes; ++i) {
                for (std::size_t j = 0; j < num_classes; ++j) {
                    const double value = static_cast<double>(confusion[i][j]);
                    if (i == j) {
                        diag_sum += value;
                    }
                    row_sums[i] += value;
                    col_sums[j] += value;
                }
            }
            double sum_row_sq = 0.0;
            double sum_col_sq = 0.0;
            double row_col_product = 0.0;
            for (std::size_t i = 0; i < num_classes; ++i) {
                sum_row_sq += row_sums[i] * row_sums[i];
                sum_col_sq += col_sums[i] * col_sums[i];
                row_col_product += row_sums[i] * col_sums[i];
            }
            const double n = static_cast<double>(total_samples);
            const double numerator = (n * diag_sum) - row_col_product;
            const double denominator = std::sqrt((n * n - sum_row_sq) * (n * n - sum_col_sq));
            if (std::abs(denominator) < 1e-12) {
                return 0.0;
            }
            return numerator / denominator;
        }


        inline double compute_confusion_entropy(const std::vector<std::vector<std::size_t>>& confusion,
                                                std::size_t total_samples) {
            if (total_samples == 0) {
                return 0.0;
            }
            const std::size_t num_classes = confusion.size();
            std::vector<double> row_prob(num_classes, 0.0);
            std::vector<double> col_prob(num_classes, 0.0);
            for (std::size_t i = 0; i < num_classes; ++i) {
                for (std::size_t j = 0; j < num_classes; ++j) {
                    const double value = static_cast<double>(confusion[i][j]) / static_cast<double>(total_samples);
                    row_prob[i] += value;
                    col_prob[j] += value;
                }
            }

            double entropy = 0.0;
            constexpr double kEps = 1e-12;
            for (std::size_t i = 0; i < num_classes; ++i) {
                for (std::size_t j = 0; j < num_classes; ++j) {
                    const double pij = static_cast<double>(confusion[i][j]) / static_cast<double>(total_samples);
                    if (pij <= 0.0) {
                        continue;
                    }
                    const double denom = row_prob[i] * col_prob[j];
                    if (denom <= 0.0) {
                        continue;
                    }
                    entropy -= pij * std::log((pij + kEps) / (denom + kEps));
                }
            }
            return entropy;
        }

        inline double compute_jaccard_micro(std::size_t total_tp,
                                             std::size_t total_fp,
                                             std::size_t total_fn) {
            const double numerator = static_cast<double>(total_tp);
            const double denominator = static_cast<double>(total_tp + total_fp + total_fn);
            return safe_div(numerator, denominator);
        }

        inline double compute_brier_baseline(const std::vector<std::size_t>& support,
                                             double total_samples) {
            if (total_samples <= 0.0) {
                return std::numeric_limits<double>::quiet_NaN();
            }
            double sum_sq = 0.0;
            for (auto value : support) {
                const double freq = static_cast<double>(value);
                sum_sq += freq * freq;
            }
            const double denom = total_samples * total_samples;
            return 1.0 - (sum_sq / denom);
        }

        inline std::pair<double, double> compute_calibration_coefficients(const std::vector<double>& probs,
                                                                          const std::vector<int>& outcomes) {
            if (probs.empty() || probs.size() != outcomes.size()) {
                return {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN()};
            }
            std::vector<double> x(probs.size(), 0.0);
            for (std::size_t i = 0; i < probs.size(); ++i) {
                const double p = std::clamp(probs[i], 1e-12, 1.0 - 1e-12);
                x[i] = std::log(p / (1.0 - p));
            }

            double beta0 = 0.0;
            double beta1 = 1.0;
            for (int iter = 0; iter < 25; ++iter) {
                double grad0 = 0.0;
                double grad1 = 0.0;
                double h00 = 0.0;
                double h01 = 0.0;
                double h11 = 0.0;

                for (std::size_t i = 0; i < x.size(); ++i) {
                    const double linear = beta0 + beta1 * x[i];
                    const double pred = 1.0 / (1.0 + std::exp(-linear));
                    const double diff = pred - static_cast<double>(outcomes[i]);
                    const double weight = pred * (1.0 - pred);
                    grad0 += diff;
                    grad1 += diff * x[i];
                    h00 += weight;
                    h01 += weight * x[i];
                    h11 += weight * x[i] * x[i];
                }

                const double det = h00 * h11 - h01 * h01;
                if (std::abs(det) < 1e-12) {
                    break;
                }
                const double delta0 = ( grad0 * h11 - grad1 * h01) / det;
                const double delta1 = (-grad0 * h01 + grad1 * h00) / det;
                beta0 -= delta0;
                beta1 -= delta1;
                if (std::abs(delta0) < 1e-8 && std::abs(delta1) < 1e-8) {
                    break;
                }
            }
            return {beta1, beta0};
        }

        using Nott::Plot::Details::compute_kolmogorov_smirnov; // TODO: KS
        inline torch::Tensor squeeze_spatial(torch::Tensor tensor) {
            if (!tensor.defined()) {
                return tensor;
            }
            tensor = tensor.squeeze();
            while (tensor.dim() > 2) {
                if (tensor.size(0) == 1) {
                    tensor = tensor.squeeze(0);
                } else {
                    tensor = std::get<0>(tensor.max(0, /*keepdim=*/false));
                }
            }
            if (tensor.dim() < 2) {
                if (tensor.numel() == 0) {
                    return {};
                }
                tensor = tensor.reshape({1, tensor.numel()});
            }
            return tensor.contiguous();
        }

        inline torch::Tensor binarize_mask(torch::Tensor tensor, double threshold) {
            if (!tensor.defined()) {
                return tensor;
            }
            if (tensor.dtype() != torch::kDouble) {
                tensor = tensor.to(torch::kDouble);
            }
            return tensor.ge(threshold).to(torch::kUInt8).contiguous();
        }

        inline torch::Tensor compute_boundary(torch::Tensor binary_mask) {
            if (!binary_mask.defined()) {
                return binary_mask;
            }
            const auto height = binary_mask.size(0);
            const auto width = binary_mask.size(1);
            auto boundary = torch::zeros_like(binary_mask, torch::kUInt8);
            auto src = binary_mask.accessor<std::uint8_t, 2>();
            auto dst = boundary.accessor<std::uint8_t, 2>();
            for (std::int64_t y = 0; y < height; ++y) {
                for (std::int64_t x = 0; x < width; ++x) {
                    if (!src[y][x]) {
                        continue;
                    }
                    bool full_neighbourhood = true;
                    for (std::int64_t dy = -1; dy <= 1 && full_neighbourhood; ++dy) {
                        for (std::int64_t dx = -1; dx <= 1; ++dx) {
                            const auto ny = y + dy;
                            const auto nx = x + dx;
                            if (ny < 0 || ny >= height || nx < 0 || nx >= width) {
                                full_neighbourhood = false;
                                break;
                            }
                            if (!src[ny][nx]) {
                                full_neighbourhood = false;
                                break;
                            }
                        }
                    }
                    if (!full_neighbourhood) {
                        dst[y][x] = 1;
                    }
                }
            }
            return boundary;
        }

        inline torch::Tensor dilate_mask(const torch::Tensor& mask, std::int64_t radius) {
            if (!mask.defined()) {
                return mask;
            }
            if (radius <= 0) {
                return mask.clone();
            }
            auto dilated = torch::zeros_like(mask, torch::kUInt8);
            const auto height = mask.size(0);
            const auto width = mask.size(1);
            auto src = mask.accessor<std::uint8_t, 2>();
            auto dst = dilated.accessor<std::uint8_t, 2>();
            for (std::int64_t y = 0; y < height; ++y) {
                for (std::int64_t x = 0; x < width; ++x) {
                    if (!src[y][x]) {
                        continue;
                    }
                    const auto y_min = std::max<std::int64_t>(0, y - radius);
                    const auto y_max = std::min<std::int64_t>(height - 1, y + radius);
                    const auto x_min = std::max<std::int64_t>(0, x - radius);
                    const auto x_max = std::min<std::int64_t>(width - 1, x + radius);
                    for (std::int64_t ny = y_min; ny <= y_max; ++ny) {
                        for (std::int64_t nx = x_min; nx <= x_max; ++nx) {
                            dst[ny][nx] = 1;
                        }
                    }
                }
            }
            return dilated;
        }

        inline double compute_boundary_iou(torch::Tensor pred_mask, torch::Tensor target_mask, double threshold) {
            auto pred_binary = compute_boundary(binarize_mask(pred_mask, threshold));
            auto target_binary = compute_boundary(binarize_mask(target_mask, threshold));
            const auto pred_count = pred_binary.sum().item<std::int64_t>();
            const auto target_count = target_binary.sum().item<std::int64_t>();
            if (pred_count == 0 && target_count == 0) {
                return 1.0;
            }
            const double height = static_cast<double>(pred_binary.size(0));
            const double width = static_cast<double>(pred_binary.size(1));
            const auto radius = std::max<std::int64_t>(1, static_cast<std::int64_t>(std::round(0.02 * std::hypot(height, width))));
            auto pred_dilated = dilate_mask(pred_binary, radius);
            auto target_dilated = dilate_mask(target_binary, radius);
            auto intersection = (pred_dilated & target_dilated).to(torch::kInt64).sum().item<double>();
            auto uni = (pred_dilated | target_dilated).to(torch::kInt64).sum().item<double>();
            if (uni <= 0.0) {
                return std::numeric_limits<double>::quiet_NaN();
            }
            return intersection / uni;
        }

        inline double compute_hausdorff_distance(torch::Tensor pred_mask, torch::Tensor target_mask, double threshold) {
            auto pred_binary = compute_boundary(binarize_mask(pred_mask, threshold));
            auto target_binary = compute_boundary(binarize_mask(target_mask, threshold));
            auto pred_points = torch::nonzero(pred_binary);
            auto target_points = torch::nonzero(target_binary);
            if (pred_points.size(0) == 0 || target_points.size(0) == 0) {
                return std::numeric_limits<double>::quiet_NaN();
            }
            pred_points = pred_points.to(torch::kDouble);
            target_points = target_points.to(torch::kDouble);
            auto distances = torch::cdist(pred_points, target_points, 2);
            if (!distances.defined() || distances.numel() == 0) {
                return std::numeric_limits<double>::quiet_NaN();
            }
            const double directed_pred = std::get<0>(distances.min(1, /*keepdim=*/false)).max().item<double>();
            const double directed_target = std::get<0>(distances.min(0, /*keepdim=*/false)).max().item<double>();
            return std::max(directed_pred, directed_target);
        }
    }

    template <class Container>
    [[nodiscard]] auto normalise_requests(const Container& metrics)
        -> std::vector<Metric::Classification::Kind>
    {
        std::vector<Metric::Classification::Kind> kinds;
        kinds.reserve(metrics.size());
        for (const auto& descriptor : metrics) {
            kinds.push_back(descriptor.kind);
        }
        return kinds;
    }

    void Print(const Report& report, const Options& options);

    template <class Model>
    [[nodiscard]] auto Evaluate(Model& model,
                                torch::Tensor inputs,
                                torch::Tensor targets,
                                Descriptor,
                                const std::vector<Metric::Classification::Descriptor>& descriptors,
                                const Options& options) -> Report
    {
        return Evaluate(model,
                        std::move(inputs),
                        std::move(targets),
                        descriptors,
                        options);
    }

    template <class Model>
    [[nodiscard]] auto Evaluate(Model& model,
                                torch::Tensor inputs,
                                torch::Tensor targets,
                                MultiDescriptor,
                                const std::vector<Metric::Classification::Descriptor>& descriptors,
                                const Options& options) -> Report
    {
        auto metrics = descriptors;
        if (std::find_if(metrics.begin(), metrics.end(), [](const auto& descriptor) {
                return descriptor.kind == Metric::Classification::Kind::JaccardIndexMacro;
            }) == metrics.end()) {
            metrics.push_back(Metric::Classification::JaccardIndexMacro);
        }
        return Evaluate(model,
                        std::move(inputs),
                        std::move(targets),
                        metrics,
                        options);
    }

    template <class Model>
    [[nodiscard]] auto Evaluate(Model& model,
                                torch::Tensor inputs,
                                torch::Tensor targets,
                                SegmentationDescriptor,
                                const std::vector<Metric::Classification::Descriptor>& descriptors,
                                const Options& options) -> Report
    {
        auto metrics = descriptors;
        return Evaluate(model,
                        std::move(inputs),
                        std::move(targets),
                        metrics,
                        options);
    }

    template <class Model>
    [[nodiscard]] auto Evaluate(Model& model,
                                torch::Tensor inputs,
                                torch::Tensor targets,
                                const std::vector<Metric::Classification::Descriptor>& descriptors,
                                const Options& options) -> Report
    {
        using MetricKind = Metric::Classification::Kind;

        auto metric_order = normalise_requests(descriptors);
        if (metric_order.empty()) {
            throw std::invalid_argument("At least one classification metric must be requested for evaluation.");
        }
        if (!inputs.defined() || !targets.defined()) {
            throw std::invalid_argument("Evaluation inputs and targets must be defined.");
        }
        if (inputs.size(0) != targets.size(0)) {
            throw std::invalid_argument("Evaluation inputs and targets must have the same number of samples.");
        }

        Report report{};
        report.order = metric_order;
        report.total_samples = static_cast<std::size_t>(inputs.size(0));
        if (report.total_samples == 0) {
            return report;
        }

        std::size_t total_samples = report.total_samples;
        const std::size_t batch_size = options.batch_size > 0 ? options.batch_size : total_samples;

        bool needs_probabilities = false;
        bool needs_probability_curves = false;
        bool needs_calibration_bins = false;
        bool needs_log_loss = false;
        bool needs_brier = false;
        bool needs_brier_skill = false;
        bool needs_rank_metrics = false;
        bool needs_top3 = false;
        bool needs_top5 = false;
        bool needs_calibration_coefficients = false;
        bool needs_ks = false;
        bool needs_segmentation_metrics = false;
        bool needs_hausdorff = false;
        bool needs_boundary_iou = false;

        for (auto kind : metric_order) {
            switch (kind) {
                case MetricKind::LogLoss:
                    needs_probabilities = true;
                    needs_log_loss = true;
                    break;
                case MetricKind::BrierScore:
                    needs_probabilities = true;
                    needs_brier = true;
                    break;
                case MetricKind::BrierSkillScore:
                    needs_probabilities = true;
                    needs_brier = true;
                    needs_brier_skill = true;
                    break;
                case MetricKind::AUCROC:
                case MetricKind::AUPRC:
                case MetricKind::AUPRG:
                case MetricKind::GiniCoefficient:
                    needs_probabilities = true;
                    needs_probability_curves = true;
                    break;
                case MetricKind::ExpectedCalibrationError:
                case MetricKind::MaximumCalibrationError:
                    needs_probabilities = true;
                    needs_calibration_bins = true;
                    break;
                case MetricKind::CalibrationSlope:
                case MetricKind::CalibrationIntercept:
                case MetricKind::HosmerLemeshowPValue:
                    needs_probabilities = true;
                    needs_calibration_coefficients = true;
                    break;
                case MetricKind::KolmogorovSmirnovStatistic:
                    needs_probabilities = true;
                    needs_ks = true;
                    break;
                case MetricKind::CoverageError:
                case MetricKind::LabelRankingAveragePrecision:
                    needs_probabilities = true;
                    needs_rank_metrics = true;
                    break;
                case MetricKind::SubsetAccuracy:
                    needs_probabilities = true;
                    break;
                case MetricKind::HausdorffDistance:
                    needs_segmentation_metrics = true;
                    needs_hausdorff = true;
                    break;
                case MetricKind::BoundaryIoU:
                    needs_segmentation_metrics = true;
                    needs_boundary_iou = true;
                    break;
                case MetricKind::Top3Accuracy:
                case MetricKind::Top3Error:
                    needs_top3 = true;
                    break;
                case MetricKind::Top5Accuracy:
                case MetricKind::Top5Error:
                    needs_top5 = true;
                    break;
                default:
                    break;
            }
        }


        if (needs_top5) {
            needs_top3 = true;
        }

        torch::NoGradGuard guard;
        const bool was_training = model.is_training();
        model.eval();

        std::size_t num_classes = 0;
        std::vector<detail::ClassCounts> class_counts;
        std::vector<std::int64_t> class_labels;
        std::vector<std::vector<std::size_t>> confusion_matrix;

        std::vector<std::vector<std::size_t>> class_bin_counts;
        std::vector<std::vector<double>> class_bin_confidence;
        std::vector<std::vector<double>> class_bin_correct;
        std::vector<std::size_t> global_bin_counts;
        std::vector<double> global_bin_confidence;
        std::vector<double> global_bin_correct;

        std::vector<detail::ProbabilityCurveData> probability_curves;
        std::vector<double> predicted_max_probabilities;
        std::vector<int> predicted_correct;
        if (needs_calibration_coefficients || needs_ks) {
            predicted_max_probabilities.reserve(total_samples);
            predicted_correct.reserve(total_samples);
        }

        double coverage_sum = 0.0;
        double lrap_sum = 0.0;

        std::size_t top3_hits = 0;
        std::size_t top5_hits = 0;
        std::size_t total_correct = 0;
        std::size_t processed_samples = 0;
        std::size_t ignored_samples = 0;

        double total_log_loss = 0.0;
        double total_brier = 0.0;

        double hausdorff_sum = 0.0;
        std::size_t hausdorff_count = 0;
        double boundary_iou_sum = 0.0;
        std::size_t boundary_iou_count = 0;

        const auto device = model.device();
        const bool non_blocking_transfers = device.is_cuda();
        const bool device_supports_buffer = device.is_cuda();
        if (options.buffer_vram > 0 && !device_supports_buffer) {
            throw std::runtime_error("VRAM buffering during evaluation requires the model to be on a CUDA device.");
        }
        const bool use_buffer = options.buffer_vram > 0;

        const bool expects_channels_last = [&]() {
            if constexpr (requires(const Model& m) { static_cast<bool>(m.expects_channels_last_inputs()); }) {
                return static_cast<bool>(model.expects_channels_last_inputs());
            } else if constexpr (requires(const Model& m) { static_cast<bool>(m.prefers_channels_last_inputs()); }) {
                return static_cast<bool>(model.prefers_channels_last_inputs());
            } else if constexpr (requires(const Model& m) { static_cast<bool>(m.channels_last_inputs()); }) {
                return static_cast<bool>(model.channels_last_inputs());
            }
            return false;
        }();

        auto ensure_memory_format = [&](torch::Tensor tensor) {
            if (!tensor.defined() || !expects_channels_last) {
                return tensor;
            }
            if (tensor.dim() == 4) {
                if (!tensor.is_contiguous(torch::MemoryFormat::ChannelsLast)) {
                    tensor = tensor.contiguous(torch::MemoryFormat::ChannelsLast);
                }
            } else if (tensor.dim() == 5) {
                if (!tensor.is_contiguous(torch::MemoryFormat::ChannelsLast3d)) {
                    tensor = tensor.contiguous(torch::MemoryFormat::ChannelsLast3d);
                }
            }
            return tensor;
        };

        if (use_buffer) {
            if (inputs.defined() && !inputs.device().is_cpu()) {
                inputs = inputs.to(torch::kCPU);
            }
            if (targets.defined() && !targets.device().is_cpu()) {
                targets = targets.to(torch::kCPU);
            }
            inputs = ensure_memory_format(std::move(inputs));
        } else {
            inputs = ensure_memory_format(std::move(inputs));
        }

        const std::size_t total_batches = batch_size == 0
                                        ? std::size_t{0}
                                        : (total_samples + batch_size - 1) / batch_size;

        Nott::StreamingOptions streaming_options{};
        if (batch_size > 0) {
            streaming_options.batch_size = batch_size;
        }
        if (use_buffer) {
            streaming_options.buffer_batches = options.buffer_vram + 1;
        }

        auto prepare_batch = [&](torch::Tensor input_batch, torch::Tensor target_batch)
            -> std::optional<Nott::StreamingBatch>
        {
            if (!input_batch.defined() || !target_batch.defined()) {
                return std::nullopt;
            }

            input_batch = ensure_memory_format(std::move(input_batch));

            Nott::StreamingBatch batch{};
            batch.inputs = std::move(input_batch);
            batch.targets = std::move(target_batch);
            if (batch.targets.defined()) {
                batch.reference_targets = Nott::DeferredHostTensor::from_tensor(batch.targets, non_blocking_transfers);
            }

            return batch;
        };

        auto process_batch = [&](torch::Tensor logits_tensor, Nott::StreamingBatch batch) {
            if (!logits_tensor.defined()) {
                return;
            }

            torch::Tensor segmentation_predictions;
            if (needs_segmentation_metrics) {
                segmentation_predictions = logits_tensor.detach();
            }

            auto logits = needs_segmentation_metrics ? segmentation_predictions : logits_tensor.detach();

            if (logits.dim() == 1) {
                logits = logits.unsqueeze(1);
            }

            if (logits.dim() > 2) {
                std::size_t spatial_elements = 1;
                for (std::int64_t dim = 2; dim < logits.dim(); ++dim) {
                    spatial_elements *= static_cast<std::size_t>(logits.size(dim));
                }
                const auto batch_dim = logits.size(0);
                const auto channel_dim = logits.size(1);
                logits = logits.flatten(2).transpose(1, 2)
                               .reshape({batch_dim * static_cast<std::int64_t>(spatial_elements), channel_dim});
            }
            if (logits.size(1) == 1) {
                auto zeros = torch::zeros_like(logits);
                logits = torch::cat({zeros, logits}, 1);
            }


            if (!logits.device().is_cpu()) {
                if (non_blocking_transfers) {
                    auto deferred_logits = Nott::DeferredHostTensor::from_tensor(logits, /*non_blocking=*/true);
                    logits = deferred_logits.materialize();
                } else {
                    logits = logits.to(torch::kCPU, logits.scalar_type(), /*non_blocking=*/false);
                }
            }
            if (logits.scalar_type() != torch::kFloat64) {
                logits = logits.to(torch::kFloat64);
            }
            if (logits.dim() != 2) {
                throw std::runtime_error("Classification evaluation expects two-dimensional logits.");
            }

            torch::Tensor target_cpu;
            if (batch.reference_targets.defined()) {
                target_cpu = batch.reference_targets.materialize();
            } else {
                target_cpu = batch.targets;
            }

            if (!target_cpu.defined())
                return;
            if (!target_cpu.device().is_cpu()) {
                if (non_blocking_transfers) {
                    auto deferred_targets = Nott::DeferredHostTensor::from_tensor(target_cpu, /*non_blocking=*/true);
                    target_cpu = deferred_targets.materialize();
                } else {
                    target_cpu = target_cpu.to(torch::kCPU, target_cpu.scalar_type(), /*non_blocking=*/false);
                }
            }
            torch::Tensor segmentation_targets;
            if (needs_segmentation_metrics) {
                segmentation_targets = target_cpu;
            }

            std::optional<std::size_t> inferred_classes;
            if (target_cpu.defined() && target_cpu.numel() > 0) {
                if (target_cpu.dim() > 1 && target_cpu.size(1) > 1) {
                    inferred_classes = static_cast<std::size_t>(target_cpu.size(1));
                } else {
                    auto flattened = target_cpu.reshape({-1});
                    if (flattened.dtype() != torch::kLong) {
                        flattened = flattened.to(torch::kLong);
                    }

                    const auto max_label = flattened.max().item<long>();
                    if (max_label >= 0) {
                        inferred_classes = static_cast<std::size_t>(max_label) + 1;
                    }
                }
            }


            const std::size_t batch_classes = static_cast<std::size_t>(logits.size(1));
            const std::size_t current_batch = static_cast<std::size_t>(logits.size(0));

            if (inferred_classes && batch_classes > *inferred_classes && *inferred_classes > 0) {
                const auto start = static_cast<long>(batch_classes - *inferred_classes);
                logits = logits.slice(/*dim=*/1, start, logits.size(1));
            }
            const auto effective_classes = static_cast<std::size_t>(logits.size(1));
            if (num_classes == 0) {
                num_classes = effective_classes;
                class_counts.assign(num_classes, {});
                class_labels.resize(num_classes);
                std::iota(class_labels.begin(), class_labels.end(), 0);
                confusion_matrix.assign(num_classes, std::vector<std::size_t>(num_classes, 0));

                if (needs_calibration_bins) {
                    const auto bins = std::max<std::size_t>(1, options.calibration_bins);
                    class_bin_counts.assign(num_classes, std::vector<std::size_t>(bins, 0));
                    class_bin_confidence.assign(num_classes, std::vector<double>(bins, 0.0));
                    class_bin_correct.assign(num_classes, std::vector<double>(bins, 0.0));
                    global_bin_counts.assign(bins, 0);
                    global_bin_confidence.assign(bins, 0.0);
                    global_bin_correct.assign(bins, 0.0);
                }

                if (needs_probability_curves) {
                    probability_curves.assign(num_classes, {});
                    for (auto& curve : probability_curves) {
                        curve.scores.reserve(total_samples);
                        curve.labels.reserve(total_samples);
                    }
                } else if (num_classes != effective_classes) {
                    throw std::runtime_error("Inconsistent number of classes encountered during evaluation.");
                }
            }

            auto predicted = logits.argmax(1).to(torch::kLong);

            torch::Tensor probabilities;
            double* probabilities_ptr = nullptr;
            std::int64_t probability_stride = 0;
            if (needs_probabilities) {
                probabilities = torch::softmax(logits, 1).contiguous().to(torch::kDouble);
                probabilities_ptr = probabilities.data_ptr<double>();
                probability_stride = probabilities.size(1);
            }

            torch::Tensor topk_indices_tensor;
            if (needs_top3) {
                const auto topk = static_cast<long>(needs_top5 ? 5 : 3);
                auto topk_result = logits.topk(topk, 1, true, true);
                topk_indices_tensor = std::get<1>(topk_result).to(torch::kLong);
            }

            const bool has_multi_channel_targets = target_cpu.dim() > 1 && target_cpu.size(1) == static_cast<long>(num_classes);
            if (has_multi_channel_targets) {
                if (target_cpu.dtype() != torch::kFloat32 && target_cpu.dtype() != torch::kFloat64) {
                    target_cpu = target_cpu.to(torch::kFloat32);
                }
                target_cpu = target_cpu.argmax(1);
            } else if (target_cpu.dtype() != torch::kLong) {
                if (target_cpu.dtype() == torch::kFloat32 || target_cpu.dtype() == torch::kFloat64) {
                    target_cpu = target_cpu.to(torch::kLong);
                } else {
                    target_cpu = target_cpu.to(torch::kLong);
                }
            }


            if (target_cpu.dim() > 1 && target_cpu.size(1) == 1 && target_cpu.dim() == 2) {
                target_cpu = target_cpu.squeeze(1);
            }

            const auto expected_elements = static_cast<std::int64_t>(current_batch);
            if (target_cpu.numel() != expected_elements) {
                target_cpu = target_cpu.reshape({target_cpu.size(0), -1}).flatten(0, 1);
            }
            if (target_cpu.numel() != expected_elements) {
                throw std::runtime_error("Evaluation targets and predictions must share the same number of elements as the logits batch.");
            }

            target_cpu = target_cpu.reshape({expected_elements}).contiguous();

            if (target_cpu.sizes() != predicted.sizes()) {
                throw std::runtime_error("Evaluation targets and predictions must share the same leading shape.");
            }
            auto pred_accessor = predicted.template accessor<long, 1>();
            auto target_accessor = target_cpu.template accessor<long, 1>();
            torch::TensorAccessor<long, 2> topk_accessor{nullptr, nullptr, nullptr};
            if (needs_top3) {
                topk_accessor = topk_indices_tensor.template accessor<long, 2>();
            }

            for (std::size_t i = 0; i < current_batch; ++i) {
                const auto label = target_accessor[i];
                const auto pred = pred_accessor[i];

                if (label < 0 || label >= static_cast<long>(num_classes)) {
                    ignored_samples += 1;
                    continue;
                }
                if (pred < 0 || pred >= static_cast<long>(num_classes)) {
                    ignored_samples += 1;
                    continue;
                }
                processed_samples += 1;
                const auto label_index = static_cast<std::size_t>(label);
                const auto pred_index = static_cast<std::size_t>(pred);

                auto& label_counts = class_counts[label_index];
                auto& pred_counts = class_counts[pred_index];

                label_counts.support += 1;
                pred_counts.predicted += 1;
                confusion_matrix[label_index][pred_index] += 1;

                if (label == pred) {
                    label_counts.tp += 1;
                    total_correct += 1;
                } else {
                    label_counts.fn += 1;
                    pred_counts.fp += 1;
                }

                if (needs_top3) {
                    const auto k_dim = static_cast<std::size_t>(topk_accessor.size(1));
                    bool counted3 = false;
                    bool counted5 = false;
                    for (std::size_t k = 0; k < k_dim; ++k) {
                        const auto top_value = topk_accessor[i][static_cast<long>(k)];
                        if (top_value == label) {
                            if (k < 3) {
                                counted3 = true;
                            }

                            if (needs_top5 && k < 5) {
                                counted5 = true;
                            }

                            if (!needs_top5) {
                                counted5 = counted3;
                            }
                            break;
                        }
                    }
                    if (counted3) {
                        top3_hits += 1;
                    }
                    if (needs_top5 && counted5) {
                        top5_hits += 1;
                    }
                }

                if (needs_probabilities) {
                    const double* row = probabilities_ptr + static_cast<std::size_t>(i) * probability_stride;
                    const double true_probability = std::clamp(row[label_index], 1e-12, 1.0);

                    if (needs_log_loss) {
                        const double loss = -std::log(true_probability);
                        total_log_loss += loss;
                        label_counts.log_loss_sum += loss;
                    }

                    if (needs_brier) {
                        double sample_brier = 0.0;
                        for (std::size_t cls = 0; cls < num_classes; ++cls) {
                            const double expected = (cls == label_index) ? 1.0 : 0.0;
                            const double diff = row[cls] - expected;
                            sample_brier += diff * diff;
                        }
                        label_counts.brier_sum += sample_brier;
                        total_brier += sample_brier;
                    }

                    if (needs_calibration_bins) {
                        const auto bins = global_bin_counts.size();
                        const double confidence = std::clamp(row[pred_index], 0.0, 1.0);
                        const auto bin = detail::clamp_bin(confidence, bins);
                        global_bin_counts[bin] += 1;
                        global_bin_confidence[bin] += confidence;
                        global_bin_correct[bin] += (label == pred) ? 1.0 : 0.0;

                        for (std::size_t cls = 0; cls < num_classes; ++cls) {
                            const double probability = std::clamp(row[cls], 0.0, 1.0);
                            const auto class_bin = detail::clamp_bin(probability, bins);
                            class_bin_counts[cls][class_bin] += 1;
                            class_bin_confidence[cls][class_bin] += probability;
                            class_bin_correct[cls][class_bin] += (cls == label_index) ? 1.0 : 0.0;
                        }
                    }

                    if (needs_probability_curves) {
                        for (std::size_t cls = 0; cls < num_classes; ++cls) {
                            probability_curves[cls].scores.push_back(row[cls]);
                            probability_curves[cls].labels.push_back(cls == label_index ? 1 : 0);
                        }
                    }

                    if (needs_calibration_coefficients || needs_ks) {
                        const double predicted_probability = std::clamp(row[pred_index], 1e-12, 1.0 - 1e-12);
                        predicted_max_probabilities.push_back(predicted_probability);
                        predicted_correct.push_back(label == pred ? 1 : 0);
                    }

                    if (needs_rank_metrics) {
                        const double true_score = row[label_index];
                        std::size_t rank = 1;
                        for (std::size_t cls = 0; cls < num_classes; ++cls) {
                            if (cls == label_index) {
                                continue;
                            }
                            const double score = row[cls];
                            if (score > true_score + 1e-12) {
                                rank += 1;
                            } else if (std::abs(score - true_score) <= 1e-12) {
                                rank += 1;
                            }
                        }
                        coverage_sum += static_cast<double>(rank);
                        lrap_sum += detail::safe_div(1.0, static_cast<double>(rank));
                    }
                }
            }
            if (needs_segmentation_metrics) {
                auto seg_preds = segmentation_predictions;
                auto seg_targets = segmentation_targets;
                if (!seg_preds.device().is_cpu()) {
                    if (non_blocking_transfers) {
                        auto deferred = Nott::DeferredHostTensor::from_tensor(seg_preds, /*non_blocking=*/true);
                        seg_preds = deferred.materialize();
                    } else {
                        seg_preds = seg_preds.to(torch::kCPU, seg_preds.scalar_type(), /*non_blocking=*/false);
                    }
                }
                if (!seg_targets.device().is_cpu()) {
                    if (non_blocking_transfers) {
                        auto deferred = Nott::DeferredHostTensor::from_tensor(seg_targets, /*non_blocking=*/true);
                        seg_targets = deferred.materialize();
                    } else {
                        seg_targets = seg_targets.to(torch::kCPU, seg_targets.scalar_type(), /*non_blocking=*/false);
                    }
                }
                seg_preds = seg_preds.to(torch::kDouble).contiguous();
                seg_targets = seg_targets.to(torch::kDouble).contiguous();
                if (seg_preds.sizes() != seg_targets.sizes()) {
                    throw std::runtime_error("Segmentation metrics require prediction and target tensors to match in shape.");
                }
                const auto sample_count = seg_preds.size(0);
                auto accumulate_segmentation_metrics = [&](torch::Tensor pred_mask, torch::Tensor target_mask) {
                    auto pred_sample = detail::squeeze_spatial(std::move(pred_mask));
                    auto target_sample = detail::squeeze_spatial(std::move(target_mask));
                    if (!pred_sample.defined() || !target_sample.defined()) {
                        return;
                    }
                    if (pred_sample.sizes() != target_sample.sizes()) {
                        return;
                    }
                    if (needs_hausdorff) {
                        const double distance = detail::compute_hausdorff_distance(pred_sample, target_sample, 0.5);
                        if (std::isfinite(distance)) {
                            hausdorff_sum += distance;
                            hausdorff_count += 1;
                        }
                    }
                    if (needs_boundary_iou) {
                        const double biou = detail::compute_boundary_iou(pred_sample, target_sample, 0.5);
                        if (std::isfinite(biou)) {
                            boundary_iou_sum += biou;
                            boundary_iou_count += 1;
                        }
                    }
                };

                for (std::int64_t sample_idx = 0; sample_idx < sample_count; ++sample_idx) {
                    auto pred_sample = seg_preds[sample_idx];
                    auto target_sample = seg_targets[sample_idx];
                    if (!pred_sample.defined() || !target_sample.defined()) {
                        continue;
                    }

                    if (pred_sample.dim() > 2 && target_sample.dim() > 2 && pred_sample.size(0) == target_sample.size(0)) {
                        const auto channels = pred_sample.size(0);
                        for (std::int64_t channel = 0; channel < channels; ++channel) {
                            accumulate_segmentation_metrics(pred_sample.select(0, channel), target_sample.select(0, channel));
                        }
                    } else {
                        accumulate_segmentation_metrics(pred_sample, target_sample);
                    }
                }
                segmentation_predictions = torch::Tensor{};
                segmentation_targets = torch::Tensor{};
            }
            logits = torch::Tensor{};
            probabilities = torch::Tensor{};
            predicted = torch::Tensor{};
            batch.targets = torch::Tensor{};
        };

        model.stream_forward(inputs, targets, streaming_options, prepare_batch, process_batch);
        if (ignored_samples > 0 && options.stream != nullptr) {
            (*options.stream) << "Warning: ignored " << ignored_samples
                              << " sample" << (ignored_samples == 1 ? "" : "s")
                              << " with out-of-range labels or predictions during evaluation.\n";
        }

        total_samples = processed_samples;
        report.total_samples = total_samples;
        if (total_samples == 0) {
            return report;
        }


        const double total_samples_d = static_cast<double>(total_samples);
        for (auto& counts : class_counts) {
            const std::size_t remainder = total_samples - (counts.tp + counts.fp + counts.fn);
            counts.tn = remainder;
        }

        std::vector<double> per_class_ece_values(num_classes, std::numeric_limits<double>::quiet_NaN());
        std::vector<double> per_class_mce_values(num_classes, std::numeric_limits<double>::quiet_NaN());
        double global_ece = std::numeric_limits<double>::quiet_NaN();
        double global_mce = std::numeric_limits<double>::quiet_NaN();
        if (needs_calibration_bins) {
            global_ece = 0.0;
            global_mce = 0.0;
            for (std::size_t b = 0; b < global_bin_counts.size(); ++b) {
                const double count = static_cast<double>(global_bin_counts[b]);
                if (count <= 0.0) {
                    continue;
                }
                const double mean_confidence = global_bin_confidence[b] / count;
                const double accuracy = global_bin_correct[b] / count;
                const double diff = std::abs(mean_confidence - accuracy);
                global_ece += (count / total_samples_d) * diff;
                global_mce = std::max(global_mce, diff);
            }

            per_class_ece_values.assign(num_classes, 0.0);
            per_class_mce_values.assign(num_classes, 0.0);
            for (std::size_t cls = 0; cls < num_classes; ++cls) {
                for (std::size_t b = 0; b < class_bin_counts[cls].size(); ++b) {
                    const double count = static_cast<double>(class_bin_counts[cls][b]);
                    if (count <= 0.0) {
                        continue;
                    }
                    const double mean_confidence = class_bin_confidence[cls][b] / count;
                    const double accuracy = class_bin_correct[cls][b] / count;
                    const double diff = std::abs(mean_confidence - accuracy);
                    per_class_ece_values[cls] += (count / total_samples_d) * diff;
                    per_class_mce_values[cls] = std::max(per_class_mce_values[cls], diff);
                }
            }
        }

        std::vector<double> auc_per_class(num_classes, std::numeric_limits<double>::quiet_NaN());
        std::vector<double> auprc_per_class(num_classes, std::numeric_limits<double>::quiet_NaN());
        std::vector<double> auprg_per_class(num_classes, std::numeric_limits<double>::quiet_NaN());
        std::vector<double> gini_per_class(num_classes, std::numeric_limits<double>::quiet_NaN());
        if (needs_probability_curves) {
            for (std::size_t cls = 0; cls < num_classes; ++cls) {
                const double auc = detail::compute_auc(probability_curves[cls]);
                const double auprc = detail::compute_average_precision(probability_curves[cls]);
                const double auprg = std::numeric_limits<double>::quiet_NaN();
                const double gini = detail::compute_gini(auc);
                auc_per_class[cls] = auc;
                auprc_per_class[cls] = auprc;
                auprg_per_class[cls] = auprg;
                gini_per_class[cls] = gini;
            }
        }

        std::vector<std::size_t> support_values;
        support_values.reserve(num_classes);
        std::size_t total_fp = 0;
        std::size_t total_fn = 0;
        for (const auto& counts : class_counts) {
            support_values.push_back(counts.support);
            total_fp += counts.fp;
            total_fn += counts.fn;
        }

        const double total_support = std::accumulate(
            support_values.begin(), support_values.end(), 0.0,
            [](double acc, std::size_t value) {
                return acc + static_cast<double>(value);
            });

        const double accuracy_global = detail::safe_div(static_cast<double>(total_correct), total_samples_d);
        const double top1_accuracy = accuracy_global;
        const double top3_accuracy = needs_top3 ? detail::safe_div(static_cast<double>(top3_hits), total_samples_d) : top1_accuracy;
        const double top5_accuracy = needs_top5 ? detail::safe_div(static_cast<double>(top5_hits), total_samples_d) : top3_accuracy;
        const double top1_error = 1.0 - top1_accuracy;
        const double top3_error = 1.0 - top3_accuracy;
        const double top5_error = 1.0 - top5_accuracy;
        const double hamming_global = 1.0 - top1_accuracy;

        double pe = 0.0;
        for (const auto& counts : class_counts) {
            const double actual = detail::safe_div(static_cast<double>(counts.support), total_samples_d);
            const double predicted = detail::safe_div(static_cast<double>(counts.predicted), total_samples_d);
            pe += actual * predicted;
        }
        const double global_kappa = detail::safe_div(accuracy_global - pe, 1.0 - pe);

        const double log_loss_global = needs_log_loss ? detail::safe_div(total_log_loss, total_samples_d) : std::numeric_limits<double>::quiet_NaN();
        const double brier_global = needs_brier ? detail::safe_div(total_brier, total_samples_d) : std::numeric_limits<double>::quiet_NaN();
        const double hausdorff_mean = hausdorff_count > 0 ? (hausdorff_sum / static_cast<double>(hausdorff_count)) : std::numeric_limits<double>::quiet_NaN();
        const double boundary_iou_mean = boundary_iou_count > 0 ? (boundary_iou_sum / static_cast<double>(boundary_iou_count)) : std::numeric_limits<double>::quiet_NaN();
        const double coverage_error = needs_rank_metrics ? detail::safe_div(coverage_sum, total_samples_d) : std::numeric_limits<double>::quiet_NaN();
        const double lrap = needs_rank_metrics ? detail::safe_div(lrap_sum, total_samples_d) : std::numeric_limits<double>::quiet_NaN();
        const double subset_accuracy = top1_accuracy;

        const double jaccard_micro = detail::compute_jaccard_micro(total_correct, total_fp, total_fn);

        const double brier_baseline = needs_brier_skill
            ? detail::compute_brier_baseline(support_values, total_samples_d)
            : std::numeric_limits<double>::quiet_NaN();
        double brier_skill_global = std::numeric_limits<double>::quiet_NaN();
        if (needs_brier_skill && std::isfinite(brier_global) && std::isfinite(brier_baseline) && std::abs(brier_baseline) > 1e-12) {
            brier_skill_global = 1.0 - (brier_global / brier_baseline);
        }

        double ks_stat = std::numeric_limits<double>::quiet_NaN();
        if (needs_ks) {
            ks_stat = detail::compute_kolmogorov_smirnov(predicted_max_probabilities, predicted_correct);
        }

        double cal_slope = std::numeric_limits<double>::quiet_NaN();
        double cal_intercept = std::numeric_limits<double>::quiet_NaN();
        if (needs_calibration_coefficients) {
            std::tie(cal_slope, cal_intercept) = detail::compute_calibration_coefficients(
                predicted_max_probabilities, predicted_correct);
        }

        const double hosmer_p_value = std::numeric_limits<double>::quiet_NaN();
        const double mcc_global = detail::compute_mcc(confusion_matrix, total_samples);
        const double confusion_entropy_global = detail::compute_confusion_entropy(confusion_matrix, total_samples);

        std::vector<double> precision_per_class(num_classes, 0.0);
        std::vector<double> recall_per_class(num_classes, 0.0);
        std::vector<double> specificity_per_class(num_classes, 0.0);
        std::vector<double> f1_per_class(num_classes, 0.0);
        std::vector<double> fbeta05_per_class(num_classes, 0.0);
        std::vector<double> fbeta2_per_class(num_classes, 0.0);
        std::vector<double> fdr_per_class(num_classes, 0.0);
        std::vector<double> fnr_per_class(num_classes, 0.0);
        std::vector<double> for_per_class(num_classes, 0.0);
        std::vector<double> fpr_per_class(num_classes, 0.0);
        std::vector<double> npv_per_class(num_classes, 0.0);
        std::vector<double> prevalence_per_class(num_classes, 0.0);
        std::vector<double> threat_per_class(num_classes, 0.0);
        std::vector<double> informedness_per_class(num_classes, 0.0);
        std::vector<double> markness_per_class(num_classes, 0.0);
        std::vector<double> youden_per_class(num_classes, 0.0);
        std::vector<double> jaccard_per_class(num_classes, 0.0);
        std::vector<double> fowlkes_per_class(num_classes, 0.0);
        std::vector<double> accuracy_per_class(num_classes, 0.0);
        std::vector<double> mcc_per_class(num_classes, 0.0);
        std::vector<double> balanced_accuracy_per_class(num_classes, 0.0);
        std::vector<double> balanced_error_per_class(num_classes, 0.0);
        std::vector<double> hamming_per_class(num_classes, 0.0);
        std::vector<double> neg_likelihood_per_class(num_classes, 0.0);
        std::vector<double> pos_likelihood_per_class(num_classes, 0.0);
        std::vector<double> brier_per_class(num_classes, std::numeric_limits<double>::quiet_NaN());
        std::vector<double> log_loss_per_class(num_classes, std::numeric_limits<double>::quiet_NaN());

        for (std::size_t cls = 0; cls < num_classes; ++cls) {
            const auto& counts = class_counts[cls];
            const double tp = static_cast<double>(counts.tp);
            const double fp = static_cast<double>(counts.fp);
            const double fn = static_cast<double>(counts.fn);
            const double tn = static_cast<double>(counts.tn);
            const double support = static_cast<double>(counts.support);

            const double precision = detail::safe_div(tp, tp + fp);
            const double recall = detail::safe_div(tp, tp + fn);
            const double specificity = detail::safe_div(tn, tn + fp);
            const double f1 = (precision + recall > 0.0) ? detail::safe_div(2.0 * precision * recall, precision + recall) : 0.0;
            const double fbeta05 = detail::safe_div(1.25 * tp, 1.25 * tp + 0.25 * fn + fp);
            const double fbeta2 = detail::safe_div(5.0 * tp, 5.0 * tp + 4.0 * fn + fp);
            const double fdr = detail::safe_div(fp, tp + fp);
            const double fnr = detail::safe_div(fn, tp + fn);
            const double forate = detail::safe_div(fn, fn + tn);
            const double fpr = detail::safe_div(fp, fp + tn);
            const double npv = detail::safe_div(tn, tn + fn);
            const double prevalence = detail::safe_div(support, total_samples_d);
            const double threat = detail::safe_div(tp, tp + fp + fn);
            const double informedness = recall + specificity - 1.0;
            const double markness = precision + npv - 1.0;
            const double youden = informedness;
            const double jaccard = detail::safe_div(tp, tp + fp + fn);
            const double fowlkes = (precision > 0.0 && recall > 0.0) ? std::sqrt(precision * recall) : 0.0;
            const double accuracy_cls = detail::safe_div(tp + tn, total_samples_d);
            double mcc = 0.0;
            const double denom = std::sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn));
            if (denom > 0.0) {
                mcc = detail::safe_div(tp * tn - fp * fn, denom);
            }
            const double balanced_accuracy = 0.5 * (recall + specificity);
            const double balanced_error = 1.0 - balanced_accuracy;
            const double hamming = detail::safe_div(fp + fn, total_samples_d);
            const double neg_likelihood = detail::safe_div(1.0 - recall, specificity);
            const double pos_likelihood = detail::safe_div(recall, 1.0 - specificity);

            precision_per_class[cls] = precision;
            recall_per_class[cls] = recall;
            specificity_per_class[cls] = specificity;
            f1_per_class[cls] = f1;
            fbeta05_per_class[cls] = fbeta05;
            fbeta2_per_class[cls] = fbeta2;
            fdr_per_class[cls] = fdr;
            fnr_per_class[cls] = fnr;
            for_per_class[cls] = forate;
            fpr_per_class[cls] = fpr;
            npv_per_class[cls] = npv;
            prevalence_per_class[cls] = prevalence;
            threat_per_class[cls] = threat;
            informedness_per_class[cls] = informedness;
            markness_per_class[cls] = markness;
            youden_per_class[cls] = youden;
            jaccard_per_class[cls] = jaccard;
            fowlkes_per_class[cls] = fowlkes;
            accuracy_per_class[cls] = accuracy_cls;
            mcc_per_class[cls] = mcc;
            balanced_accuracy_per_class[cls] = balanced_accuracy;
            balanced_error_per_class[cls] = balanced_error;
            hamming_per_class[cls] = hamming;
            neg_likelihood_per_class[cls] = neg_likelihood;
            pos_likelihood_per_class[cls] = pos_likelihood;

            if (needs_brier && counts.support > 0) {
                brier_per_class[cls] = counts.brier_sum / support;
            }
            if (needs_log_loss && counts.support > 0) {
                log_loss_per_class[cls] = counts.log_loss_sum / support;
            }
        }

        std::vector<std::vector<double>> per_class_values(
            metric_order.size(),
            std::vector<double>(num_classes, std::numeric_limits<double>::quiet_NaN()));
        std::vector<Report::SummaryRow> summary;
        summary.reserve(metric_order.size());

        for (std::size_t metric_index = 0; metric_index < metric_order.size(); ++metric_index) {
            const auto kind = metric_order[metric_index];
            double macro_sum = 0.0;
            double macro_count = 0.0;
            double weighted_sum = 0.0;
            auto& dest = per_class_values[metric_index];

            for (std::size_t cls = 0; cls < num_classes; ++cls) {
                double value = std::numeric_limits<double>::quiet_NaN();
                switch (kind) {
                    case MetricKind::Accuracy:
                    case MetricKind::Top1Accuracy:
                    case MetricKind::SubsetAccuracy:
                        value = accuracy_per_class[cls];
                        break;
                    case MetricKind::BalancedAccuracy:
                        value = balanced_accuracy_per_class[cls];
                        break;
                    case MetricKind::BalancedErrorRate:
                        value = balanced_error_per_class[cls];
                        break;
                    case MetricKind::F1:
                        value = f1_per_class[cls];
                        break;
                    case MetricKind::FBeta0Point5:
                        value = fbeta05_per_class[cls];
                        break;
                    case MetricKind::FBeta2:
                        value = fbeta2_per_class[cls];
                        break;
                    case MetricKind::FalseDiscoveryRate:
                        value = fdr_per_class[cls];
                        break;
                    case MetricKind::FalseNegativeRate:
                        value = fnr_per_class[cls];
                        break;
                    case MetricKind::FalseOmissionRate:
                        value = for_per_class[cls];
                        break;
                    case MetricKind::FalsePositiveRate:
                        value = fpr_per_class[cls];
                        break;
                    case MetricKind::FowlkesMallows:
                        value = fowlkes_per_class[cls];
                        break;
                    case MetricKind::HammingLoss:
                        value = hamming_per_class[cls];
                        break;
                    case MetricKind::Informedness:
                    case MetricKind::YoudenIndex:
                        value = informedness_per_class[cls];
                        break;
                    case MetricKind::JaccardIndexMicro:
                    case MetricKind::JaccardIndexMacro:
                    case MetricKind::ThreatScore:
                        value = jaccard_per_class[cls];
                        break;
                    case MetricKind::Markness:
                        value = markness_per_class[cls];
                        break;
                    case MetricKind::Matthews:
                        value = mcc_per_class[cls];
                        break;
                    case MetricKind::NegativeLikelihoodRatio:
                        value = neg_likelihood_per_class[cls];
                        break;
                    case MetricKind::NegativePredictiveValue:
                        value = npv_per_class[cls];
                        break;
                    case MetricKind::PositiveLikelihoodRatio:
                        value = pos_likelihood_per_class[cls];
                        break;
                    case MetricKind::PositivePredictiveValue:
                    case MetricKind::Precision:
                        value = precision_per_class[cls];
                        break;
                    case MetricKind::Prevalence:
                        value = prevalence_per_class[cls];
                        break;
                    case MetricKind::Recall:
                    case MetricKind::TruePositiveRate:
                        value = recall_per_class[cls];
                        break;
                    case MetricKind::Specificity:
                    case MetricKind::TrueNegativeRate:
                        value = specificity_per_class[cls];
                        break;
                    case MetricKind::Top1Error:
                        value = 1.0 - accuracy_per_class[cls];
                        break;
                    case MetricKind::LogLoss:
                        value = log_loss_per_class[cls];
                        break;
                    case MetricKind::BrierScore:
                        value = brier_per_class[cls];
                        break;
                    case MetricKind::ExpectedCalibrationError:
                        value = per_class_ece_values[cls];
                        break;
                    case MetricKind::MaximumCalibrationError:
                        value = per_class_mce_values[cls];
                        break;
                    case MetricKind::AUCROC:
                        value = auc_per_class[cls];
                        break;
                    case MetricKind::AUPRC:
                        value = auprc_per_class[cls];
                        break;
                    case MetricKind::AUPRG:
                        value = auprg_per_class[cls];
                        break;
                    case MetricKind::GiniCoefficient:
                        value = gini_per_class[cls];
                        break;
                    default:
                        break;
                }

                dest[cls] = value;
                if (std::isfinite(value)) {
                    macro_sum += value;
                    macro_count += 1.0;
                    weighted_sum += value * static_cast<double>(class_counts[cls].support);
                }
            }

            double macro = (macro_count > 0.0) ? (macro_sum / macro_count) : std::numeric_limits<double>::quiet_NaN();
            double weighted = (total_support > 0.0) ? (weighted_sum / total_support) : std::numeric_limits<double>::quiet_NaN();

            switch (kind) {
                case MetricKind::Top1Error:
                    macro = top1_error;
                    weighted = top1_error;
                    break;
                case MetricKind::Top3Accuracy:
                    macro = top3_accuracy;
                    weighted = top3_accuracy;
                    break;
                case MetricKind::Top3Error:
                    macro = top3_error;
                    weighted = top3_error;
                    break;
                case MetricKind::Top5Accuracy:
                    macro = top5_accuracy;
                    weighted = top5_accuracy;
                    break;
                case MetricKind::Top5Error:
                    macro = top5_error;
                    weighted = top5_error;
                    break;
                case MetricKind::HammingLoss:
                    macro = hamming_global;
                    weighted = hamming_global;
                    break;
                case MetricKind::JaccardIndexMicro:
                    macro = jaccard_micro;
                    weighted = jaccard_micro;
                    break;
                case MetricKind::ExpectedCalibrationError:
                    macro = global_ece;
                    weighted = global_ece;
                    break;
                case MetricKind::MaximumCalibrationError:
                    macro = global_mce;
                    weighted = global_mce;
                    break;
                case MetricKind::HausdorffDistance:
                    macro = hausdorff_mean;
                    weighted = hausdorff_mean;
                    break;
                case MetricKind::BoundaryIoU:
                    macro = boundary_iou_mean;
                    weighted = boundary_iou_mean;
                    break;
                case MetricKind::CalibrationSlope:
                    macro = cal_slope;
                    weighted = cal_slope;
                    break;
                case MetricKind::CalibrationIntercept:
                    macro = cal_intercept;
                    weighted = cal_intercept;
                    break;
                case MetricKind::HosmerLemeshowPValue:
                    macro = hosmer_p_value;
                    weighted = hosmer_p_value;
                    break;
                case MetricKind::KolmogorovSmirnovStatistic:
                    macro = ks_stat;
                    weighted = ks_stat;
                    break;
                case MetricKind::CohensKappa:
                    macro = global_kappa;
                    weighted = global_kappa;
                    break;
                case MetricKind::ConfusionEntropy:
                    macro = confusion_entropy_global;
                    weighted = confusion_entropy_global;
                    break;
                case MetricKind::CoverageError:
                    macro = coverage_error;
                    weighted = coverage_error;
                    break;
                case MetricKind::LabelRankingAveragePrecision:
                    macro = lrap;
                    weighted = lrap;
                    break;
                case MetricKind::LogLoss:
                    macro = log_loss_global;
                    weighted = log_loss_global;
                    break;
                case MetricKind::BrierScore:
                    macro = brier_global;
                    weighted = brier_global;
                    break;
                case MetricKind::BrierSkillScore:
                    macro = brier_skill_global;
                    weighted = brier_skill_global;
                    break;
                case MetricKind::Matthews:
                    macro = mcc_global;
                    weighted = mcc_global;
                    break;
                default:
                    break;
            }

            summary.push_back(Report::SummaryRow{kind, macro, weighted});
        }

        report.summary = summary;
        report.per_class = per_class_values;
        report.labels = class_labels;
        report.support = support_values;
        report.overall_accuracy = top1_accuracy;

        if (was_training) {
            model.train();
        } else {
            model.eval();
        }

        if (options.stream && (options.print_summary || options.print_per_class)) {
            Print(report, options);
        }

        return report;
    }

    inline void Print(const Report& report, const Options& options) {
        if (!options.stream) {
            return;
        }

        auto& stream = *options.stream;
        const auto& metrics = report.order;
        const auto metric_count = metrics.size();

        using namespace Utils::Terminal;
        const auto color = Colors::kBrightBlue;

        auto make_spacing = [](std::size_t width) { return width + 2; };

        if (options.print_summary && !report.summary.empty()) {
            std::vector<std::string> metric_names;
            std::vector<std::string> macro_values;
            std::vector<std::string> weighted_values;
            metric_names.reserve(report.summary.size());
            macro_values.reserve(report.summary.size());
            weighted_values.reserve(report.summary.size());

            std::size_t metric_width = std::string("Metric").size();
            std::size_t macro_width = std::string("Macro").size();
            std::size_t weighted_width = std::string("Weighted (support)").size();

            for (const auto& row : report.summary) {
                auto name = std::string(detail::metric_name(row.metric));
                auto macro = detail::format_double(row.macro);
                auto weighted = detail::format_double(row.weighted);
                metric_width = std::max(metric_width, name.size());
                macro_width = std::max(macro_width, macro.size());
                weighted_width = std::max(weighted_width, weighted.size());
                metric_names.push_back(std::move(name));
                macro_values.push_back(std::move(macro));
                weighted_values.push_back(std::move(weighted));
            }

            const std::vector<std::size_t> spacings{
                make_spacing(metric_width),
                make_spacing(macro_width),
                make_spacing(weighted_width)
            };

            const auto top = HTop(spacings, color, options.frame_style);
            const auto mid = HMid(spacings, color);
            const auto bottom = HBottom(spacings, color, options.frame_style);

            auto print_row = [&](std::string_view c1, std::string_view c2, std::string_view c3) {
                std::ostringstream row;
                row << std::setfill(' ');
                row << Symbols::kBoxVertical << ' ';
                row << std::left << std::setw(static_cast<int>(metric_width)) << c1;
                row << std::right;
                row << ' ' << Symbols::kBoxVertical << ' ';
                row << std::setw(static_cast<int>(macro_width)) << c2;
                row << ' ' << Symbols::kBoxVertical << ' ';
                row << std::setw(static_cast<int>(weighted_width)) << c3;
                row << ' ' << Symbols::kBoxVertical;
                stream << row.str() << '\n';
            };

            stream << '\n' << top << '\n';
            print_row("Evaluation: Classification", "", "");
            stream << mid << '\n';
            print_row("Metric", "Macro", "Weighted (support)");
            stream << mid << '\n';
            for (std::size_t i = 0; i < report.summary.size(); ++i) {
                print_row(metric_names[i], macro_values[i], weighted_values[i]);
            }
            stream << bottom << '\n';


            if (std::isfinite(report.overall_accuracy)) {
                stream << "\nOverall accuracy (micro): "
                       << detail::format_double(report.overall_accuracy) << '\n';
            }
        }

        if (options.print_per_class && !report.per_class.empty() && !report.labels.empty()) {
            const auto num_classes = report.labels.size();

            std::size_t metric_width = std::string("Metric").size();
            metric_width = std::max(metric_width, std::string("Support").size());
            for (auto kind : metrics) {
                metric_width = std::max(metric_width, std::string(detail::metric_name(kind)).size());
            }

            std::vector<std::string> headers;
            std::vector<std::size_t> class_widths(num_classes, 0);
            headers.reserve(num_classes);
            for (std::size_t i = 0; i < num_classes; ++i) {
                auto label = std::string("Label ") + std::to_string(report.labels[i]);
                class_widths[i] = std::max(class_widths[i], label.size());
                headers.push_back(std::move(label));
            }

            std::vector<std::string> support_values_strings;
            support_values_strings.reserve(num_classes);
            for (std::size_t i = 0; i < num_classes; ++i) {
                auto value = detail::format_size(report.support[i]);
                class_widths[i] = std::max(class_widths[i], value.size());
                support_values_strings.push_back(std::move(value));
            }

            for (std::size_t metric_index = 0; metric_index < metric_count && metric_index < report.per_class.size(); ++metric_index) {
                for (std::size_t cls = 0; cls < num_classes; ++cls) {
                    auto value = detail::format_double(report.per_class[metric_index][cls]);
                    class_widths[cls] = std::max(class_widths[cls], value.size());
                }
            }

            std::vector<std::size_t> spacings;
            spacings.reserve(num_classes + 1);
            spacings.push_back(make_spacing(metric_width));
            for (std::size_t i = 0; i < num_classes; ++i) {
                spacings.push_back(make_spacing(class_widths[i]));
            }

            const auto top = HTop(spacings, color, options.frame_style);
            const auto mid = HMid(spacings, color);
            const auto bottom = HBottom(spacings, color, options.frame_style);

            auto print_row = [&](std::string_view name, const std::vector<std::string>& values) {
                std::ostringstream row;
                row << std::setfill(' ');
                row << Symbols::kBoxVertical << ' ';
                row << std::left << std::setw(static_cast<int>(metric_width)) << name;
                row << std::right;
                for (std::size_t i = 0; i < num_classes; ++i) {
                    row << ' ' << Symbols::kBoxVertical << ' ';
                    row << std::setw(static_cast<int>(class_widths[i])) << values[i];
                }
                row << ' ' << Symbols::kBoxVertical;
                stream << row.str() << '\n';
            };

            stream << '\n' << top << '\n';
            print_row("Per-class", std::vector<std::string>(num_classes));
            stream << mid << '\n';
            print_row("Metric", headers);
            stream << mid << '\n';
            print_row("Support", support_values_strings);

            if (!report.per_class.empty()) {
                stream << mid << '\n';
                for (std::size_t metric_index = 0; metric_index < metric_count && metric_index < report.per_class.size(); ++metric_index) {
                    std::vector<std::string> values;
                    values.reserve(num_classes);
                    for (std::size_t cls = 0; cls < num_classes; ++cls) {
                        values.push_back(detail::format_double(report.per_class[metric_index][cls]));
                    }
                    print_row(std::string(detail::metric_name(metrics[metric_index])), values);
                    if (metric_index + 1 < metric_count && metric_index + 1 < report.per_class.size()) {
                        stream << mid << '\n';
                    }
                }
            }

            stream << bottom << '\n';
        }
    }
}
#endif //Nott_CLASSIFICATION_HPP