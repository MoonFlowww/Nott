#ifndef Nott_CALIBRATION_HPP
#define Nott_CALIBRATION_HPP
/// This file is an factory, must exempt it from any logical-code. For functions look into "/details"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <iomanip>
#include <memory>
#include <optional>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include <torch/torch.h>

#include "../utils/gnuplot.hpp"


namespace Nott::Calibration {
    template <class...>
    inline constexpr bool always_false_v = false;

    struct TemperatureScalingDescriptor {
        std::size_t max_iterations{50};
        double learning_rate{0.01};
    };

    using Descriptor = std::variant<TemperatureScalingDescriptor>;

    class Method {
    public:
        virtual ~Method() = default;

        virtual void attach(torch::nn::Module& model, const torch::Device& device) = 0;
        virtual void fit(const torch::Tensor& logits, const torch::Tensor& targets) = 0;
        virtual void plot(std::ostream& stream) const = 0;
        [[nodiscard]] virtual torch::Tensor transform(torch::Tensor logits) const = 0;
    };

    using MethodPtr = std::shared_ptr<Method>;

    struct Options {
        std::size_t reliability_bins{15};
        std::ostream* stream{nullptr};
        std::optional<std::size_t> forward_chunk_size{};
        std::size_t forward_buffer_batches{2};
    };


    namespace Details {
        struct ReliabilityBin {
            double confidence_sum{0.0};
            double accuracy_sum{0.0};
            std::size_t count{0};
        };

        struct ReliabilityComputation {
            std::vector<ReliabilityBin> bins{};
            double log_loss{std::numeric_limits<double>::quiet_NaN()};
            double auc{std::numeric_limits<double>::quiet_NaN()};
        };

        inline double compute_auc(const std::vector<double>& scores, const std::vector<int>& labels)
        {
            const auto total = labels.size();
            if (total == 0 || scores.size() != total) {
                return std::numeric_limits<double>::quiet_NaN();
            }

            std::size_t positives = 0;
            for (int label : labels) {
                positives += (label == 1) ? 1 : 0;
            }
            const std::size_t negatives = total - positives;
            if (positives == 0 || negatives == 0) {
                return std::numeric_limits<double>::quiet_NaN();
            }

            std::vector<std::pair<double, int>> pairs(total);
            for (std::size_t i = 0; i < total; ++i) {
                pairs[i] = {scores[i], labels[i]};
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

        inline std::size_t clamp_bin_index(double confidence, std::size_t bins)
        {
            if (bins == 0) {
                return 0;
            }

            const double clamped = std::clamp(confidence, 0.0, 1.0);
            auto index = static_cast<std::size_t>(clamped * static_cast<double>(bins));
            if (index >= bins) {
                index = bins - 1;
            }
            return index;
        }

        inline torch::Tensor normalise_targets(torch::Tensor targets, std::int64_t num_classes)
        {
            if (!targets.defined()) {
                throw std::invalid_argument("Calibration requires defined target tensor.");
            }

            auto prepared = targets.detach();
            if (prepared.device().type() != torch::kCPU) {
                prepared = prepared.to(torch::kCPU);
            }

            if (prepared.dim() > 1) {
                const auto last_dim = prepared.dim() - 1;
                if (prepared.size(last_dim) == num_classes) {
                    prepared = prepared.argmax(last_dim);
                } else if (prepared.size(last_dim) == 1) {
                    prepared = prepared.squeeze(last_dim);
                } else {
                    prepared = prepared.reshape({prepared.size(0)});
                }
            }

            if (prepared.dtype() != torch::kLong) {
                prepared = prepared.to(torch::kLong);
            }

            if (prepared.dim() != 1) {
                throw std::runtime_error("Calibration targets must be reducible to a one-dimensional tensor.");
            }

            const auto min_val = prepared.min().item<long>();
            const auto max_val = prepared.max().item<long>();
            if (min_val < 0 || max_val >= num_classes) {
                std::ostringstream oss;
                oss << "Targets out of range [0," << (num_classes - 1)
                    << "]: min=" << min_val << ", max=" << max_val << ".";
                throw std::runtime_error(oss.str());
            }


            return prepared.contiguous();
        }

        inline ReliabilityComputation compute_reliability(torch::Tensor logits, torch::Tensor targets, std::size_t bin_count) {
            if (!logits.defined()) {
                throw std::invalid_argument("Calibration requires defined logits tensor.");
            }

            if (logits.dim() == 1) {
                logits = logits.unsqueeze(1);
            } else if (logits.dim() > 2) {
                logits = logits.reshape({logits.size(0), -1});
            }

            const auto classes = logits.size(1);
            auto prepared_targets = normalise_targets(std::move(targets), classes);
            if (prepared_targets.size(0) != logits.size(0)) {
                throw std::runtime_error("Calibration logits and targets must share the same number of samples.");
            }

            auto logits_cpu = logits.detach().to(torch::kCPU, torch::kFloat64).contiguous();
            auto probabilities = torch::softmax(logits_cpu, 1).contiguous();
            auto max_result = probabilities.max(1);
            auto confidences = std::get<0>(max_result).to(torch::kCPU).contiguous();
            auto predictions = std::get<1>(max_result).to(torch::kCPU, torch::kLong).contiguous();

            const auto total = confidences.size(0);
            ReliabilityComputation result{};
            if (total == 0) {
                result.bins.resize(std::max<std::size_t>(1, bin_count));
                return result;
            }
            const auto* confidence_ptr = confidences.data_ptr<double>();
            const auto* prediction_ptr = predictions.data_ptr<long>();
            const auto* target_ptr = prepared_targets.data_ptr<long>();
            const auto* probability_ptr = probabilities.data_ptr<double>();

            const std::size_t stride = static_cast<std::size_t>(probabilities.size(1));

            const std::size_t bins = std::max<std::size_t>(1, bin_count);
            result.bins.assign(bins, ReliabilityBin{});

            double log_loss_sum{0.0};
            std::vector<double> auc_scores{};
            std::vector<int> auc_labels{};
            auc_scores.reserve(classes == 2 ? static_cast<std::size_t>(total) : 0);
            auc_labels.reserve(auc_scores.capacity());


            for (std::int64_t idx = 0; idx < total; ++idx) {
                const double confidence = confidence_ptr[idx];
                const std::size_t bin_index = clamp_bin_index(confidence, bins);
                auto& entry = result.bins[bin_index];
                entry.count += 1;
                entry.confidence_sum += confidence;
                entry.accuracy_sum += (prediction_ptr[idx] == target_ptr[idx]) ? 1.0 : 0.0;


                const auto target_index = static_cast<std::size_t>(target_ptr[idx]);
                if (target_index >= stride) {
                    std::ostringstream oss;
                    oss << "Target index " << target_index << " >= num_classes " << stride
                        << " at sample " << idx << ".";
                    throw std::runtime_error(oss.str());
                }
                const auto base_index = static_cast<std::size_t>(idx) * stride;
                const auto true_index = base_index + target_index;
                const double true_probability = std::clamp(probability_ptr[true_index], 1e-12, 1.0);
                log_loss_sum += -std::log(true_probability);

                if (classes == 2) {
                    const double positive_probability = std::clamp(probability_ptr[base_index + 1], 0.0, 1.0);
                    auc_scores.push_back(positive_probability);
                    auc_labels.push_back(target_ptr[idx] == 1 ? 1 : 0);
                }
            }
            auto compute_auc_ovr = [&](const torch::Tensor& probs, const torch::Tensor& y, int C)->double {
                std::vector<double> aucs;
                aucs.reserve(C);
                auto y_cpu = y.to(torch::kCPU, torch::kLong).contiguous();
                auto p_cpu = probs.to(torch::kCPU, torch::kFloat64).contiguous();
                const double* P = p_cpu.data_ptr<double>();
                const long*   Y = y_cpu.data_ptr<long>();
                const int64_t N = y_cpu.size(0), stride = p_cpu.size(1);
                for (int c = 0; c < C; ++c) {
                    std::vector<double> score; score.reserve(N);
                    std::vector<int>    lab;   lab.reserve(N);
                    for (int64_t i = 0; i < N; ++i) {
                        score.push_back(std::clamp(P[i*stride + c], 0.0, 1.0));
                        lab.push_back(Y[i] == c ? 1 : 0);
                    }
                    aucs.push_back(compute_auc(score, lab));
                }
                double s = 0.0; int k = 0;
                for (auto a : aucs) if (std::isfinite(a)) { s += a; ++k; }
                return k ? s / k : std::numeric_limits<double>::quiet_NaN();
            };

            result.log_loss = log_loss_sum / static_cast<double>(total);
            if (classes == 2) {
                result.auc = compute_auc(auc_scores, auc_labels);
            }

            return result;
        }


        inline void plot_reliability_diagram(const Method& method,
                                             const std::pair<torch::Tensor, torch::Tensor>& validation,
                                             const Options& options,
                                             const torch::Device& device) {
            auto logits = validation.first;
            auto targets = validation.second;

            if (!logits.defined() || !targets.defined()) {
                throw std::invalid_argument("Validation logits and targets must be defined for reliability plotting.");
            }
            auto raw_computation = compute_reliability(logits, targets.clone(), options.reliability_bins);

            auto device_logits = logits;
            if (device_logits.device() != device) {
                device_logits = device_logits.to(device);
            }

            torch::NoGradGuard guard;
            auto transformed = method.transform(std::move(device_logits)).detach().to(torch::kCPU);

            auto calibrated_computation = compute_reliability(std::move(transformed), std::move(targets), options.reliability_bins);


            struct ReliabilityStats {
                std::vector<double> confidence_points{};
                std::vector<double> accuracy_points{};
                double ece{0.0};
                double mce{0.0};
                std::size_t total_samples{0};

                [[nodiscard]] bool has_points() const
                {
                    return !confidence_points.empty();
                }
            };

            const auto compute_stats = [](const std::vector<ReliabilityBin>& bins) {
                ReliabilityStats stats{};

                for (const auto& bin : bins) {
                    stats.total_samples += bin.count;
                }

                if (stats.total_samples == 0) {
                    return stats;
                }

                stats.confidence_points.reserve(bins.size());
                stats.accuracy_points.reserve(bins.size());

                for (const auto& bin : bins) {
                    if (bin.count == 0) {
                        continue;
                    }

                    const double avg_confidence = bin.confidence_sum / static_cast<double>(bin.count);
                    const double avg_accuracy = bin.accuracy_sum / static_cast<double>(bin.count);
                    stats.confidence_points.push_back(avg_confidence);
                    stats.accuracy_points.push_back(avg_accuracy);


                    const double difference = std::abs(avg_confidence - avg_accuracy);
                    stats.ece += difference * (static_cast<double>(bin.count) / static_cast<double>(stats.total_samples));
                    stats.mce = std::max(stats.mce, difference);
                }

                return stats;
            };

            auto raw_stats = compute_stats(raw_computation.bins);
            auto calibrated_stats = compute_stats(calibrated_computation.bins);

            if (calibrated_stats.total_samples == 0) {
                if (options.stream != nullptr) {
                    *options.stream << "Calibration: no validation samples available for reliability plotting." << '\n';
                }
                return;
            }

            const auto format_metric = [](double value) {
                if (!std::isfinite(value)) {
                    return std::string("N/A");
                }
                std::ostringstream formatted;
                formatted << std::fixed << std::setprecision(6) << value;
                return formatted.str();
            };

            const auto format_curve_title = [&format_metric](const std::string& base,
                                            double log_loss,
                                            double auc) {
                std::ostringstream title;
                title << base << " (LogLoss: " << format_metric(log_loss)
                      << ", AUC: " << format_metric(auc) << ')';
                return title.str();
            };


            if (options.stream != nullptr) {
                auto& out = *options.stream;
                const auto previous_flags = out.flags();
                const auto previous_precision = out.precision();


                out << std::fixed << std::setprecision(6)
                << "Calibration ECE: " << calibrated_stats.ece
                << ", MCE: " << calibrated_stats.mce << '\n';
                out << "Calibrated LogLoss: " << format_metric(calibrated_computation.log_loss)
                    << ", AUC: " << format_metric(calibrated_computation.auc) << '\n';
                if (raw_stats.total_samples != 0 && raw_stats.has_points()) {
                    out << "Uncalibrated ECE: " << raw_stats.ece
                        << ", MCE: " << raw_stats.mce << '\n';
                    out << "Uncalibrated LogLoss: " << format_metric(raw_computation.log_loss)
                        << ", AUC: " << format_metric(raw_computation.auc) << '\n';
                }
                out.flags(previous_flags);
                out.precision(previous_precision);
            }

            if (!calibrated_stats.has_points()) {
                if (options.stream != nullptr) {
                    *options.stream << "Calibration: insufficient populated bins for reliability plot." << '\n';
                }
                return;
            }

            Utils::Gnuplot plotter{};
            plotter.setTitle("Reliability Diagram");
            plotter.setXLabel("Predicted Probability");
            plotter.setYLabel("Empirical Probability");
            plotter.command("set xrange [0:1]");
            plotter.command("set yrange [0:1]");
            plotter.command("set key top left");
            plotter.command("set grid");

            Utils::Gnuplot::PlotStyle diagonal_style{};
            diagonal_style.mode = Utils::Gnuplot::PlotMode::Lines;
            diagonal_style.lineWidth = 1.5;
            diagonal_style.lineColor = "#7f7f7f";

            Utils::Gnuplot::PlotStyle model_style{};
            model_style.mode = Utils::Gnuplot::PlotMode::LinesPoints;
            model_style.lineWidth = 2.0;
            model_style.pointType = 7;
            model_style.pointSize = 1.2;
            model_style.lineColor = "#1f77b4";

            Utils::Gnuplot::PlotStyle raw_model_style = model_style;
            raw_model_style.lineColor = "#d62728";
            raw_model_style.pointType = 5;


            Utils::Gnuplot::DataSet2D diagonal{
                std::vector<double>{0.0, 1.0},
                std::vector<double>{0.0, 1.0},
                "Perfect calibration",
                diagonal_style
            };
            std::vector<Utils::Gnuplot::DataSet2D> datasets;
            datasets.reserve(3);
            datasets.push_back(std::move(diagonal));

            if (raw_stats.has_points()) {
                auto raw_title = format_curve_title("Uncalibrated model", raw_computation.log_loss, raw_computation.auc);
                datasets.push_back(Utils::Gnuplot::DataSet2D{
                    std::move(raw_stats.confidence_points),
                    std::move(raw_stats.accuracy_points),
                    std::move(raw_title),
                    raw_model_style
                });
            }

            auto calibrated_title = format_curve_title("Calibrated model", calibrated_computation.log_loss, calibrated_computation.auc);

            Utils::Gnuplot::DataSet2D model_curve{
                std::move(calibrated_stats.confidence_points),
                std::move(calibrated_stats.accuracy_points),
                std::move(calibrated_title),
                model_style
            };
            datasets.push_back(std::move(model_curve));

            plotter.plot(std::move(datasets));
        }
    }

#include "details/temperature_scaling.hpp"



    inline MethodPtr make_method(const Descriptor& descriptor)
    {
        return std::visit(
            [](const auto& concrete_descriptor) -> MethodPtr {
                using DescriptorType = std::decay_t<decltype(concrete_descriptor)>;
                if constexpr (std::is_same_v<DescriptorType, TemperatureScalingDescriptor>) {
                    return Details::make_temperature_scaling_method(concrete_descriptor);
                } else {
                    static_assert(always_false_v<DescriptorType>, "Unsupported calibration descriptor");
                }
            },
            descriptor);
    }

    using CalibrationDataCallback = std::function<std::pair<torch::Tensor, torch::Tensor>(torch::nn::Module&)>;

    inline MethodPtr Calibrate(torch::nn::Module& model,
                        const torch::Device& device,
                        const Descriptor& descriptor,
                        CalibrationDataCallback data_callback,
                        std::optional<std::pair<torch::Tensor, torch::Tensor>> validation = std::nullopt,
                        Options options = {},
                        bool plot = false) {
        if (!data_callback) {
            throw std::invalid_argument("Calibration requires a valid logits callback.");
        }

        auto method = make_method(descriptor);
        method->attach(model, device);
        auto calibration_pair = data_callback(model);
        auto& logits = calibration_pair.first;
        auto& targets = calibration_pair.second;
        if (!logits.defined() || !targets.defined()) {
            throw std::invalid_argument("Calibration callback returned undefined logits or targets.");
        }

        method->fit(logits, targets);

        if (options.stream != nullptr) {
            *options.stream << "Calibration method: ";
            method->plot(*options.stream);
            *options.stream << '\n';
        }

        if (plot) {
            if (!validation.has_value()) {
                if (options.stream != nullptr) {
                    *options.stream << "Calibration: validation data required for reliability plot." << '\n';
                }
            } else {
                try {
                    Details::plot_reliability_diagram(*method, *validation, options, device);
                } catch (const std::exception& ex) {
                    if (options.stream != nullptr) {
                        *options.stream << "Calibration: failed to plot reliability diagram: " << ex.what() << '\n';
                    }
                }
            }
        }
        return method;
    }
}


#endif //Nott_CALIBRATION_HPP