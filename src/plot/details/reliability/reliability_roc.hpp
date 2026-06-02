#ifndef Nott_RELIABILITY_ROC_HPP
#define Nott_RELIABILITY_ROC_HPP

#include <array>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>
#include <cmath>

#include <torch/torch.h>

#include "reliability_curve_utils.hpp"
#include "../../../utils/gnuplot.hpp"
#include "../../../utils/terminal.hpp"
#include "../reliability.hpp"

namespace Nott {
    class Model;
}


namespace Nott::Plot::Details::Reliability {
    namespace detail {


        inline void RenderROCFromSeries(Model& /*model*/,
                                        const Plot::Reliability::ROCDescriptor& descriptor,
                                        std::vector<Curves::BinarySeries> series)
        {
            if (series.empty()) {
                throw std::invalid_argument("ROC rendering requires at least one series.");
            }

            for (std::size_t index = 0; index < series.size(); ++index) {
                if (series[index].label.empty()) {
                    series[index].label = Curves::DefaultLabelForIndex(index);
                }
            }

            const auto& options = descriptor.options;

            Utils::Gnuplot plotter("gnuplot", options.terminal);
            plotter.setTitle("Receiver Operating Characteristic");
            plotter.setXLabel("False Positive Rate");
            plotter.setYLabel("True Positive Rate");

            plotter.setGrid(true);
            plotter.setKey("bottom right");

            const bool adjustScale = options.adjustScale;
            constexpr double logEpsilon = 1e-6;
            constexpr double logBase = 2.0;

            if (adjustScale) {

                plotter.setRange('x', logEpsilon, 1.0);
                plotter.setRange('y', logEpsilon, 1.0 - logEpsilon);
                plotter.setAxisScale('x', Utils::Gnuplot::AxisScale::Log);
                plotter.setAxisScale('y', Utils::Gnuplot::AxisScale::LogOneMinus);
            } else {
                plotter.setRange('x', 0.0, 1.0);
                plotter.setRange('y', 0.0, 1.0);
            }

            const auto transformFpr = [adjustScale](double value) {
                if (!adjustScale) {
                    return value;
                }
                constexpr double epsilon = 1e-6;
                return value <= 0.0 ? epsilon : value;
            };

            const auto transformTpr = [adjustScale](double value) {
                if (!adjustScale) {
                    return value;
                }
                constexpr double epsilon = 1e-6;
                if (value <= epsilon) {
                    return epsilon;
                }
                const double upperLimit = 1.0 - epsilon;
                return value >= upperLimit ? upperLimit : value;
            };

            std::vector<Utils::Gnuplot::DataSet2D> datasets;
            datasets.reserve(series.size());

            for (std::size_t index = 0; index < series.size(); ++index) {
                const auto& entry = series[index];
                const auto curve = Curves::BuildCurve(entry);

                std::vector<double> falsePositiveRates;
                std::vector<double> truePositiveRates;
                falsePositiveRates.reserve(curve.points.size());
                truePositiveRates.reserve(curve.points.size());

                for (const auto& point : curve.points) {
                    const double fpr = static_cast<double>(point.falsePositives)
                        / static_cast<double>(curve.totalNegatives);
                    const double tpr = static_cast<double>(point.truePositives)
                        / static_cast<double>(curve.totalPositives);
                    falsePositiveRates.push_back(transformFpr(fpr));
                    truePositiveRates.push_back(transformTpr(tpr));
                }

                Utils::Gnuplot::PlotStyle style{};
                style.mode = Utils::Gnuplot::PlotMode::LinesPoints;
                style.lineWidth = 2.0;
                style.pointType = 7;
                style.pointSize = 1.1;
                style.lineColor = detail::PickColor(index);

                datasets.push_back(Utils::Gnuplot::DataSet2D{
                    std::move(falsePositiveRates),
                    std::move(truePositiveRates),
                    entry.label,
                    style});
            }

            plotter.plot(std::move(datasets));
        }
    } // namespace detail

    inline void RenderROC(Model& model,
                           const Plot::Reliability::ROCDescriptor& descriptor,
                           std::vector<Curves::BinarySeries> series)
    {
        detail::RenderROCFromSeries(model, descriptor, std::move(series));
    }

    inline void RenderROC(Model& model, const Plot::Reliability::ROCDescriptor& descriptor, torch::Tensor inputs, torch::Tensor targets) {
        std::vector<Curves::BinarySeries> series;
        series.emplace_back(Curves::MakeSeriesFromSamples(model, std::move(inputs), std::move(targets), ""));
        RenderROC(model, descriptor, std::move(series));
    }

    inline void RenderROC(Model& model, const Plot::Reliability::ROCDescriptor& descriptor, torch::Tensor trainInputs, torch::Tensor trainTargets, torch::Tensor testInputs, torch::Tensor testTargets)
    {
        auto series = Curves::MakeSeriesFromSamples(model,
                                                    std::move(trainInputs),
                                                    std::move(trainTargets),
                                                    std::move(testInputs),
                                                    std::move(testTargets),
                                                    "Train", "Test");
        RenderROC(model, descriptor, std::move(series));
    }

    inline void RenderROC(Model& model,
                           const Plot::Reliability::ROCDescriptor& descriptor,
                           const std::vector<double>& probabilities,
                           const std::vector<int64_t>& targets)
    {
        std::vector<Curves::BinarySeries> series;
        series.emplace_back(Curves::MakeSeriesFromVectors(probabilities, targets, ""));
        RenderROC(model, descriptor, std::move(series));
    }

    inline void RenderROC(Model& model,
                           const Plot::Reliability::ROCDescriptor& descriptor,
                           const std::vector<double>& trainProbabilities,
                           const std::vector<int64_t>& trainTargets,
                           const std::vector<double>& testProbabilities,
                           const std::vector<int64_t>& testTargets)
    {
        std::vector<Curves::BinarySeries> series;
        series.emplace_back(Curves::MakeSeriesFromVectors(trainProbabilities, trainTargets, "Train"));
        series.emplace_back(Curves::MakeSeriesFromVectors(testProbabilities, testTargets, "Test"));
        RenderROC(model, descriptor, std::move(series));
    }
}


#endif //Nott_RELIABILITY_ROC_HPP