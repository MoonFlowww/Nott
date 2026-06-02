#ifndef Nott_RELIABILITY_PR_HPP
#define Nott_RELIABILITY_PR_HPP

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>
#include <cmath>
#include <torch/torch.h>

#include "../../../utils/terminal.hpp"
#include "reliability_curve_utils.hpp"
#include "../../../utils/gnuplot.hpp"
#include "../reliability.hpp"

namespace Nott {

    class Model;
}


namespace Nott::Plot::Details::Reliability {
    namespace detail {

        inline void RenderPRFromSeries(Model&, const Plot::Reliability::PRDescriptor& descriptor, std::vector<Curves::BinarySeries> series) {
            if (series.empty()) {
                throw std::invalid_argument("Precision-Recall rendering requires at least one series.");
            }

            for (std::size_t index = 0; index < series.size(); ++index) {
                if (series[index].label.empty()) {
                    series[index].label = Curves::DefaultLabelForIndex(index);
                }
            }

            const auto& options = descriptor.options;

            Utils::Gnuplot plotter("gnuplot", options.terminal);
            plotter.setTitle("Precision-Recall Curve");
            plotter.setXLabel("Recall");
            plotter.setYLabel("Precision");
            plotter.setGrid(true);
            plotter.setKey("top right");
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


            std::vector<Utils::Gnuplot::DataSet2D> datasets;
            datasets.reserve(series.size());

            const auto clampForLog = [adjustScale](double value) {
                if (!adjustScale) {
                    return value;
                }
                constexpr double epsilon = 1e-6;
                return value <= epsilon ? epsilon : value;
            };
            const auto clampForLogOneMinus = [adjustScale](double value) {
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
            for (std::size_t index = 0; index < series.size(); ++index) {
                const auto& entry = series[index];
                const auto curve = Curves::BuildCurve(entry);

                std::vector<double> recallValues;
                std::vector<double> precisionValues;
                std::vector<double> rawRecallValues;
                recallValues.reserve(curve.points.size());
                precisionValues.reserve(curve.points.size());
                rawRecallValues.reserve(curve.points.size());

                for (const auto& point : curve.points) {
                    const double tp = static_cast<double>(point.truePositives);
                    const double fp = static_cast<double>(point.falsePositives);
                    const double recallRaw = tp / static_cast<double>(curve.totalPositives);
                    const double precisionDenominator = tp + fp;
                    const double precisionRaw = precisionDenominator > 0.0 ? tp / precisionDenominator : 1.0;

                    rawRecallValues.push_back(recallRaw);

                    recallValues.push_back(clampForLog(recallRaw));
                    precisionValues.push_back(clampForLogOneMinus(precisionRaw));
                }

                if (recallValues.empty()) {
                    continue;
                }

                if (!rawRecallValues.empty() && rawRecallValues.front() > 0.0) {
                    recallValues.insert(recallValues.begin(), clampForLog(0.0));
                    precisionValues.insert(precisionValues.begin(), clampForLogOneMinus(1.0));
                }
                if (!rawRecallValues.empty() && rawRecallValues.back() < 1.0) {
                    recallValues.push_back(clampForLog(1.0));
                    precisionValues.push_back(clampForLogOneMinus(0.0));
                }

                Utils::Gnuplot::PlotStyle style{};
                style.mode = Utils::Gnuplot::PlotMode::LinesPoints;
                style.lineWidth = 2.0;
                style.pointType = 7;
                style.pointSize = 1.1;
                style.lineColor = detail::PickColor(index);

                datasets.push_back(Utils::Gnuplot::DataSet2D{
                    std::move(recallValues),
                    std::move(precisionValues),
                    entry.label,
                    style});
            }

            if (datasets.empty()) {
                throw std::runtime_error("Precision-Recall rendering could not assemble any datasets.");
            }

            plotter.plot(std::move(datasets));
        }
    } // namespace detail

    inline void RenderPR(Model& model,
                          const Plot::Reliability::PRDescriptor& descriptor,
                          std::vector<Curves::BinarySeries> series)
    {
        detail::RenderPRFromSeries(model, descriptor, std::move(series));
    }

    inline void RenderPR(Model& model, const Plot::Reliability::PRDescriptor& descriptor, torch::Tensor inputs, torch::Tensor targets)
    {
        std::vector<Curves::BinarySeries> series;
        series.emplace_back(Curves::MakeSeriesFromSamples(model, std::move(inputs), std::move(targets), ""));
        RenderPR(model, descriptor, std::move(series));
    }

    inline void RenderPR(Model& model, const Plot::Reliability::PRDescriptor& descriptor, torch::Tensor trainInputs, torch::Tensor trainTargets, torch::Tensor testInputs, torch::Tensor testTargets)
    {
        auto series = Curves::MakeSeriesFromSamples(model,
                                                    std::move(trainInputs),
                                                    std::move(trainTargets),
                                                    std::move(testInputs),
                                                    std::move(testTargets),
                                                    "Train",
                                                    "Test");
        RenderPR(model, descriptor, std::move(series));
    }

    inline void RenderPR(Model& model,
                          const Plot::Reliability::PRDescriptor& descriptor,
                          const std::vector<double>& probabilities,
                          const std::vector<int64_t>& targets)
    {
        std::vector<Curves::BinarySeries> series;
        series.emplace_back(Curves::MakeSeriesFromVectors(probabilities, targets, ""));
        RenderPR(model, descriptor, std::move(series));
    }

    inline void RenderPR(Model& model,
                          const Plot::Reliability::PRDescriptor& descriptor,
                          const std::vector<double>& trainProbabilities,
                          const std::vector<int64_t>& trainTargets,
                          const std::vector<double>& testProbabilities,
                          const std::vector<int64_t>& testTargets)
    {
        std::vector<Curves::BinarySeries> series;
        series.emplace_back(Curves::MakeSeriesFromVectors(trainProbabilities, trainTargets, "Train"));
        series.emplace_back(Curves::MakeSeriesFromVectors(testProbabilities, testTargets, "Test"));
        RenderPR(model, descriptor, std::move(series));
    }
}


#endif //Nott_RELIABILITY_PR_HPP