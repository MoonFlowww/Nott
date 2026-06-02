#ifndef Nott_RELIABILITY_DET_HPP
#define Nott_RELIABILITY_DET_HPP


#include <array>
#include <cstddef>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

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


        inline void RenderDETFromSeries(Model& /*model*/,
                                        const Plot::Reliability::DETDescriptor& descriptor,
                                        std::vector<Curves::BinarySeries> series)
        {
            if (series.empty()) {
                throw std::invalid_argument("DET rendering requires at least one series.");
            }

            for (std::size_t index = 0; index < series.size(); ++index) {
                if (series[index].label.empty()) {
                    series[index].label = Curves::DefaultLabelForIndex(index);
                }
            }

            const auto& options = descriptor.options;

            Utils::Gnuplot plotter("gnuplot", options.terminal);
            plotter.setTitle("Detection Error Tradeoff");
            plotter.setXLabel("False Positive Rate");
            plotter.setYLabel("False Negative Rate");

            plotter.setGrid(true);
            plotter.setKey("top right");

            const bool adjustScale = options.adjustScale;

            constexpr double logEpsilon = 1e-6;

            if (adjustScale) {
                plotter.setRange('x', logEpsilon, 1.0);
                plotter.setRange('y', logEpsilon, 1.0);
                plotter.setAxisScale('x', Utils::Gnuplot::AxisScale::Log);
                plotter.setAxisScale('y', Utils::Gnuplot::AxisScale::Log);

            } else {
                plotter.setRange('x', 0.0, 1.0);
                plotter.setRange('y', 0.0, 1.0);
            }



            std::vector<Utils::Gnuplot::DataSet2D> datasets;
            datasets.reserve(series.size());

            for (std::size_t index = 0; index < series.size(); ++index) {
                const auto& entry = series[index];
                const auto curve = Curves::BuildCurve(entry);

                std::vector<double> falsePositiveRates;
                std::vector<double> falseNegativeRates;
                falsePositiveRates.reserve(curve.points.size());
                falseNegativeRates.reserve(curve.points.size());

                for (const auto& point : curve.points) {
                    double fpr = static_cast<double>(point.falsePositives) / static_cast<double>(curve.totalNegatives);
                    double fnr = static_cast<double>(point.falseNegatives) / static_cast<double>(curve.totalPositives);

                    if (adjustScale) {
                        fpr = fpr <= 0.0 ? logEpsilon : fpr;
                        fnr = fnr <= 0.0 ? logEpsilon : fnr;
                    }


                    falsePositiveRates.push_back(fpr);
                    falseNegativeRates.push_back(fnr);
                }

                Utils::Gnuplot::PlotStyle style{};
                style.mode = Utils::Gnuplot::PlotMode::LinesPoints;
                style.lineWidth = 2.0;
                style.pointType = 7;
                style.pointSize = 1.1;
                style.lineColor = detail::PickColor(index);

                datasets.push_back(Utils::Gnuplot::DataSet2D{
                    std::move(falsePositiveRates),
                    std::move(falseNegativeRates),
                    entry.label,
                    style});
            }

            plotter.plot(std::move(datasets));
        }
    } // namespace detail

    inline void RenderDET(Model& model, const Plot::Reliability::DETDescriptor& descriptor, std::vector<Curves::BinarySeries> series) {
        detail::RenderDETFromSeries(model, descriptor, std::move(series));
    }

    inline void RenderDET(Model& model, const Plot::Reliability::DETDescriptor& descriptor, torch::Tensor inputs, torch::Tensor targets) {
        std::vector<Curves::BinarySeries> series;
        series.emplace_back(Curves::MakeSeriesFromSamples(model, std::move(inputs), std::move(targets), ""));
        RenderDET(model, descriptor, std::move(series));
    }

    inline void RenderDET(Model& model, const Plot::Reliability::DETDescriptor& descriptor, torch::Tensor trainInputs, torch::Tensor trainTargets, torch::Tensor testInputs, torch::Tensor testTargets) {
        auto series = Curves::MakeSeriesFromSamples(model,
                                                    std::move(trainInputs),
                                                    std::move(trainTargets),
                                                    std::move(testInputs),
                                                    std::move(testTargets),
                                                    "Train",
                                                    "Test");
        RenderDET(model, descriptor, std::move(series));
    }

    inline void RenderDET(Model& model, const Plot::Reliability::DETDescriptor& descriptor, const std::vector<double>& probabilities, const std::vector<int64_t>& targets) {
        std::vector<Curves::BinarySeries> series;
        series.emplace_back(Curves::MakeSeriesFromVectors(probabilities, targets, ""));
        RenderDET(model, descriptor, std::move(series));
    }

    inline void RenderDET(Model& model, const Plot::Reliability::DETDescriptor& descriptor, const std::vector<double>& trainProbabilities, const std::vector<int64_t>& trainTargets, const std::vector<double>& testProbabilities, const std::vector<int64_t>& testTargets) {
        std::vector<Curves::BinarySeries> series;
        series.emplace_back(Curves::MakeSeriesFromVectors(trainProbabilities, trainTargets, "Train"));
        series.emplace_back(Curves::MakeSeriesFromVectors(testProbabilities, testTargets, "Test"));
        RenderDET(model, descriptor, std::move(series));
    }
}

#endif //Nott_RELIABILITY_DET_HPP