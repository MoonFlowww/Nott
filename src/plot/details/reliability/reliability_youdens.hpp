#ifndef Nott_RELIABILITY_YOUDENS_HPP
#define Nott_RELIABILITY_YOUDENS_HPP

#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include <torch/torch.h>

#include "reliability_curve_utils.hpp"
#include "../../../utils/gnuplot.hpp"
#include "../reliability.hpp"
#include "../../../utils/terminal.hpp"

namespace Nott {
    class Model;
}


namespace Nott::Plot::Details::Reliability {
    namespace detail {


        inline void RenderYoudensFromSeries(Model& /*model*/,
                                            const Plot::Reliability::YoudensDescriptor& descriptor,
                                            std::vector<Curves::BinarySeries> series)
        {
            if (series.empty()) {
                throw std::invalid_argument("Youden's J rendering requires at least one series.");
            }

            for (std::size_t index = 0; index < series.size(); ++index) {
                if (series[index].label.empty()) {
                    series[index].label = Curves::DefaultLabelForIndex(index);
                }
            }

            const auto& options = descriptor.options;

            Utils::Gnuplot plotter("gnuplot", options.terminal);
            plotter.setTitle("Youden's J Statistic");
            plotter.setXLabel("Decision Threshold");
            plotter.setYLabel("Youden's J");
            plotter.command("set xrange [0:1]");
            plotter.command("set yrange [-1:1]");
            plotter.setGrid(true);
            plotter.setKey("top left");


            std::vector<Utils::Gnuplot::DataSet2D> datasets;
            datasets.reserve(series.size());

            for (std::size_t index = 0; index < series.size(); ++index) {
                const auto& entry = series[index];
                const auto curve = Curves::BuildCurve(entry);

                std::vector<double> thresholds;
                std::vector<double> youdenValues;
                thresholds.reserve(curve.points.size());
                youdenValues.reserve(curve.points.size());

                for (const auto& point : curve.points) {
                    if (!std::isfinite(point.threshold)) {
                        continue;
                    }
                    const double fpr = static_cast<double>(point.falsePositives)
                        / static_cast<double>(curve.totalNegatives);
                    const double tpr = static_cast<double>(point.truePositives)
                        / static_cast<double>(curve.totalPositives);
                    thresholds.push_back(point.threshold);
                    youdenValues.push_back(tpr - fpr);
                }

                if (thresholds.empty()) {
                    continue;
                }

                Utils::Gnuplot::PlotStyle style{};
                style.mode = Utils::Gnuplot::PlotMode::LinesPoints;
                style.lineWidth = 2.0;
                style.pointType = 7;
                style.pointSize = 1.1;
                style.lineColor = detail::PickColor(index);

                datasets.push_back(Utils::Gnuplot::DataSet2D{
                    std::move(thresholds),
                    std::move(youdenValues),
                    entry.label,
                    style});
            }

            if (datasets.empty()) {
                throw std::runtime_error("Youden's J rendering could not assemble any datasets.");
            }

            plotter.plot(std::move(datasets));
        }
    } // namespace detail

    inline void RenderYoudens(Model& model,
                               const Plot::Reliability::YoudensDescriptor& descriptor,
                               std::vector<Curves::BinarySeries> series)
    {
        detail::RenderYoudensFromSeries(model, descriptor, std::move(series));
    }

    inline void RenderYoudens(Model& model,
                               const Plot::Reliability::YoudensDescriptor& descriptor,
                               torch::Tensor inputs,
                               torch::Tensor targets)
    {
        std::vector<Curves::BinarySeries> series;
        series.emplace_back(Curves::MakeSeriesFromSamples(model, std::move(inputs), std::move(targets), ""));

        RenderYoudens(model, descriptor, std::move(series));
    }

    inline void RenderYoudens(Model& model,
                               const Plot::Reliability::YoudensDescriptor& descriptor,
                               torch::Tensor trainInputs,
                               torch::Tensor trainTargets,
                               torch::Tensor testInputs,
                               torch::Tensor testTargets)
    {
        std::vector<Curves::BinarySeries> series;
        series.emplace_back(Curves::MakeSeriesFromTensor(std::move(trainInputs), std::move(trainTargets), "Train"));
        series.emplace_back(Curves::MakeSeriesFromTensor(std::move(testInputs), std::move(testTargets), "Test"));
        RenderYoudens(model, descriptor, std::move(series));
    }

    inline void RenderYoudens(Model& model,
                               const Plot::Reliability::YoudensDescriptor& descriptor,
                               const std::vector<double>& probabilities,
                               const std::vector<int64_t>& targets)
    {
        std::vector<Curves::BinarySeries> series;
        series.emplace_back(Curves::MakeSeriesFromVectors(probabilities, targets, ""));
        RenderYoudens(model, descriptor, std::move(series));
    }

    inline void RenderYoudens(Model& model,
                               const Plot::Reliability::YoudensDescriptor& descriptor,
                               const std::vector<double>& trainProbabilities,
                               const std::vector<int64_t>& trainTargets,
                               const std::vector<double>& testProbabilities,
                               const std::vector<int64_t>& testTargets)
    {
        std::vector<Curves::BinarySeries> series;
        series.emplace_back(Curves::MakeSeriesFromVectors(trainProbabilities, trainTargets, "Train"));
        series.emplace_back(Curves::MakeSeriesFromVectors(testProbabilities, testTargets, "Test"));
        RenderYoudens(model, descriptor, std::move(series));
    }
}

#endif //Nott_RELIABILITY_YOUDENS_HPP