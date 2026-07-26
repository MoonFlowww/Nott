#ifndef Nott_RELIABILITY_LIME_HPP
#define Nott_RELIABILITY_LIME_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <torch/torch.h>

#include "../../../core/core.hpp"
#include "../../../utils/gnuplot.hpp"
#include "reliability_curve_utils.hpp"

namespace Nott::Plot::Details::Reliability {
    namespace detail {
        inline auto ComputeLIMEWeights(torch::Tensor masks,
                                       torch::Tensor outputs,
                                       torch::Tensor distances) -> torch::Tensor
        {
            auto weights = torch::exp(-distances.pow(2) / 0.25).unsqueeze(1);
            auto sqrtW = weights.sqrt();
            auto Xw = masks * sqrtW;
            auto yw = outputs.unsqueeze(1) * sqrtW;

            auto Xt = Xw.transpose(0, 1);
            auto XtX = Xt.matmul(Xw);
            auto lambda = 1e-3;
            XtX += lambda * torch::eye(XtX.size(0), XtX.options());
            auto XtY = Xt.matmul(yw);
            auto solution = torch::linalg_solve(XtX, XtY).squeeze();
            return solution;
        }

        inline void RenderLIMEPlot(std::size_t sample_index, const std::vector<std::pair<std::size_t, double>>& importances, bool show_weights, const Utils::Gnuplot::TerminalOptions& terminalOptions) {
            Utils::Gnuplot plotter("gnuplot", terminalOptions);
            plotter.setTitle("LIME - sample " + std::to_string(sample_index));
            plotter.setXLabel("Feature");
            plotter.setYLabel("Contribution");
            plotter.setGrid(true);
            plotter.command("set style data boxes");
            plotter.command("set style fill solid 0.65 border -1");
            plotter.command("set boxwidth 0.8");
            plotter.unsetKey();

            std::vector<double> x(importances.size());
            std::vector<double> y(importances.size());
            std::vector<std::pair<double, std::string>> tics;

            for (std::size_t i = 0; i < importances.size(); ++i) {
                x[i] = static_cast<double>(i);
                y[i] = importances[i].second;
                tics.emplace_back(x[i], "f" + std::to_string(importances[i].first));
            }

            if (!tics.empty()) {
                plotter.setLabeledTics('x', tics);
                plotter.setRange('x', -0.5, static_cast<double>(tics.size()) - 0.5);
            }

            Utils::Gnuplot::PlotStyle style{};
            style.mode = Utils::Gnuplot::PlotMode::Boxes;
            style.lineColor = "#1f77b4";

            Utils::Gnuplot::DataSet2D dataset{std::move(x), std::move(y), {}, style};
            plotter.plot(dataset);

            if (show_weights) {
                std::ostringstream stream;
                stream << "[Nott] LIME top features for sample " << sample_index << ":\n";
                for (const auto& [feature, weight] : importances) {
                    stream << "  feature " << feature << ": " << std::fixed << std::setprecision(6)
                           << weight << '\n';
                }
                std::cout << stream.str();
            }
        }
    } // namespace detail

    inline void RenderLIME(Model& model,
                           const Plot::Reliability::LIMEDescriptor& descriptor,
                           torch::Tensor inputs,
                           torch::Tensor targets)
    {
        if (!inputs.defined()) {
            throw std::invalid_argument("LIME requires defined input tensors.");
        }
        if (inputs.dim() < 2) {
            throw std::invalid_argument("LIME expects inputs shaped as (batch, features...).");
        }

        const auto total_samples = static_cast<std::size_t>(inputs.size(0));
        if (total_samples == 0) {
            throw std::invalid_argument("LIME requires at least one sample.");
        }




        auto flattened_targets = Interpretability::FlattenTargets(std::move(targets), total_samples);
        auto selected_indices = Interpretability::SelectIndices(
            total_samples,
            descriptor.options.samples == 0
                ? std::min<std::size_t>(total_samples, static_cast<std::size_t>(8))
                : descriptor.options.samples,
            descriptor.options.random);

        const auto device = model.device();
        inputs = inputs.to(device);

        const bool was_training = model.is_training();
        model.eval();

        const std::size_t perturbations = std::max<std::size_t>(
            descriptor.options.samples == 0 ? 500 : descriptor.options.samples, 64);

        for (auto index : selected_indices) {
            auto sample = inputs.index({static_cast<int64_t>(index)}).clone();
            auto original_shape = sample.sizes().vec();
            auto flattened_sample = sample.flatten();

            const auto features = static_cast<std::size_t>(flattened_sample.numel());
            if (features == 0) {
                throw std::invalid_argument("LIME expects feature dimension greater than zero.");
            }
            auto target_class = flattened_targets.index({static_cast<int64_t>(index)}).item<int64_t>();

            auto masks = torch::bernoulli(torch::full({static_cast<int64_t>(perturbations),
                                                        static_cast<int64_t>(features)},
                                                       0.5,
                                                       sample.options()))
                            .to(torch::kFloat32);
            masks.index_put_({0}, torch::ones_like(masks.index({0})));

            auto distances = (masks - 1.0).abs().sum(1) / static_cast<double>(features);
            auto expanded = flattened_sample.unsqueeze(0).expand({static_cast<int64_t>(perturbations), flattened_sample.size(0)});
            auto perturbed = masks * expanded;

            std::vector<int64_t> perturbed_shape;
            perturbed_shape.reserve(original_shape.size() + 1);
            perturbed_shape.push_back(static_cast<int64_t>(perturbations));
            perturbed_shape.insert(perturbed_shape.end(), original_shape.begin(), original_shape.end());
            perturbed = perturbed.view(perturbed_shape);


            auto logits = model.forward(perturbed);
            if (logits.dim() == 1) {
                logits = logits.unsqueeze(0);
            }
            if (logits.dim() != 2) {
                throw std::runtime_error("LIME expects model outputs shaped as (batch, classes).");
            }

            auto probabilities = torch::softmax(logits, 1);
            auto responses = probabilities.index({torch::indexing::Slice(), target_class});

            auto weights = detail::ComputeLIMEWeights(masks, responses, distances)
                               .to(torch::kDouble)
                               .cpu();
            auto accessor = weights.accessor<double, 1>();

            std::vector<std::pair<std::size_t, double>> ranked;
            ranked.reserve(features);
            double normalizer = 0.0;
            for (std::size_t f = 0; f < features; ++f) {
                double value = accessor[static_cast<int64_t>(f)];
                ranked.emplace_back(f, value);
                normalizer += std::abs(value);
            }

            std::sort(ranked.begin(), ranked.end(), [](const auto& lhs, const auto& rhs) {
                return std::abs(lhs.second) > std::abs(rhs.second);
            });

            if (descriptor.options.normalize && normalizer > 0.0) {
                for (auto& entry : ranked) {
                    entry.second /= normalizer;
                }
            }

            const std::size_t top_k = std::min<std::size_t>(ranked.size(), 10);
            ranked.resize(top_k);

            detail::RenderLIMEPlot(index, ranked, descriptor.options.showWeights, descriptor.options.terminal);
        }

        if (was_training) {
            model.train();
        }
    }
}

#endif // Nott_RELIABILITY_LIME_HPP
