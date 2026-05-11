#ifndef Nott_EVALUATION_HPP
#define Nott_EVALUATION_HPP
/// This file is an factory, must exempt it from any logical-code. For functions look into "/details"
#include <utility>
#include <vector>

#include <torch/torch.h>

#include "details/classification.hpp"
#include "details/timeserie.hpp"

namespace Nott::Evaluation {
    using ClassificationOptions = Details::Classification::Options;
    using TimeseriesOptions = Details::Timeseries::Options;

    using ClassificationDescriptor = Details::Classification::Descriptor;
    inline constexpr ClassificationDescriptor Classification{};

    using MultiClassificationDescriptor = Details::Classification::MultiDescriptor;
    inline constexpr MultiClassificationDescriptor MultiClassification{};

    using SegmentationDescriptor = Details::Classification::SegmentationDescriptor;
    inline constexpr SegmentationDescriptor Segmentation{};

    using ClassificationReport = Details::Classification::Report;


    using TimeseriesDescriptor = Details::Timeseries::Descriptor;
    inline constexpr TimeseriesDescriptor Timeseries{};

    using TimeseriesReport = Details::Timeseries::Report;

    using Options = ClassificationOptions;

    template <class Model>
    [[nodiscard]] inline auto Evaluate(Model& model, torch::Tensor inputs, torch::Tensor targets,
                                       ClassificationDescriptor, std::vector<Metric::Classification::Descriptor> metrics,
                                       const ClassificationOptions& options = ClassificationOptions{}) -> ClassificationReport {
        return Details::Classification::Evaluate(
            model,
            std::move(inputs),
            std::move(targets),
            Details::Classification::Descriptor{},
            std::move(metrics),
            options);
    }

    template <class Model>
    [[nodiscard]] inline auto Evaluate(Model& model, torch::Tensor inputs, torch::Tensor targets,
                                       MultiClassificationDescriptor, std::vector<Metric::Classification::Descriptor> metrics,
                                       const ClassificationOptions& options = ClassificationOptions{}) -> ClassificationReport {
        return Details::Classification::Evaluate(
            model,
            std::move(inputs),
            std::move(targets),
            Details::Classification::MultiDescriptor{},
            std::move(metrics),
            options);
    }

    template <class Model>
    [[nodiscard]] inline auto Evaluate(Model& model, torch::Tensor inputs, torch::Tensor targets,
                                       SegmentationDescriptor, std::vector<Metric::Classification::Descriptor> metrics,
                                       const ClassificationOptions& options = ClassificationOptions{}) -> ClassificationReport {
        return Details::Classification::Evaluate(
            model,
            std::move(inputs),
            std::move(targets),
            Details::Classification::SegmentationDescriptor{},
            std::move(metrics),
            options);
    }

    template <class Model> [[nodiscard]] inline auto Evaluate(Model& model, torch::Tensor inputs, torch::Tensor targets,
                                   TimeseriesDescriptor, std::vector<Metric::Timeseries::Descriptor> metrics,
                                   const TimeseriesOptions& options = TimeseriesOptions{}) -> TimeseriesReport {
        return Details::Timeseries::Evaluate(
            model,
            std::move(inputs),
            std::move(targets),
            std::move(metrics),
            options);
    }
}

#endif //Nott_EVALUATION_HPP