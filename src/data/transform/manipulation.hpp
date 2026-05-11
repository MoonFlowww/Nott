#ifndef Nott_MANIPULATION_HPP
#define Nott_MANIPULATION_HPP
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <iostream>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <torch/torch.h>
#include <torch/nn/functional.h>


#include "augment/atmospheric_drift.hpp"
#include "augment/chromatic_aberration.hpp"
#include "augment/clahe.hpp"
#include "augment/cloud_occlusion.hpp"
#include "augment/cutout.hpp"
#include "augment/flip.hpp"
#include "augment/grid_distortion.hpp"
#include "augment/random_brightness_contrast.hpp"
#include "augment/optical_distortion.hpp"
#include "augment/sun_angle_jitter.hpp"
#include "normalization/registry.hpp"

namespace Nott::Data::Manipulation {
    using ::Nott::Data::Transform::Augmentation::AtmosphericDrift;
    using ::Nott::Data::Transform::Augmentation::ChromaticAberration;
    using ::Nott::Data::Transform::Augmentation::CLAHE;
    using ::Nott::Data::Transform::Augmentation::CloudOcclusion;
    using ::Nott::Data::Transform::Augmentation::Cutout;
    using ::Nott::Data::Transform::Augmentation::Flip;
    using ::Nott::Data::Transform::Augmentation::GridDistortion;
    using ::Nott::Data::Transform::Augmentation::OpticalDistortion;
    using ::Nott::Data::Transform::Augmentation::RandomBrightnessContrast;
    using ::Nott::Data::Transform::Augmentation::SunAngleJitter;

    namespace Normalization = ::Nott::Data::Transform::Normalization;

    /// Fisher-Yates
    inline std::pair<torch::Tensor, torch::Tensor> Shuffle(const torch::Tensor& inputs, const torch::Tensor& targets, const std::optional<std::uint64_t>& seed = std::nullopt) {
        if (inputs.dim() == 0 || targets.dim() == 0 || inputs.size(0) == 0) {
            return {inputs.clone(), targets.clone()};
        }

        if (inputs.size(0) != targets.size(0)) {
            throw std::invalid_argument("Inputs and targets must contain the same number of samples to shuffle.");
        }

        if (seed.has_value()) {
            torch::manual_seed(static_cast<std::uint64_t>(*seed));
        }

        auto permutation = torch::randperm(inputs.size(0), torch::TensorOptions().dtype(torch::kLong).device(inputs.device()));
        if (permutation.device() != inputs.device()) {
            permutation = permutation.to(inputs.device());
        }

        auto shuffled_inputs = inputs.index_select(0, permutation);
        auto shuffled_targets = targets.index_select(0, permutation);
        return {shuffled_inputs, shuffled_targets};
    }

    inline std::pair<torch::Tensor, torch::Tensor> Fraction(const torch::Tensor& inputs, const torch::Tensor& targets, float fraction) {
        if (!inputs.defined() || !targets.defined()) {
            throw std::invalid_argument("Fraction expects both input and target tensors to be defined.");
        }

        if (inputs.dim() == 0 || targets.dim() == 0) {
            throw std::invalid_argument("Fraction expects tensors with at least one dimension.");
        }

        if (inputs.size(0) != targets.size(0)) {
            throw std::invalid_argument("Inputs and targets must contain the same number of samples to extract a fraction.");
        }

        const auto total_samples = inputs.size(0);
        if (total_samples == 0) {
            return {inputs.clone(), targets.clone()};
        }

        const auto clamped_fraction = std::clamp(fraction, 0.0f, 1.0f);
        auto desired_samples = static_cast<int64_t>(std::round(static_cast<double>(total_samples) * static_cast<double>(clamped_fraction)));
        desired_samples = std::clamp<int64_t>(desired_samples, 0, total_samples);

        if (desired_samples == total_samples) {
            return {inputs.clone(), targets.clone()};
        }

        auto fraction_inputs = inputs.narrow(0, 0, desired_samples).clone();
        auto fraction_targets = targets.narrow(0, 0, desired_samples).clone();
        return {fraction_inputs, fraction_targets};
    }


    inline torch::Tensor Upsample(const torch::Tensor& input,
                                  const std::vector<double>& scale_factors,
                                  torch::nn::functional::InterpolateFuncOptions::mode_t mode = torch::kNearest,
                                  bool align_corners = false,
                                  bool recompute_scale_factor = false) {
        using namespace torch::nn::functional;

        InterpolateFuncOptions options;
        options = options.mode(mode);
        options = options.align_corners(align_corners);
        options = options.recompute_scale_factor(recompute_scale_factor);
        options = options.scale_factor(scale_factors);

        return interpolate(input, options);
    }

    inline torch::Tensor Grayscale(const torch::Tensor& input,
                                   const std::array<double, 3>& weights = {0.2989, 0.5870, 0.1140}) {
        if (input.dim() < 3) {
            throw std::invalid_argument("Grayscale expects a tensor with at least 3 dimensions (C, H, W).");
        }

        auto tensor = input;
        bool had_batch = false;
        if (tensor.dim() == 3) {
            tensor = tensor.unsqueeze(0);
            had_batch = true;
        }

        if (tensor.size(1) != 3) {
            throw std::invalid_argument("Grayscale expects the channel dimension to have size 3.");
        }

        auto options = tensor.options().dtype(torch::kFloat32);
        auto weight_tensor = torch::from_blob(const_cast<double*>(weights.data()), {3}, torch::TensorOptions().dtype(torch::kDouble));
        weight_tensor = weight_tensor.to(options.dtype()).to(tensor.device()).view({3, 1, 1});

        auto grayscale = (tensor.to(options.dtype()) * weight_tensor).sum(1, true);
        if (had_batch) {
            grayscale = grayscale.squeeze(0);
        }
        return grayscale;
    }

}

namespace Nott::Data::Check {
    struct ImbalanceReport {
        torch::Tensor train_counts;
        torch::Tensor test_counts;
        torch::Tensor train_distribution;
        torch::Tensor test_distribution;
        double kl_divergence{0.0};
        double train_entropy{0.0};
        double test_entropy{0.0};

        [[nodiscard]] std::string str() const {
            std::ostringstream oss;
            oss << "Class imbalance diagnostics:\n";
            oss << "  Train counts: " << train_counts.to(torch::kCPU) << "\n";
            oss << "  Test counts: " << test_counts.to(torch::kCPU) << "\n";
            oss << "  Train distribution: " << train_distribution.to(torch::kCPU) << "\n";
            oss << "  Test distribution: " << test_distribution.to(torch::kCPU) << "\n";
            oss << "  KL(train || test): " << kl_divergence << "\n";
            oss << "  Entropy(train): " << train_entropy << ", Entropy(test): " << test_entropy;
            return oss.str();
        }
    };

    inline torch::Tensor compute_counts(const torch::Tensor& targets, int64_t num_classes) {
        auto cpu_targets = targets.to(torch::kCPU);
        if (cpu_targets.dim() != 1) {
            cpu_targets = cpu_targets.reshape({-1});
        }
        if (cpu_targets.dtype() != torch::kLong) {
            cpu_targets = cpu_targets.to(torch::kLong);
        }
        return torch::bincount(cpu_targets, {}, num_classes);
    }

    inline torch::Tensor safe_distribution(const torch::Tensor& counts) {
        auto total = counts.sum().item<double>();
        if (total <= 0.0) {
            return torch::zeros_like(counts, torch::TensorOptions().dtype(torch::kDouble));
        }
        return counts.to(torch::kDouble) / total;
    }

    inline double entropy(const torch::Tensor& distribution) {
        auto eps = 1e-12;
        auto clipped = torch::clamp(distribution, eps, 1.0);
        auto entropy_tensor = -(clipped * torch::log(clipped));
        return entropy_tensor.sum().item<double>();
    }

    inline ImbalanceReport Imbalance(const torch::Tensor& train_targets,
                                     const torch::Tensor& test_targets,
                                     std::optional<int64_t> explicit_classes = std::nullopt) {
        const auto train_max = train_targets.numel() > 0
            ? train_targets.to(torch::kCPU).max().item<int64_t>() + 1
            : int64_t{0};
        const auto test_max = test_targets.numel() > 0
            ? test_targets.to(torch::kCPU).max().item<int64_t>() + 1
            : int64_t{0};

        const auto inferred_classes = std::max({
            train_max,
            test_max,
            explicit_classes.value_or(int64_t{0})
        });

        const auto num_classes = std::max<int64_t>(inferred_classes, int64_t{1});

        auto train_counts = compute_counts(train_targets, num_classes);
        auto test_counts = compute_counts(test_targets, num_classes);

        auto train_distribution = safe_distribution(train_counts);
        auto test_distribution = safe_distribution(test_counts);

        auto eps = 1e-12;
        auto clipped_train = torch::clamp(train_distribution, eps, 1.0);
        auto clipped_test = torch::clamp(test_distribution, eps, 1.0);

        auto kl_tensor = clipped_train * (torch::log(clipped_train) - torch::log(clipped_test));
        auto kl_divergence = kl_tensor.sum().item<double>();

        ImbalanceReport report{};
        report.train_counts = train_counts;
        report.test_counts = test_counts;
        report.train_distribution = train_distribution;
        report.test_distribution = test_distribution;
        report.kl_divergence = kl_divergence;
        report.train_entropy = entropy(train_distribution);
        report.test_entropy = entropy(test_distribution);
        return report;
    }

    inline std::string Shuffled(const torch::Tensor& targets, double threshold = 0.95) {
        if (targets.numel() <= 1) {
            return "Dataset order: insufficient data to determine shuffle";
        }

        auto flattened = targets.reshape({-1}).to(torch::kCPU);
        if (flattened.dtype() != torch::kLong && flattened.dtype() != torch::kInt64) {
            flattened = flattened.to(torch::kLong);
        }

        auto runs = std::get<0>(torch::unique_consecutive(flattened));
        const auto run_ratio = static_cast<double>(runs.size(0)) / static_cast<double>(flattened.size(0));

        std::ostringstream oss;
        const bool shuffled = run_ratio >= threshold;
        oss << "Dataset order: " << (shuffled ? "likely shuffled" : "likely grouped")
            << " (unique-consecutive ratio=" << run_ratio << ")";
        return oss.str();
    }

    inline std::vector<int> Size(const torch::Tensor& x, const std::string& title = "Tensor Shape") {
        at::IntArrayRef sizes = x.sizes();
        std::cout << title << ": (";
        for (size_t i = 0; i < sizes.size(); i++) {
            std::cout << sizes[i];
            if (i + 1 < sizes.size()) std::cout << ", ";
        }
        std::cout << ")" << std::endl;
        return {x.sizes().begin(), x.sizes().end()};
    }
}

#endif //Nott_MANIPULATION_HPP