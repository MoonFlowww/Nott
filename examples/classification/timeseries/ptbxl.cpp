/**
 * @brief PTB-XL ECG classification with deep Conv1d-Residual backbone.
 *
 * Architecture:
 *   Conv1d stem (leads -> 64 -> 128 -> 256) with stride-2 downsampling
 *   -> 3x Residual blocks (Conv1d 256->256->256) + SiLU + SoftDropout
 *   -> AdaptiveAvgPool1d(1) -> Flatten
 *   -> FC(256 -> 128) + SiLU + HardDropout(0.2)
 *   -> FC(128 -> num_classes)
 *
 * Training regime:
 *   - Optimizer: AdamW(lr=3e-4, weight_decay=1e-2) + CosineAnnealing LR with warmup
 *   - Loss: CrossEntropy with label_smoothing=0.05
 *   - Regularization: L2(5e-5) + Orthogonality(1e-4)
 *   - Z-score normalization (lag=30) on raw signals
 *   - 25 epochs, batch_size=64, AMP enabled, graph capture mode
 */

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <regex>
#include <string>
#include <stdexcept>
#include <utility>
#include <vector>

#include <torch/torch.h>

#include "../../../include/Nott.h"

int main() {
    const bool use_cuda = torch::cuda::is_available();

    /* ---- Data loading ---- */
    const auto dataset = Nott::Data::Load::PTBXL<>(
        "/home/moonfloww/Projects/DATASETS/Timeserie/ECG_ACC",
        true,   /* low_resolution */
        0.8f,   /* train_fraction */
        true,   /* normalize */
        false   /* multilabel */
    );

    auto [train_signals, train_labels, val_signals, val_labels] = dataset;

    const int64_t input_length = train_signals.size(2);  /* timesteps */
    const int64_t num_leads = train_signals.size(1);      /* ECG leads */
    const int64_t num_classes = train_labels.max().item<int64_t>() + 1;

    if (input_length <= 0 || num_leads <= 0) {
        std::cerr << "Invalid dataset dimensions.\n";
        return 1;
    }

    std::cout << "PTB-XL: " << train_signals.size(0) << " train samples, "
              << val_signals.size(0) << " val samples, "
              << num_leads << " leads x " << input_length << " timesteps, "
              << num_classes << " classes\n";

    /* ---- Normalization ---- */
    Nott::Data::Transform::Normalization::Zscore(train_signals,
        {.lag = 30, .forward_only = true});
    Nott::Data::Transform::Normalization::Zscore(val_signals,
        {.lag = 30, .forward_only = true});

    /* ---- Build model ---- */
    Nott::Model model("PTBXL_ECG_ResNet");
    model.use_cuda(use_cuda);

    const int64_t B = 64;
    const int64_t epochs = 25;
    const int64_t steps_per_epoch = (train_signals.size(0) + B - 1) / B;

    /** Conv1d stem: lead-wise feature extraction with progressive channel expansion */
    model.add(Nott::Layer::Conv1d(
        {.in_channels = num_leads, .out_channels = 64,
         .kernel_size = {15}, .stride = {2}, .padding = {7}, .bias = false},
        Nott::Activation::SiLU,
        Nott::Initialization::HeNormal
    ), "stem1");

    model.add(Nott::Layer::SoftDropout({.probability = 0.05}), "sd_stem1");

    model.add(Nott::Layer::Conv1d(
        {.in_channels = 64, .out_channels = 128,
         .kernel_size = {9}, .stride = {2}, .padding = {4}, .bias = false},
        Nott::Activation::SiLU,
        Nott::Initialization::HeNormal
    ), "stem2");

    model.add(Nott::Layer::SoftDropout({.probability = 0.1}), "sd_stem2");

    model.add(Nott::Layer::Conv1d(
        {.in_channels = 128, .out_channels = 256,
         .kernel_size = {7}, .stride = {2}, .padding = {3}, .bias = false},
        Nott::Activation::SiLU,
        Nott::Initialization::HeNormal
    ), "stem3");

    /** Deep residual blocks with Conv1d for feature refinement */
    model.add(Nott::Block::Residual({
        Nott::Layer::Conv1d(
            {.in_channels = 256, .out_channels = 256,
             .kernel_size = {5}, .stride = {1}, .padding = {2}, .bias = false},
            Nott::Activation::SiLU,
            Nott::Initialization::HeNormal),
        Nott::Layer::SoftDropout({.probability = 0.1}),
        Nott::Layer::Conv1d(
            {.in_channels = 256, .out_channels = 256,
             .kernel_size = {5}, .stride = {1}, .padding = {2}, .bias = false},
            Nott::Activation::Identity,
            Nott::Initialization::HeNormal),
    }, /*repeats=*/3), "res_block");

    /** Global pooling and classification head */
    model.add(Nott::Layer::AdaptiveAvgPool1d({{1}}), "gap");

    model.add(Nott::Layer::Flatten(), "flatten");

    model.add(Nott::Layer::SoftDropout({.probability = 0.2}), "sd_head");

    model.add(Nott::Layer::FC(
        {256, 128, true},
        Nott::Activation::SiLU,
        Nott::Initialization::HeNormal
    ), "fc1");

    model.add(Nott::Layer::HardDropout({.probability = 0.3}), "hd1");

    model.add(Nott::Layer::FC(
        {128, num_classes, true},
        Nott::Activation::Identity,
        Nott::Initialization::XavierUniform
    ), "logits");

    /* ---- Training configuration ---- */
    model.set_optimizer(
        Nott::Optimizer::AdamW({
            .learning_rate = 3e-4,
            .beta1 = 0.9,
            .beta2 = 0.999,
            .eps = 1e-8,
            .weight_decay = 1e-2,
            .amsgrad = false
        }),
        Nott::LrScheduler::CosineAnnealing({
            .T_max = static_cast<std::size_t>(epochs) * static_cast<std::size_t>(steps_per_epoch),
            .eta_min = 1e-6,
            .warmup_steps = 3 * static_cast<std::size_t>(steps_per_epoch),
            .warmup_start_factor = 0.1
        })
    );

    model.set_loss(Nott::Loss::CrossEntropy({.label_smoothing = 0.05f}));

    model.set_regularization({
        Nott::Regularization::L2({.coefficient = 5e-5}),
        Nott::Regularization::Orthogonality({.coefficient = 1e-4}),
    });

    /* ---- Training ---- */
    Nott::Data::Check::Size(train_signals, "Train signals");
    Nott::Data::Check::Size(train_labels, "Train labels");

    std::cout << "\nTraining PTBXL_ECG_ResNet for " << epochs << " epochs...\n";
    model.train(train_signals, train_labels, {
        .epoch = static_cast<std::size_t>(epochs),
        .batch_size = static_cast<std::size_t>(B),
        .shuffle = true,
        .restore_best_state = true,
        .test = std::vector<at::Tensor>{val_signals, val_labels},
        .graph_mode = Nott::GraphMode::Capture,
        .enable_amp = true
    });

    /* ---- Evaluation ---- */
    std::cout << "\nEvaluating on validation set...\n";
    model.evaluate(val_signals, val_labels,
        Nott::Evaluation::Classification,
        {
            Nott::Metric::Classification::Accuracy,
            Nott::Metric::Classification::F1,
            Nott::Metric::Classification::Precision,
            Nott::Metric::Classification::Recall,
            Nott::Metric::Classification::AUCROC,
            Nott::Metric::Classification::BalancedAccuracy,
            Nott::Metric::Classification::LogLoss,
            Nott::Metric::Classification::CohensKappa,
            Nott::Metric::Classification::Informedness,
        },
        {.batch_size = static_cast<std::size_t>(B), .buffer_vram = 1}
    );

    return 0;
}
