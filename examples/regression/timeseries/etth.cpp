/**
 * @brief ETTh1 electricity transformer temperature forecasting with TCN backbone.
 *
 * Architecture:
 *   Conv1d stem (features → 64 → 128 → 256) with stride-2 downsampling
 *   → Residual Conv1d block (256→256→256) × 2 repeats
 *   → AdaptiveAvgPool1d(1) → Flatten
 *   → FC(256 → 128) + SiLU + HardDropout(0.15)
 *   → FC(128 → 1) regression head
 *
 * Data preparation:
 *   - ETTh1 CSV loaded with 80/20 train/test split
 *   - Sliding windows of lookback=96 timesteps to predict next-step OT
 *   - Z-score normalization per feature
 *
 * Training regime:
 *   - Optimizer: AdamW(lr=5e-4, weight_decay=1e-2) + CosineAnnealing LR with warmup
 *   - Loss: SmoothL1 (robust to outliers, better than MSE for temperature data)
 *   - Regularization: L2(1e-4) + Orthogonality(5e-5)
 *   - 30 epochs, batch_size=64, AMP enabled, graph capture mode
 *
 * Evaluation metrics: MSE, MAE, RMSE reported on held-out test set.
 *
 */

#include <iostream>
#include <cstddef>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>

#include <torch/torch.h>

#include "../../../include/Nott.h"

int main() {
    const bool use_cuda = torch::cuda::is_available();

    /* ---- Data loading ---- */
    auto [x_raw, y_raw, x_test_raw, y_test_raw] =
        Nott::Data::Load::ETTh(
            "/home/moonfloww/Projects/DATASETS/ETT/ETTh1/ETTh1.csv",
            0.8f,   /* train_fraction */
            0.2f,   /* test_fraction */
            true    /* normalize */
        );

    const int64_t total_train = x_raw.size(0);
    const int64_t num_features = x_raw.size(1);
    const int64_t lookback = 96;

    std::cout << "ETTh1: " << total_train << " train timesteps, "
              << x_test_raw.size(0) << " test timesteps, "
              << num_features << " features, lookback=" << lookback << "\n";

    /* ---- Create sliding windows for supervised forecasting ---- */
    auto make_windows = [&](const torch::Tensor& x, const torch::Tensor& y) {
        const int64_t T = x.size(0);
        if (T <= lookback) {
            throw std::runtime_error("Not enough timesteps for lookback window.");
        }
        const int64_t N = T - lookback;
        // x_windows: (N, lookback, num_features) → reshape to (N, num_features, lookback) for Conv1d
        auto x_win = torch::empty({N, num_features, lookback},
                                   x.options().dtype(torch::kFloat32));
        auto y_win = torch::empty({N}, y.options().dtype(torch::kFloat32));

        for (int64_t i = 0; i < N; ++i) {
            auto window = x.slice(0, i, i + lookback);  // (lookback, F)
            x_win[i] = window.transpose(0, 1).contiguous();  // (F, lookback)
            y_win[i] = y[i + lookback];
        }
        return std::make_pair(x_win, y_win);
    };

    auto [x_train, y_train] = make_windows(x_raw, y_raw);
    auto [x_test, y_test]   = make_windows(x_test_raw, y_test_raw);

    /* ---- Split train into train/validation (90/10) ---- */
    const int64_t val_size = x_train.size(0) / 10;
    auto x_val = x_train.slice(0, 0, val_size);
    auto y_val = y_train.slice(0, 0, val_size);
    x_train = x_train.slice(0, val_size);
    y_train = y_train.slice(0, val_size);

    std::cout << "After windowing: " << x_train.size(0) << " train, "
              << x_val.size(0) << " val, "
              << x_test.size(0) << " test samples\n";

    /* ---- Build model ---- */
    Nott::Model model("ETTh1_TCN");
    model.use_cuda(use_cuda);

    const int64_t B = 64;
    const int64_t epochs = 30;
    const int64_t steps_per_epoch = (x_train.size(0) + B - 1) / B;

    /** TCN stem: progressive temporal downsampling with channel expansion */
    model.add(Nott::Layer::Conv1d(
        {.in_channels = num_features, .out_channels = 64,
         .kernel_size = {7}, .stride = {2}, .padding = {3}, .bias = false},
        Nott::Activation::SiLU,
        Nott::Initialization::HeNormal
    ), "stem1");

    model.add(Nott::Layer::SoftDropout({.probability = 0.05}), "sd1");

    model.add(Nott::Layer::Conv1d(
        {.in_channels = 64, .out_channels = 128,
         .kernel_size = {5}, .stride = {2}, .padding = {2}, .bias = false},
        Nott::Activation::SiLU,
        Nott::Initialization::HeNormal
    ), "stem2");

    model.add(Nott::Layer::SoftDropout({.probability = 0.1}), "sd2");

    model.add(Nott::Layer::Conv1d(
        {.in_channels = 128, .out_channels = 256,
         .kernel_size = {5}, .stride = {2}, .padding = {2}, .bias = false},
        Nott::Activation::SiLU,
        Nott::Initialization::HeNormal
    ), "stem3");

    /** Deep residual block for feature refinement */
    model.add(Nott::Block::Residual({
        Nott::Layer::Conv1d(
            {.in_channels = 256, .out_channels = 256,
             .kernel_size = {3}, .stride = {1}, .padding = {1}, .bias = false},
            Nott::Activation::SiLU,
            Nott::Initialization::HeNormal),
        Nott::Layer::SoftDropout({.probability = 0.1}),
        Nott::Layer::Conv1d(
            {.in_channels = 256, .out_channels = 256,
             .kernel_size = {3}, .stride = {1}, .padding = {1}, .bias = false},
            Nott::Activation::Identity,
            Nott::Initialization::HeNormal),
    }, /*repeats=*/2), "res_block");

    /** Global temporal pooling */
    model.add(Nott::Layer::AdaptiveAvgPool1d({{1}}), "gap");

    model.add(Nott::Layer::Flatten(), "flatten");

    /** Regression head */
    model.add(Nott::Layer::SoftDropout({.probability = 0.1}), "sd_head");

    model.add(Nott::Layer::FC(
        {256, 128, true},
        Nott::Activation::SiLU,
        Nott::Initialization::HeNormal
    ), "fc1");

    model.add(Nott::Layer::HardDropout({.probability = 0.15}), "hd1");

    model.add(Nott::Layer::FC(
        {128, 64, true},
        Nott::Activation::SiLU,
        Nott::Initialization::HeNormal
    ), "fc2");

    model.add(Nott::Layer::FC(
        {64, 1, true},
        Nott::Activation::Identity,
        Nott::Initialization::XavierUniform
    ), "output");

    /* ---- Training configuration ---- */
    model.set_optimizer(
        Nott::Optimizer::AdamW({
            .learning_rate = 5e-4,
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

    /** SmoothL1 is more robust to outliers than MSE for temperature data */
    model.set_loss(Nott::Loss::SmoothL1({.beta = 1.0}));

    model.set_regularization({
        Nott::Regularization::L2({.coefficient = 1e-4}),
        Nott::Regularization::Orthogonality({.coefficient = 5e-5}),
    });

    /* ---- Training ---- */
    Nott::Data::Check::Size(x_train, "Train windows");
    Nott::Data::Check::Size(y_train, "Train targets");

    std::cout << "\nTraining ETTh1_TCN for " << epochs << " epochs...\n";
    model.train(x_train, y_train, {
        .epoch = static_cast<std::size_t>(epochs),
        .batch_size = static_cast<std::size_t>(B),
        .shuffle = true,
        .restore_best_state = true,
        .test = std::vector<at::Tensor>{x_val, y_val},
        .graph_mode = Nott::GraphMode::Capture,
        .enable_amp = true
    });

    /* ---- Evaluation ---- */
    std::cout << "\nEvaluating on test set...\n";

    model.eval();
    torch::NoGradGuard no_grad;

    const int64_t test_batches = (x_test.size(0) + B - 1) / B;
    double total_mse = 0.0, total_mae = 0.0;
    int64_t total_samples = 0;

    for (int64_t i = 0; i < test_batches; ++i) {
        const int64_t start = i * B;
        const int64_t end = std::min(start + B, x_test.size(0));

        auto batch_x = x_test.slice(0, start, end);
        auto batch_y = y_test.slice(0, start, end);

        if (use_cuda) {
            batch_x = batch_x.to(torch::kCUDA);
            batch_y = batch_y.to(torch::kCUDA);
        }

        auto pred = model.forward(batch_x).squeeze(-1);
        auto diff = pred - batch_y;

        total_mse += diff.pow(2).sum().item<double>();
        total_mae += diff.abs().sum().item<double>();
        total_samples += batch_x.size(0);
    }

    double mse = total_mse / static_cast<double>(total_samples);
    double mae = total_mae / static_cast<double>(total_samples);
    double rmse = std::sqrt(mse);

    std::cout << "\n=== Test Set Regression Metrics ===\n";
    std::cout << "  MSE:  " << mse << "\n";
    std::cout << "  RMSE: " << rmse << "\n";
    std::cout << "  MAE:  " << mae << "\n";

    /* ---- Baseline comparison: naive persistence model ---- */
    {
        double persist_mse = 0.0;
        int64_t persist_N = x_test_raw.size(0) - 1;
        for (int64_t i = 0; i < persist_N; ++i) {
            double pred = y_test_raw[i].item<double>();  // last known value
            double actual = y_test_raw[i + 1].item<double>();
            persist_mse += (pred - actual) * (pred - actual);
        }
        double persist_rmse = std::sqrt(persist_mse / static_cast<double>(persist_N));
        std::cout << "  Persistence baseline RMSE: " << persist_rmse << "\n";
        std::cout << "  Improvement over baseline: "
                  << (1.0 - rmse / persist_rmse) * 100.0 << "%\n";
    }

    return 0;
}
