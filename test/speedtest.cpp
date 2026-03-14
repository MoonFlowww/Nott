#include <torch/torch.h>
#include <../include/Nott.h>
#include <chrono>
#include <limits>
#include <string>
#include <algorithm>
#include <vector>
#include <numeric>
#include <iostream>

struct NetImpl : torch::nn::Module {
    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr}, conv3{nullptr}, conv4{nullptr}, conv5{nullptr};
    torch::nn::MaxPool2d pool{nullptr};
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};

    NetImpl() {
        conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 32, 3).stride(1).padding(1)));
        conv2 = register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 32, 3).stride(1).padding(1)));
        pool  = register_module("pool",  torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions({2, 2}).stride({2, 2})));

        conv3 = register_module("conv3", torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 64, 3).stride(1).padding(1)));
        conv4 = register_module("conv4", torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, 3).stride(1).padding(1)));
        conv5 = register_module("conv5", torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 128, 3).stride(1).padding(1)));

        fc1 = register_module("fc1", torch::nn::Linear(1152, 524));
        fc2 = register_module("fc2", torch::nn::Linear(524, 126));
        fc3 = register_module("fc3", torch::nn::Linear(126, 10));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(conv1->forward(x));
        x = torch::relu(conv2->forward(x));
        x = pool->forward(x);

        x = torch::relu(conv3->forward(x));
        x = torch::relu(conv4->forward(x));
        x = pool->forward(x);

        x = torch::relu(conv5->forward(x));
        x = pool->forward(x);

        x = x.view({x.size(0), -1});
        x = torch::relu(fc1->forward(x));
        x = torch::relu(fc2->forward(x));
        x = fc3->forward(x);
        return x;
    }
};
TORCH_MODULE(Net);
namespace LatencyUtils {
    struct StepLatencyStats {
        std::string title;
        double total_ms = 0.0;
        double min_ms   = std::numeric_limits<double>::max();
        double max_ms   = 0.0;
        std::size_t steps = 0;

        // Online central moments
        double mean = 0.0;
        double M2   = 0.0; // sum of squared deviations
        double M3   = 0.0; // sum of cubed deviations

        double median_ms = 0.0;
        double p10_ms    = 0.0;
        double p90_ms    = 0.0;
        double p98_ms    = 0.0;
        double mode_ms   = 0.0;

        void record(double ms, std::size_t count = 1) {
            if (count == 0) return;

            for (std::size_t c = 0; c < count; ++c) {
                double n1 = static_cast<double>(steps);
                double n  = n1 + 1.0;

                double delta    = ms - mean;
                double delta_n  = delta / n;
                double delta_n2 = delta_n * delta_n;
                double term1    = delta * delta_n * n1;

                // Pebay update for central moments (up to 3rd)
                double newM3   = M3 + term1 * delta_n * (n1 - 1.0) - 3.0 * delta_n * M2;
                double newM2   = M2 + term1;
                double newMean = mean + delta_n;

                mean = newMean;
                M2   = newM2;
                M3   = newM3;

                ++steps;
            }

            total_ms += ms * static_cast<double>(count);
            min_ms = std::min(min_ms, ms);
            max_ms = std::max(max_ms, ms);
        }

        double average_ms() const {
            return steps == 0 ? 0.0 : mean;
        }

        double variance_ms() const {
            if (steps < 2) return 0.0;
            return M2 / static_cast<double>(steps - 1);
        }

        double std_ms() const {
            return std::sqrt(variance_ms());
        }

        // Fisher's unbiased sample skewness
        double skew() const {
            if (steps < 3) return 0.0;

            const double n  = static_cast<double>(steps);
            const double m2 = M2 / n;
            const double m3 = M3 / n;
            if (m2 <= 0.0) return 0.0;

            const double g1 = m3 / std::pow(m2, 1.5); // moment skewness
            return std::sqrt(n * (n - 1.0)) / (n - 2.0) * g1;
        }

        double coeff_of_variation() const {
            const double mu  = average_ms();
            const double sig = std_ms();
            if (mu == 0.0) return 0.0;
            return sig / mu;
        }

        double steps_per_second() const {
            const double mu = average_ms();
            if (mu <= 0.0) return 0.0;
            return 1000.0 / mu;
        }
    };


    void print_stats(const StepLatencyStats& stats) {
        const double min_ms = stats.steps == 0 ? 0.0 : stats.min_ms;
        const double max_ms = stats.steps == 0 ? 0.0 : stats.max_ms;
        const double mu     = stats.average_ms();
        const double med = stats.median_ms;
        const double std    = stats.std_ms();
        const double skew   = stats.skew();

        std::cout << "\n" << stats.title << "\n";
        std::cout
            << "  Steps (after filters): " << stats.steps << "\n"
            << "  Avg latency:           " << mu   << " ms\n"
            << "  Med latency:           " << med   << " ms\n"
            << "  Std latency:           " << std  << " ms\n"
            << "  Skew latency:          " << skew << "\n";

        if (stats.steps > 0) {
            std::cout
                << "  P10 / P50 / P90 / P98: "
                << stats.p10_ms    << " / "
                << stats.median_ms << " / "
                << stats.p90_ms    << " / "
                << stats.p98_ms    << " ms\n"
                << "  Mode latency (hist):   " << stats.mode_ms << " ms\n"
                << "  Coeff. of variation:   " << stats.coeff_of_variation() << "\n"
                << "  Throughput:            " << stats.steps_per_second() << " steps/s\n";
        }

        std::cout
            << "  ~~~~~~~~~~~~~~~~~~~~~~~~~~~\n"
            << "  Min latency:            " << min_ms << " ms\n"
            << "  Max latency:            " << max_ms << " ms\n"
            << std::endl;
    }




    struct TukeyFence {
        double lower = 0.0;
        double upper = 0.0;
    };

    static double quantile_linear(const std::vector<double>& sorted, double p) {
        if (sorted.empty()) return 0.0;
        if (sorted.size() == 1) return sorted[0];

        const double idx = p * static_cast<double>(sorted.size() - 1);
        const std::size_t i = static_cast<std::size_t>(idx);
        const double frac = idx - static_cast<double>(i);

        if (i + 1 < sorted.size())
            return sorted[i] * (1.0 - frac) + sorted[i + 1] * frac;
        return sorted[i];
    }

    static TukeyFence compute_tukey_fence(const std::vector<double>& sorted_vals, double k) {
        if (sorted_vals.empty()) return {};

        const double q1 = quantile_linear(sorted_vals, 0.25);
        const double q3 = quantile_linear(sorted_vals, 0.75);
        const double iqr = q3 - q1;

        TukeyFence f;
        f.lower = q1 - k * iqr;
        f.upper = q3 + k * iqr;
        return f;
    }

    static double estimate_mode_from_sorted(const std::vector<double>& sorted) {
        if (sorted.empty()) return 0.0;
        if (sorted.size() == 1) return sorted[0];

        const double minv = sorted.front();
        const double maxv = sorted.back();
        if (minv == maxv) return minv;

        const std::size_t n = sorted.size();

        const double q1  = quantile_linear(sorted, 0.25);
        const double q3  = quantile_linear(sorted, 0.75);
        const double iqr = q3 - q1;

        // Freedman–Diaconis bin width
        double bin_width = 0.0;
        if (iqr > 0.0) {
            bin_width = 2.0 * iqr * std::pow(static_cast<double>(n), -1.0 / 3.0);
        }
        if (bin_width <= 0.0) {
            bin_width = (maxv - minv) / std::min<std::size_t>(n, 32);
        }

        std::size_t bin_count = static_cast<std::size_t>(std::ceil((maxv - minv) / bin_width));
        bin_count = std::clamp<std::size_t>(bin_count, 4, 128);

        std::vector<std::size_t> hist(bin_count, 0);
        const double inv_width = static_cast<double>(bin_count) / (maxv - minv);

        for (double v : sorted) {
            std::size_t idx = static_cast<std::size_t>((v - minv) * inv_width);
            if (idx >= bin_count) idx = bin_count - 1;
            hist[idx]++;
        }

        std::size_t best_bin   = 0;
        std::size_t best_count = hist[0];
        for (std::size_t i = 1; i < bin_count; ++i) {
            if (hist[i] > best_count) {
                best_count = hist[i];
                best_bin   = i;
            }
        }

        const double width = (maxv - minv) / static_cast<double>(bin_count);
        return minv + (static_cast<double>(best_bin) + 0.5) * width;
    }



    StepLatencyStats build_stats_from_samples(const std::string& title, const std::vector<double>& samples, std::size_t warmup_steps = 100, double tukey_k = 1.0) {
        StepLatencyStats stats{title};

        if (samples.size() <= warmup_steps)
            return stats;

        auto begin = samples.begin() + static_cast<std::ptrdiff_t>(warmup_steps);
        std::vector<double> tail(begin, samples.end());

        if (tail.empty()) {
            return stats;
        }

        const bool use_tukey = tukey_k >= 0.0;
        std::vector<double> sorted = tail;
        if (use_tukey)
            std::sort(sorted.begin(), sorted.end());

        std::vector<double> filtered;
        filtered.reserve(sorted.size());

        if (use_tukey) {
            const TukeyFence fence = compute_tukey_fence(sorted, tukey_k);

            for (double v : tail) {
                if (v < fence.lower || v > fence.upper)
                    continue;
                stats.record(v);
                filtered.push_back(v);
            }
        } else {
            for (double v : tail) {
                stats.record(v);
                filtered.push_back(v);
            }
        }

        if (!filtered.empty()) {
            std::sort(filtered.begin(), filtered.end());
            stats.median_ms = quantile_linear(filtered, 0.50);
            stats.p10_ms    = quantile_linear(filtered, 0.10);
            stats.p90_ms    = quantile_linear(filtered, 0.90);
            stats.p98_ms    = quantile_linear(filtered, 0.98);
            stats.mode_ms   = estimate_mode_from_sorted(filtered);
        }

        return stats;
    }

}

inline Nott::Loss::Details::CrossEntropyDescriptor set(Nott::Model& model, const bool&device) { // Used to reset learnable parameters between Nott PreBuild and Nott Custom
    model.add(Nott::Layer::Conv2d({.in_channels = 1, .out_channels = 32, .kernel_size = {3,3}, .stride={1,1}, .padding={1,1}, .dilation={1,1}}, Nott::Activation::ReLU, Nott::Initialization::HeUniform));
    model.add(Nott::Layer::Conv2d({.in_channels = 32, .out_channels = 32, .kernel_size = {3,3}, .stride={1,1}, .padding={1,1}, .dilation={1,1}}, Nott::Activation::ReLU, Nott::Initialization::HeUniform));
    model.add(Nott::Layer::MaxPool2d({.kernel_size={2,2}, .stride={2,2}, .padding={0,0}}));

    model.add(Nott::Layer::Conv2d({.in_channels = 32, .out_channels = 64, .kernel_size = {3,3}, .stride={1,1}, .padding={1,1}, .dilation={1,1}}, Nott::Activation::ReLU, Nott::Initialization::HeUniform));
    model.add(Nott::Layer::Conv2d({.in_channels = 64, .out_channels = 64, .kernel_size = {3,3}, .stride={1,1}, .padding={1,1}, .dilation={1,1}}, Nott::Activation::ReLU, Nott::Initialization::HeUniform));
    model.add(Nott::Layer::MaxPool2d({.kernel_size={2,2}, .stride={2,2}, .padding={0,0}}));

    model.add(Nott::Layer::Conv2d({.in_channels = 64, .out_channels = 128, .kernel_size = {3,3}, .stride={1,1}, .padding={1,1}, .dilation={1,1}}, Nott::Activation::ReLU, Nott::Initialization::HeUniform));
    model.add(Nott::Layer::MaxPool2d({.kernel_size={2,2}, .stride={2,2}, .padding={0,0}}));

    model.add(Nott::Layer::Flatten());
    model.add(Nott::Layer::FC({1152, 524}, Nott::Activation::ReLU, Nott::Initialization::HeUniform));
    model.add(Nott::Layer::FC({524, 126}, Nott::Activation::ReLU, Nott::Initialization::HeUniform));
    model.add(Nott::Layer::FC({126, 10}, Nott::Activation::Identity, Nott::Initialization::HeUniform));

    model.set_optimizer(Nott::Optimizer::SGD({.learning_rate = 1e-3f, .momentum = 0.9, .nesterov = true}));
    const auto ce = Nott::Loss::CrossEntropy({.label_smoothing = 0.02f});
    model.set_loss(ce);
    model.use_cuda(device);
    return ce;
}


static torch::Tensor stage_for_device(torch::Tensor tensor, const bool& device) { const auto dev = device ? torch::kCUDA : torch::kCPU;
    auto pinned = Nott::async_pin_memory(std::move(tensor));
    auto host_tensor = pinned.materialize();
    const bool non_blocking = device && host_tensor.is_pinned();
    return host_tensor.to(dev, host_tensor.scalar_type(), non_blocking);
}
struct ClassificationMetrics {
    double precision = 0.0;
    double true_positive_rate = 0.0;
    double f1_score = 0.0;
};

static void print_metrics(const std::string& label, const ClassificationMetrics& metrics) {
    std::cout << "  " << label << " Precision: " << metrics.precision
              << "  TPR: " << metrics.true_positive_rate
              << "  F1: " << metrics.f1_score << "\n";
}

static ClassificationMetrics compute_macro_metrics(const std::vector<int64_t>& tp,
                                                   const std::vector<int64_t>& fp,
                                                   const std::vector<int64_t>& fn) {
    ClassificationMetrics metrics;
    if (tp.empty()) {
        return metrics;
    }

    double precision_sum = 0.0;
    double recall_sum    = 0.0;
    double f1_sum        = 0.0;
    std::size_t counted_classes = 0;

    for (std::size_t i = 0; i < tp.size(); ++i) {
        const double tp_d = static_cast<double>(tp[i]);
        const double fp_d = static_cast<double>(fp[i]);
        const double fn_d = static_cast<double>(fn[i]);

        const double denom_precision = tp_d + fp_d;
        const double denom_recall    = tp_d + fn_d;

        double precision = 0.0;
        double recall    = 0.0;

        if (denom_precision > 0.0)
            precision = tp_d / denom_precision;
        if (denom_recall > 0.0)
            recall = tp_d / denom_recall;

        double f1 = 0.0;
        if (precision + recall > 0.0)
            f1 = 2.0 * precision * recall / (precision + recall);

        precision_sum += precision;
        recall_sum    += recall;
        f1_sum        += f1;
        ++counted_classes;
    }

    if (counted_classes > 0) {
        metrics.precision          = precision_sum / static_cast<double>(counted_classes);
        metrics.true_positive_rate = recall_sum / static_cast<double>(counted_classes);
        metrics.f1_score           = f1_sum / static_cast<double>(counted_classes);
    }

    return metrics;
}

template <typename ForwardFn>
static ClassificationMetrics evaluate_classifier(ForwardFn&& forward_fn, const torch::Tensor& data, const torch::Tensor& labels, int64_t batch_size, bool use_cuda, int64_t num_classes) {
    std::vector<int64_t> tp(num_classes, 0);
    std::vector<int64_t> fp(num_classes, 0);
    std::vector<int64_t> fn(num_classes, 0);

    torch::NoGradGuard no_grad;
    const int64_t total = data.size(0);

    for (int64_t i = 0; i < total; i += batch_size) {
        const int64_t end = std::min(i + batch_size, total);
        auto inputs  = stage_for_device(data.index({torch::indexing::Slice(i, end)}), use_cuda);
        auto targets = stage_for_device(labels.index({torch::indexing::Slice(i, end)}), use_cuda);

        auto logits = forward_fn(inputs);
        auto preds  = logits.argmax(1);

        auto preds_cpu   = preds.to(torch::kCPU);
        auto targets_cpu = targets.to(torch::kCPU);

        const auto* preds_ptr   = preds_cpu.template data_ptr<int64_t>();
        const auto* targets_ptr = targets_cpu.template data_ptr<int64_t>();
        const int64_t local_size = preds_cpu.size(0);

        for (int64_t j = 0; j < local_size; ++j) {
            const int64_t pred   = preds_ptr[j];
            const int64_t actual = targets_ptr[j];

            if (actual >= 0 && actual < num_classes) {
                if (pred == actual) {
                    tp[actual]++;
                } else {
                    fn[actual]++;
                }
            }

            if (pred >= 0 && pred < num_classes) {
                if (pred != actual) {
                    fp[pred]++;
                }
            }
        }
    }

    return compute_macro_metrics(tp, fp, fn);
}



int main() {
    auto t1 = std::chrono::high_resolution_clock::now();
    const float freq = 1.f;
    auto [x1, y1, x2, y2] = Nott::Data::Load::MNIST("/home/moonfloww/Projects/DATASETS/Image/MNIST", freq, 1.f, true);
    const bool IsCuda = torch::cuda::is_available();
    const int64_t epochs= 10;
    const int64_t B= 64;
    const int64_t num_classes = 10;
    const int64_t N= x1.size(0);
    Nott::Data::Check::Size(x1);

    std::cout << "Setup:"
        <<"\n    Samples: " << freq*60000 << " (" << freq*100<<"%)"
        <<"\n    Epochs:  " << epochs
        <<"\n    Batchs:  " << B
        <<"\n    Steps:   " << (epochs*((N + B - 1) / B))<< std::endl;

    const int64_t steps_per_epoch = (N + B - 1) / B;
    const std::size_t total_steps_estimate = static_cast<std::size_t>(epochs * steps_per_epoch);

    std::vector<double> Nott_prebuilt_samples;
    std::vector<double> Nott_custom_samples;
    std::vector<double> libtorch_samples;

    Nott_prebuilt_samples.reserve(total_steps_estimate);
    Nott_custom_samples.reserve(total_steps_estimate);
    libtorch_samples.reserve(total_steps_estimate);

    ClassificationMetrics Nott_prebuilt_metrics;
    ClassificationMetrics Nott_custom_metrics;
    ClassificationMetrics libtorch_metrics;


    // Nott built-in train()
    std::cout << "training 100% Nott" << std::endl;
    Nott::Model model1("");
    set(model1, IsCuda); // define the network
    model1.clear_training_telemetry(); // not necessary
    model1.train(x1, y1, {.epoch = epochs, .batch_size = B, .shuffle=false, .monitor = true, .buffer_vram=2, .graph_mode = Nott::GraphMode::Disabled, .enable_amp = true, .memory_format = torch::MemoryFormat::ChannelsLast});

    const auto& telemetry = model1.training_telemetry();
    for (const auto& epoch : telemetry.epochs()) {
        const double step_latency_sec = epoch.step_latency_value();
        if (step_latency_sec <= 0.0) {
            continue;
        }

        std::size_t epoch_steps = 0;
        if (epoch.duration_seconds > 0.0) {
            const double estimated_steps = epoch.duration_seconds / step_latency_sec;
            if (estimated_steps > 0.0) {
                epoch_steps = static_cast<std::size_t>(std::llround(estimated_steps));
            }
        }
        if (epoch_steps == 0) {
            epoch_steps = static_cast<std::size_t>(std::max<int64_t>(1, steps_per_epoch));
        }

        const double step_latency_ms = step_latency_sec * 1000.0;
        for (std::size_t s = 0; s < epoch_steps; ++s) {
            Nott_prebuilt_samples.push_back(step_latency_ms);
        }
    }
    model1.eval();
    Nott_prebuilt_metrics = evaluate_classifier([&](const torch::Tensor& inputs) { return model1.forward(inputs); }, x2, y2, B, IsCuda, num_classes);


    // 50% Nott (manual training loop using Nott modelf
    std::cout << "training 50% Nott" << std::endl;
    Nott::Model model2("");
    const auto ce = set(model2, IsCuda); // define the network
    model2.clear_training_telemetry(); // not necessary

    // Pin host tensors once so all subsequent narrow()+to(CUDA, non_blocking=true) transfers
    // go through DMA without stalling the CPU | same I/O path the prebuilt runner uses.
    if (IsCuda) {
        if (!x1.is_pinned()) x1 = x1.pin_memory();
        if (!y1.is_pinned()) y1 = y1.pin_memory();
    }

    for (int64_t e = 0; e < epochs; ++e) {
        for (int64_t i = 0; i < N; i += B) {
            auto step_start = std::chrono::high_resolution_clock::now();
            const int64_t end = std::min(i + B, N);

            const bool nb = IsCuda && x1.is_pinned();
            auto inputs  = x1.narrow(0, i, end - i).to(torch::kCUDA, x1.scalar_type(), nb);
            auto targets = y1.narrow(0, i, end - i).to(torch::kCUDA, y1.scalar_type(), nb);

            model2.zero_grad();
            auto logits = model2.forward(inputs);
            auto loss   = Nott::Loss::Details::compute(ce, logits, targets);

            if (i==0) std::cout << "Epoch " << e << " loss = " << loss.item<double>() << std::endl;

            loss.backward();
            model2.step();

            auto step_end = std::chrono::high_resolution_clock::now();
            double step_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(step_end - step_start).count();
            Nott_custom_samples.push_back(step_ms);
        }
    }

    model2.eval();
    Nott_custom_metrics = evaluate_classifier(
        [&](const torch::Tensor& inputs) { return model2.forward(inputs); }, x2, y2, B, IsCuda, num_classes);


    //libtorch
    std::cout << "training 100% LibTorch" << std::endl;

    Net net;
    net->to(torch::kCUDA);
    torch::optim::SGD optimizer(net->parameters(), torch::optim::SGDOptions(1e-3f).momentum(0.9).nesterov(true));

    torch::nn::CrossEntropyLoss criterion(torch::nn::CrossEntropyLossOptions().label_smoothing(0.02f));
    net->train();

    for (int64_t e = 0; e < epochs; ++e) {
        for (int64_t i = 0; i < N; i += B) { auto step_start = std::chrono::high_resolution_clock::now();
            const int64_t end = std::min(i + B, N);

            const bool nb = IsCuda && x1.is_pinned();
            auto inputs  = x1.narrow(0, i, end - i).to(torch::kCUDA, x1.scalar_type(), nb);
            auto targets = y1.narrow(0, i, end - i).to(torch::kCUDA, y1.scalar_type(), nb);

            optimizer.zero_grad();
            auto logits = net->forward(inputs);
            auto loss = criterion(logits, targets);///static_cast<double>(B)
            if (i==0) std::cout << "Epoch " << e << " loss = " << loss.item<double>() << std::endl;
            loss.backward();
            optimizer.step();
            auto step_end = std::chrono::high_resolution_clock::now();
            double step_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(step_end - step_start).count();
            libtorch_samples.push_back(step_ms);
        }
    }
    net->eval();
    libtorch_metrics = evaluate_classifier([&](const torch::Tensor& inputs) { return net->forward(inputs); }, x2, y2, B, IsCuda, num_classes);


    // Compute stats
    std::cout << "\n\n\n";

    const LatencyUtils::StepLatencyStats Nott_prebuilt_stats =
        LatencyUtils::build_stats_from_samples("Nott Train() (Tukey 0.98)", Nott_prebuilt_samples, 500, 0.98);

    const LatencyUtils::StepLatencyStats Nott_custom_stats =
        LatencyUtils::build_stats_from_samples("Nott + Custom Train() (Tukey 0.98)", Nott_custom_samples, 500, 0.98);

    const LatencyUtils::StepLatencyStats libtorch_stats =
        LatencyUtils::build_stats_from_samples("Libtorch Raw (Tukey 0.98)", libtorch_samples, 500, 0.98);

    print_stats(Nott_prebuilt_stats);
    print_stats(Nott_custom_stats);
    print_stats(libtorch_stats);

    auto compute_overhead = [](double lhs, double rhs) {
        if (rhs == 0.0) return 0.0;
        return (lhs - rhs) / rhs * 100.0;
    };

    const double NottPreBuild = Nott_prebuilt_stats.steps_per_second();
    const double Nottcustom = Nott_custom_stats.steps_per_second();
    const double LibTorchRaw  = libtorch_stats.steps_per_second();


    std::cout << "\n\nOverhead %" << std::endl;
    std::cout << "   Nott vs Libtorch Overhead: " << compute_overhead(NottPreBuild, LibTorchRaw) << "%\n";
    std::cout << "   Nott PreBuild Train() vs Custom Train() Overhead: " << compute_overhead(NottPreBuild, Nottcustom) << "%\n";
    std::cout << "   Nott Custom Train() vs Libtorch Overhead: " << compute_overhead(Nottcustom, LibTorchRaw) << "%\n";


    std::cout << "\nClassification Metrics (macro-averaged)\n";
    print_metrics("Nott Train()", Nott_prebuilt_metrics);
    print_metrics("Nott Custom Train()", Nott_custom_metrics);
    print_metrics("LibTorch", libtorch_metrics);

    std::cout << "\n\n Total Runtime: " << std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now()-t1).count()/60.0f << "min"<< std::endl;
    return 0;
}