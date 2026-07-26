// Transfer-bound training bench: cheap model, large CPU-pinned inputs, so the per-batch
// H2D copy is a real fraction of the step. Isolates the input-prefetch overlap: with the
// copy issued on the prefetch stream it hides behind compute; without, it serializes.
#include "../../include/Nott.h"

#include <torch/torch.h>
#include <chrono>
#include <iomanip>
#include <iostream>

using namespace Nott;

namespace {

constexpr int64_t kSamples = 1024;
constexpr int64_t kClasses = 10;
constexpr int64_t kChannels = 3;
constexpr int64_t kHW = 256;   // big spatial -> big H2D per batch
constexpr std::size_t kEpochs = 12;
constexpr std::size_t kWarmup = 4;
constexpr std::size_t kBatch = 64;

// light model: two small convs then global pool. Compute is modest so the H2D copy is not dwarfed.
Model build_light() {
    Model m("light");
    m.add(Layer::Conv2d({kChannels, 8, {3, 3}, {1, 1}, {1, 1}}, Activation::ReLU, Initialization::HeUniform));
    m.add(Layer::Conv2d({8, 8, {3, 3}, {1, 1}, {1, 1}}, Activation::ReLU, Initialization::HeUniform));
    m.add(Layer::AdaptiveAvgPool2d({.output_size = {1, 1}}));
    m.add(Layer::Flatten());
    m.add(Layer::FC({8, kClasses}, Activation::Identity, Initialization::XavierUniform));
    return m;
}

}

int main() {
    const bool use_cuda = torch::cuda::is_available();
    std::cout << "device: " << (use_cuda ? "cuda" : "cpu")
              << " | per-batch H2D approx "
              << (kBatch * kChannels * kHW * kHW * 4) / (1024 * 1024) << " MB\n";

    torch::manual_seed(11);
    auto inputs = torch::randn({kSamples, kChannels, kHW, kHW});   // stays on CPU, train() pins it
    auto targets = torch::randint(0, kClasses, {kSamples}, torch::TensorOptions().dtype(torch::kLong));

    auto m = build_light();
    m.use_cuda(use_cuda);
    m.set_loss(Loss::CrossEntropy({}));
    m.set_optimizer(Optimizer::Adam({.learning_rate = 1e-3}));

    TrainOptions o{};
    o.epoch = kEpochs;
    o.batch_size = kBatch;
    o.shuffle = false;
    o.monitor = false;

    m.train(inputs, targets, o);

    const auto &ep = m.training_telemetry().epochs();
    double sum = 0.0; std::size_t n = 0;
    for (std::size_t i = 0; i < ep.size(); ++i) if (i >= kWarmup) { sum += ep[i].duration_seconds * 1000.0; ++n; }
    std::cout << std::fixed << std::setprecision(2)
              << "steady ms/epoch " << (n ? sum / n : 0.0)
              << " | last_loss " << std::setprecision(4) << ep.back().train_loss_value() << "\n";
    return 0;
}
