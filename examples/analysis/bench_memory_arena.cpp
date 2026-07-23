// peak CUDA memory for execute_plan()'s compiled-routing path, early buffer release on/off
#include <torch/torch.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include "../../include/Nott.h"

#include <functional>
#include <iostream>

using namespace Nott;

namespace {

// chained via links() so compiled routing engages, not the plain-Sequential fallback
Model build_chain(std::size_t depth, int64_t width) {
    Model model("bench_chain");
    std::vector<LinkSpec> specs;
    for (std::size_t i = 0; i < depth; ++i) {
        model.add(Layer::FC({width, width, true}, Activation::ReLU), "layer_" + std::to_string(i));
    }
    specs.push_back(LinkSpec{Port::Input("@input"), Port::Module("layer_0")});
    for (std::size_t i = 1; i < depth; ++i) {
        specs.push_back(LinkSpec{Port::Module("layer_" + std::to_string(i - 1)), Port::Module("layer_" + std::to_string(i))});
    }
    specs.push_back(LinkSpec{Port::Module("layer_" + std::to_string(depth - 1)), Port::Output("@output")});
    model.links(std::move(specs), false);
    return model;
}

}

double peak_mb_of(const std::function<void()> &call) {
    torch::cuda::synchronize();
    c10::cuda::CUDACachingAllocator::resetPeakStats(0);
    call();
    torch::cuda::synchronize();
    const auto stats = c10::cuda::CUDACachingAllocator::getDeviceStats(0);
    return static_cast<double>(stats.allocated_bytes[0].peak) / (1024.0 * 1024.0);
}

int main() {
    if (!torch::cuda::is_available()) {
        std::cout << "CUDA not available; peak memory stats need a CUDA device.\n";
        return 0;
    }

    constexpr int64_t kWidth = 2048;
    constexpr int64_t kBatch = 256;
    constexpr std::size_t kDepth = 40;

    auto input = torch::randn({kBatch, kWidth}, torch::TensorOptions().device(torch::kCUDA));

    // no_grad isolates our own release logic from autograd's references
    {
        auto model = build_chain(kDepth, kWidth);
        model.use_cuda(true);
        model.eval();
        torch::NoGradGuard no_grad;
        for (int i = 0; i < 3; ++i) { auto out = model.forward(input); (void) out; }
        const double peak_mb = peak_mb_of([&] { auto out = model.forward(input); (void) out; });
        std::cout << "inference (no_grad): peak " << peak_mb << " MB\n";
    }

    // measures how much (if anything) release buys on top of autograd's own references
    {
        auto model = build_chain(kDepth, kWidth);
        model.use_cuda(true);
        model.set_loss(Nott::Loss::MSE());
        model.set_optimizer(Nott::Optimizer::SGD({.learning_rate = 1e-3}));
        model.train(true);
        auto target = torch::randn({kBatch, kWidth}, torch::TensorOptions().device(torch::kCUDA));
        for (int i = 0; i < 3; ++i) {
            auto out = model.forward(input);
            auto loss = model.compute_loss(out, target);
            loss.backward();
            model.zero_grad();
        }
        const double peak_mb = peak_mb_of([&] {
            auto out = model.forward(input);
            auto loss = model.compute_loss(out, target);
            loss.backward();
            model.zero_grad();
        });
        std::cout << "training (forward+backward): peak " << peak_mb << " MB\n";
    }

    return 0;
}
