// Optimizer-step microbench: naive per-tensor Adam (one set of small kernels per param,
// what torch::optim::Adam does in C++) vs a _foreach_ batched step (one kernel per op
// across all params). Isolates the step math from forward/backward to bound how much a
// fused/foreach optimizer could save, and prints the step's kernel time so we can weigh
// it against a real train iteration before committing to implement it in the framework.
#include <torch/torch.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAEvent.h>

#include <chrono>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <vector>

namespace {

// two regimes: "large" = few big tensors, bandwidth-bound (what a conv/FC net's weights
// look like); "tiny" = many small tensors, launch-bound (many small layers/biases) where
// per-tensor kernel launch overhead dominates and foreach can actually help.
std::vector<std::vector<int64_t>> param_shapes(bool tiny) {
    std::vector<std::vector<int64_t>> s;
    if (tiny) {
        for (int i = 0; i < 2000; ++i) s.push_back({256});     // tons of small params
        for (int i = 0; i < 400; ++i) s.push_back({64, 64});   // small weights
        return s;
    }
    for (int i = 0; i < 60; ++i) s.push_back({1024, 1024});   // big weights
    for (int i = 0; i < 60; ++i) s.push_back({2048, 512});    // more weights
    for (int i = 0; i < 120; ++i) s.push_back({1024});        // biases / norms
    for (int i = 0; i < 120; ++i) s.push_back({512});         // small params
    return s;
}

struct AdamCfg { double lr = 1e-3, b1 = 0.9, b2 = 0.999, eps = 1e-8; };

// one naive Adam step, per-tensor, using ordinary ops (mirrors the C++ optimizer loop)
void adam_naive(std::vector<torch::Tensor>& p, std::vector<torch::Tensor>& g,
                std::vector<torch::Tensor>& m, std::vector<torch::Tensor>& v,
                int64_t step, const AdamCfg& c) {
    const double bc1 = 1.0 - std::pow(c.b1, step);
    const double bc2 = 1.0 - std::pow(c.b2, step);
    const double step_size = c.lr / bc1;
    for (size_t i = 0; i < p.size(); ++i) {
        m[i].mul_(c.b1).add_(g[i], 1.0 - c.b1);
        v[i].mul_(c.b2).addcmul_(g[i], g[i], 1.0 - c.b2);
        auto denom = v[i].sqrt().div_(std::sqrt(bc2)).add_(c.eps);
        p[i].addcdiv_(m[i], denom, -step_size);
    }
}

// same math, batched across all tensors with _foreach_ ops (few kernels total)
void adam_foreach(std::vector<torch::Tensor>& p, std::vector<torch::Tensor>& g,
                  std::vector<torch::Tensor>& m, std::vector<torch::Tensor>& v,
                  int64_t step, const AdamCfg& c) {
    const double bc1 = 1.0 - std::pow(c.b1, step);
    const double bc2 = 1.0 - std::pow(c.b2, step);
    const double step_size = c.lr / bc1;
    at::_foreach_mul_(m, c.b1);
    at::_foreach_add_(m, g, 1.0 - c.b1);
    at::_foreach_mul_(v, c.b2);
    at::_foreach_addcmul_(v, g, g, 1.0 - c.b2);
    auto denom = at::_foreach_sqrt(v);
    at::_foreach_div_(denom, std::sqrt(bc2));
    at::_foreach_add_(denom, c.eps);
    at::_foreach_addcdiv_(p, m, denom, -step_size);
}

std::vector<torch::Tensor> make(const std::vector<std::vector<int64_t>>& shapes,
                                const torch::TensorOptions& o, bool randn) {
    std::vector<torch::Tensor> t;
    for (auto& s : shapes) t.push_back(randn ? torch::randn(s, o) : torch::zeros(s, o));
    return t;
}

double time_ms(std::function<void()> body, int iters, bool cuda) {
    at::cuda::CUDAEvent start(cudaEventDefault), stop(cudaEventDefault);
    if (cuda) start.record();
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) body();
    if (cuda) { stop.record(); torch::cuda::synchronize(); return start.elapsed_time(stop) / iters; }
    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
}

}

int main() {
    const bool cuda = torch::cuda::is_available();
    auto opts = torch::TensorOptions().dtype(torch::kFloat32)
        .device(cuda ? torch::kCUDA : torch::kCPU);
    torch::NoGradGuard ng;

  for (bool tiny : {false, true}) {
    auto shapes = param_shapes(tiny);
    int64_t total = 0; for (auto& s : shapes) { int64_t n = 1; for (auto d : s) n *= d; total += n; }
    std::cout << "\n[" << (tiny ? "tiny/launch-bound" : "large/bandwidth-bound") << "]"
              << " device: " << (cuda ? "cuda" : "cpu")
              << " | tensors: " << shapes.size()
              << " | params: " << total / 1000000 << "M\n";

    auto grads = make(shapes, opts, true);

    auto bench = [&](const char* label, void(*fn)(std::vector<torch::Tensor>&,
                     std::vector<torch::Tensor>&, std::vector<torch::Tensor>&,
                     std::vector<torch::Tensor>&, int64_t, const AdamCfg&)) {
        torch::manual_seed(99);  // same param init across both benches so checksums compare
        auto p = make(shapes, opts, true);
        auto m = make(shapes, opts, false);
        auto v = make(shapes, opts, false);
        int64_t step = 0;
        auto body = [&]() { ++step; fn(p, grads, m, v, step, AdamCfg{}); };
        for (int i = 0; i < 20; ++i) body();
        if (cuda) torch::cuda::synchronize();
        double ms = time_ms(body, 200, cuda);
        std::cout << std::left << std::setw(10) << label
                  << " step " << std::fixed << std::setprecision(4) << ms << " ms"
                  << " | checksum " << std::setprecision(6) << p[0].abs().sum().item<double>() << "\n";
        return ms;
    };

    double n = bench("naive", adam_naive);
    double f = bench("foreach", adam_foreach);
    std::cout << "foreach speedup on the step: " << std::setprecision(2) << (n / f) << "x\n";
  }
    return 0;
}
