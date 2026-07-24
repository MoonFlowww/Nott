// Two things measured against src/attention/details/kernel.hpp's
// ScaledDotProductKernelImpl::forward():
// 1) fast path (no mask, is_causal) vs masked path forced trivially with an all-zero
//    key_padding_mask -- both fused now, this is the cost of building/combining a mask
//    tensor and passing attn_mask instead of letting SDPA's is_causal skip it entirely.
// 2) masked path (now fused) vs a local reimplementation of the eager matmul+softmax+matmul
//    code that used to be there, with a real (non-trivial) key_padding_mask.
#include <torch/torch.h>
#include "../../include/Nott.h"

#include <chrono>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

using Nott::Attention::Details::ScaledDotProductKernel;

namespace {

struct Shape {
    std::string label;
    int64_t batch;
    int64_t heads;
    int64_t seq_len;
    int64_t head_dim;
};

double time_ms(bool is_cuda, int iterations, const std::function<torch::Tensor()> &call) {
    for (int i = 0; i < 5; ++i) {
        auto out = call();
        (void) out;
    }
    if (is_cuda) torch::cuda::synchronize();

    const auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        auto out = call();
        (void) out;
    }
    if (is_cuda) torch::cuda::synchronize();
    const auto end = std::chrono::high_resolution_clock::now();

    return std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start).count()
        / iterations;
}

// pre-fusion eager path, kept only here for comparison
torch::Tensor eager_masked_attention(const torch::Tensor &query, const torch::Tensor &key,
                                      const torch::Tensor &value, const torch::Tensor &key_padding_mask,
                                      bool causal) {
    auto scores = torch::matmul(query, key.transpose(-2, -1));
    scores = scores / std::sqrt(static_cast<double>(query.size(-1)));

    auto mask = key_padding_mask.to(torch::kBool).unsqueeze(1).unsqueeze(2);
    scores = scores.masked_fill(mask, -std::numeric_limits<float>::infinity());

    if (causal) {
        const auto seq_len = scores.size(-1);
        const auto tgt_len = scores.size(-2);
        auto causal_mask = torch::ones({tgt_len, seq_len}, scores.options().dtype(torch::kBool)).triu(1);
        scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), -std::numeric_limits<float>::infinity());
    }

    auto attn = torch::softmax(scores, -1);
    return torch::matmul(attn, value);
}

}

int main() {
    const bool is_cuda = torch::cuda::is_available();
    const auto device = is_cuda ? torch::kCUDA : torch::kCPU;
    std::cout << "device: " << (is_cuda ? "cuda" : "cpu") << "\n\n";

    const std::vector<Shape> shapes = {
        {"small  (b=8,h=4,seq=64,d=32)",    8, 4, 64, 32},
        {"medium (b=16,h=8,seq=256,d=64)", 16, 8, 256, 64},
        {"large  (b=8,h=12,seq=512,d=64)",  8, 12, 512, 64},
    };

    constexpr int kIterations = 50;

    std::cout << "-- fast path (is_causal) vs masked path forced trivially (zero mask) --\n";
    std::cout << std::left
        << std::setw(32) << "shape" << std::setw(12) << "variant"
        << std::setw(14) << "fast (ms)" << std::setw(14) << "masked (ms)" << "ratio\n";
    std::cout << std::string(84, '-') << "\n";

    for (const auto &shape: shapes) {
        for (auto variant: {Nott::Attention::Variant::Full, Nott::Attention::Variant::Causal}) {
            const auto options = torch::TensorOptions().device(device);
            auto q = torch::randn({shape.batch, shape.heads, shape.seq_len, shape.head_dim}, options);
            auto k = torch::randn({shape.batch, shape.heads, shape.seq_len, shape.head_dim}, options);
            auto v = torch::randn({shape.batch, shape.heads, shape.seq_len, shape.head_dim}, options);
            auto zero_padding_mask = torch::zeros({shape.batch, shape.seq_len}, options.dtype(torch::kBool));

            ScaledDotProductKernel kernel(/*dropout=*/0.0, variant);
            kernel->to(device);
            kernel->eval();

            const double fast_ms = time_ms(is_cuda, kIterations, [&] { return kernel->forward(q, k, v); });
            const double masked_ms = time_ms(is_cuda, kIterations,
                [&] { return kernel->forward(q, k, v, {}, zero_padding_mask); });

            const std::string variant_name = variant == Nott::Attention::Variant::Causal ? "causal" : "full";
            std::cout << std::left
                << std::setw(32) << shape.label << std::setw(12) << variant_name
                << std::setw(14) << fast_ms << std::setw(14) << masked_ms
                << (masked_ms / fast_ms) << "x\n";
        }
    }

    std::cout << "\n-- masked path: eager (pre-fusion) vs fused, real key_padding_mask --\n";
    std::cout << std::left
        << std::setw(32) << "shape" << std::setw(12) << "variant"
        << std::setw(14) << "eager (ms)" << std::setw(14) << "fused (ms)" << "speedup\n";
    std::cout << std::string(84, '-') << "\n";

    for (const auto &shape: shapes) {
        for (auto variant: {Nott::Attention::Variant::Full, Nott::Attention::Variant::Causal}) {
            const auto options = torch::TensorOptions().device(device);
            auto q = torch::randn({shape.batch, shape.heads, shape.seq_len, shape.head_dim}, options);
            auto k = torch::randn({shape.batch, shape.heads, shape.seq_len, shape.head_dim}, options);
            auto v = torch::randn({shape.batch, shape.heads, shape.seq_len, shape.head_dim}, options);
            auto key_padding_mask = torch::zeros({shape.batch, shape.seq_len}, options.dtype(torch::kBool));
            key_padding_mask.index_put_({torch::indexing::Slice(), torch::indexing::Slice(shape.seq_len / 2, torch::indexing::None)}, true);

            const bool causal = variant == Nott::Attention::Variant::Causal;
            ScaledDotProductKernel kernel(/*dropout=*/0.0, variant);
            kernel->to(device);
            kernel->eval();

            const double eager_ms = time_ms(is_cuda, kIterations,
                [&] { return eager_masked_attention(q, k, v, key_padding_mask, causal); });
            const double fused_ms = time_ms(is_cuda, kIterations,
                [&] { return kernel->forward(q, k, v, {}, key_padding_mask); });

            const std::string variant_name = causal ? "causal" : "full";
            std::cout << std::left
                << std::setw(32) << shape.label << std::setw(12) << variant_name
                << std::setw(14) << eager_ms << std::setw(14) << fused_ms
                << (eager_ms / fused_ms) << "x\n";
        }
    }

    return 0;
}
