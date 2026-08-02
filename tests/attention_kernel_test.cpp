#include "test_prelude.hpp"

#include <torch/torch.h>
#include <limits>
#include "../include/Nott.h"

using namespace Nott;
using Nott::Attention::Details::ScaledDotProductKernel;

// forward() always ends up calling the fused at::scaled_dot_product_attention now (see
// kernel.hpp): the no-mask path uses is_causal directly, the masked path folds
// key_padding_mask/attn_mask/causal into one additive mask first. These tests check the
// masked path is semantically correct against ground truth (a masked key must have zero
// influence on the output) rather than against the eager code that used to be there.

TEST_CASE("attention kernel: key_padding_mask makes masked keys have zero influence") {
    torch::manual_seed(0);
    ScaledDotProductKernel kernel(/*dropout=*/0.0, Attention::Variant::Full);
    kernel->eval();

    const auto query = torch::randn({2, 3, 5, 8});
    const auto key = torch::randn({2, 3, 5, 8});
    auto value = torch::randn({2, 3, 5, 8});

    auto key_padding_mask = torch::zeros({2, 5}, torch::kBool);
    key_padding_mask.index_put_({torch::indexing::Slice(), 2}, true); // mask out key position 2

    auto output_before = kernel->forward(query, key, value, {}, key_padding_mask);

    auto value_perturbed = value.clone();
    value_perturbed.index_put_({torch::indexing::Slice(), torch::indexing::Slice(), 2, torch::indexing::Slice()},
        torch::randn({2, 3, 8}) * 1000.0);
    auto output_after = kernel->forward(query, key, value_perturbed, {}, key_padding_mask);

    CHECK(torch::allclose(output_before, output_after, /*rtol=*/1e-5, /*atol=*/1e-6));
}

TEST_CASE("attention kernel: additive attn_mask makes masked position have zero influence") {
    torch::manual_seed(1);
    ScaledDotProductKernel kernel(/*dropout=*/0.0, Attention::Variant::Full);
    kernel->eval();

    const auto query = torch::randn({2, 3, 5, 8});
    const auto key = torch::randn({2, 3, 5, 8});
    auto value = torch::randn({2, 3, 5, 8});

    auto attn_mask = torch::zeros({5, 5});
    attn_mask.index_put_({torch::indexing::Slice(), 3}, -std::numeric_limits<float>::infinity());

    auto output_before = kernel->forward(query, key, value, attn_mask);

    auto value_perturbed = value.clone();
    value_perturbed.index_put_({torch::indexing::Slice(), torch::indexing::Slice(), 3, torch::indexing::Slice()},
        torch::randn({2, 3, 8}) * 1000.0);
    auto output_after = kernel->forward(query, key, value_perturbed, attn_mask);

    CHECK(torch::allclose(output_before, output_after, /*rtol=*/1e-5, /*atol=*/1e-6));
}

TEST_CASE("attention kernel: Causal + key_padding_mask matches Causal fast path when nothing is padded") {
    torch::manual_seed(2);
    ScaledDotProductKernel kernel(/*dropout=*/0.0, Attention::Variant::Causal);
    kernel->eval();

    const auto query = torch::randn({2, 3, 5, 8});
    const auto key = torch::randn({2, 3, 5, 8});
    const auto value = torch::randn({2, 3, 5, 8});

    auto no_padding_mask = torch::zeros({2, 5}, torch::kBool);

    auto fast_path_output = kernel->forward(query, key, value);
    auto masked_path_output = kernel->forward(query, key, value, {}, no_padding_mask);

    CHECK(torch::allclose(fast_path_output, masked_path_output, /*rtol=*/1e-5, /*atol=*/1e-6));
}

TEST_CASE("attention kernel: Causal variant actually masks future positions") {
    torch::manual_seed(2);
    ScaledDotProductKernel kernel(/*dropout=*/0.0, Attention::Variant::Causal);
    kernel->eval();

    const auto query = torch::randn({1, 1, 4, 8});
    const auto key = torch::randn({1, 1, 4, 8});
    const auto value = torch::randn({1, 1, 4, 8});

    auto causal_output = kernel->forward(query, key, value);

    ScaledDotProductKernel full_kernel(/*dropout=*/0.0, Attention::Variant::Full);
    full_kernel->eval();
    auto full_output = full_kernel->forward(query, key, value);

    CHECK_FALSE(torch::allclose(causal_output, full_output));
    CHECK(torch::isfinite(causal_output).all().item<bool>());
}
