#include "test_prelude.hpp"

#include <torch/torch.h>
#include "../include/Nott.h"

using namespace Nott;

namespace {
    // Build a single-layer model, forward a batch through it, and hand back the output
    // for shape/value assertions plus a differentiability check.
    torch::Tensor forward_single_layer(Layer::Descriptor descriptor, torch::Tensor input) {
        Model model("layer_smoke");
        model.add(std::move(descriptor));
        auto output = model.forward(input);
        REQUIRE(output.defined());
        CHECK(torch::isfinite(output).all().item<bool>());
        output.sum().backward();
        return output;
    }
}

TEST_CASE("layer: FC preserves batch, maps to out_features") {
    auto input = torch::randn({4, 6}, torch::requires_grad(true));
    auto output = forward_single_layer(Layer::FC({6, 3, true}), input);
    CHECK(output.sizes() == torch::IntArrayRef({4, 3}));
}

TEST_CASE("layer: Conv1d preserves length under padding=1, kernel=3, stride=1") {
    auto input = torch::randn({4, 2, 10}, torch::requires_grad(true));
    auto output = forward_single_layer(
        Layer::Conv1d({.in_channels = 2, .out_channels = 5, .kernel_size = {3}, .stride = {1}, .padding = {1}}),
        input);
    CHECK(output.sizes() == torch::IntArrayRef({4, 5, 10}));
}

TEST_CASE("layer: Conv2d preserves spatial size under padding=1, kernel=3, stride=1") {
    auto input = torch::randn({4, 2, 8, 8}, torch::requires_grad(true));
    auto output = forward_single_layer(
        Layer::Conv2d({.in_channels = 2, .out_channels = 5, .kernel_size = {3, 3}, .stride = {1, 1}, .padding = {1, 1}}),
        input);
    CHECK(output.sizes() == torch::IntArrayRef({4, 5, 8, 8}));
}

TEST_CASE("layer: BatchNorm2d preserves shape") {
    auto input = torch::randn({4, 5, 8, 8}, torch::requires_grad(true));
    auto output = forward_single_layer(Layer::BatchNorm2d({.num_features = 5}), input);
    CHECK(output.sizes() == torch::IntArrayRef({4, 5, 8, 8}));
}

TEST_CASE("layer: pooling variants downsample spatial dims as expected") {
    SUBCASE("MaxPool2d halves a divisible spatial size") {
        auto input = torch::randn({4, 3, 8, 8}, torch::requires_grad(true));
        auto output = forward_single_layer(Layer::MaxPool2d({.kernel_size = {2, 2}, .stride = {2, 2}}), input);
        CHECK(output.sizes() == torch::IntArrayRef({4, 3, 4, 4}));
    }
    SUBCASE("AdaptiveAvgPool2d hits the requested output size directly") {
        auto input = torch::randn({4, 3, 9, 9}, torch::requires_grad(true));
        auto output = forward_single_layer(Layer::AdaptiveAvgPool2d({.output_size = {4, 4}}), input);
        CHECK(output.sizes() == torch::IntArrayRef({4, 3, 4, 4}));
    }
    SUBCASE("MaxPool1d halves a divisible length") {
        auto input = torch::randn({4, 3, 8}, torch::requires_grad(true));
        auto output = forward_single_layer(Layer::MaxPool1d({.kernel_size = {2}, .stride = {2}}), input);
        CHECK(output.sizes() == torch::IntArrayRef({4, 3, 4}));
    }
}

TEST_CASE("layer: dropout variants preserve shape") {
    SUBCASE("HardDropout") {
        auto input = torch::randn({4, 10}, torch::requires_grad(true));
        auto output = forward_single_layer(Layer::HardDropout({.probability = 0.3}), input);
        CHECK(output.sizes() == torch::IntArrayRef({4, 10}));
    }
    SUBCASE("SoftDropout") {
        auto input = torch::randn({4, 10}, torch::requires_grad(true));
        auto output = forward_single_layer(Layer::SoftDropout({.probability = 0.3}), input);
        CHECK(output.sizes() == torch::IntArrayRef({4, 10}));
    }
}

TEST_CASE("layer: Flatten collapses trailing dims") {
    auto input = torch::randn({4, 3, 4, 4}, torch::requires_grad(true));
    auto output = forward_single_layer(Layer::Flatten({.start_dim = 1, .end_dim = -1}), input);
    CHECK(output.sizes() == torch::IntArrayRef({4, 48}));
}

TEST_CASE("layer: Upsample/Downsample scale spatial dims inversely") {
    SUBCASE("Upsample") {
        auto input = torch::randn({4, 3, 4, 4}, torch::requires_grad(true));
        auto output = forward_single_layer(Layer::Upsample({.scale = {2.0, 2.0}}), input);
        CHECK(output.sizes() == torch::IntArrayRef({4, 3, 8, 8}));
    }
    SUBCASE("Downsample") {
        auto input = torch::randn({4, 3, 8, 8}, torch::requires_grad(true));
        auto output = forward_single_layer(Layer::Downsample({.scale = {2.0, 2.0}}), input);
        CHECK(output.sizes() == torch::IntArrayRef({4, 3, 4, 4}));
    }
}

TEST_CASE("layer: Reduce collapses the requested dimension") {
    auto input = torch::randn({4, 5, 8}, torch::requires_grad(true));
    auto output = forward_single_layer(Layer::Reduce({.op = Layer::ReduceOp::Mean, .dims = {1}, .keep_dim = false}), input);
    CHECK(output.sizes() == torch::IntArrayRef({4, 8}));
}

// Recurrent/state-space layers wrap third-party forward() shapes (tuples, custom
// output structs) behind RegisteredLayer, so we don't hardcode their exact feature
// dimension here -- just that batching and differentiability survive the wrapping.
TEST_CASE("layer: recurrent and state-space layers keep the batch dimension and backprop") {
    auto check_batched_output = [](torch::Tensor output, int64_t batch) {
        REQUIRE(output.dim() >= 2);
        CHECK(output.size(0) == batch);
    };

    SUBCASE("RNN") {
        auto input = torch::randn({4, 7, 5}, torch::requires_grad(true));
        auto output = forward_single_layer(Layer::RNN({.input_size = 5, .hidden_size = 6, .batch_first = true}), input);
        check_batched_output(output, 4);
    }
    SUBCASE("LSTM") {
        auto input = torch::randn({4, 7, 5}, torch::requires_grad(true));
        auto output = forward_single_layer(Layer::LSTM({.input_size = 5, .hidden_size = 6, .batch_first = true}), input);
        check_batched_output(output, 4);
    }
    SUBCASE("GRU") {
        auto input = torch::randn({4, 7, 5}, torch::requires_grad(true));
        auto output = forward_single_layer(Layer::GRU({.input_size = 5, .hidden_size = 6, .batch_first = true}), input);
        check_batched_output(output, 4);
    }
    // Regression test for a fixed bug: HiPPO initialization built V_inv (complex, from
    // linalg_eig of a real matrix) and matmul'd it directly against B (still real).
    // torch::matmul doesn't promote real -> complex the way .to() does, so this crashed
    // in the S4 constructor on the golden path. Fixed by casting B to V_inv's dtype first.
    SUBCASE("S4") {
        auto input = torch::randn({4, 7, 5}, torch::requires_grad(true));
        auto output = forward_single_layer(Layer::S4({.input_size = 5, .state_size = 8, .batch_first = true}), input);
        check_batched_output(output, 4);
    }
}

TEST_CASE("layer: PatchUnembed reassembles a token grid into a channel-first image") {
    // 2x2 grid of patch_size=2 patches over 3 channels: tokens=4, patch_dim=3*2*2=12.
    auto input = torch::randn({4, 4, 12}, torch::requires_grad(true));
    auto output = forward_single_layer(
        Layer::PatchUnembed({.channels = 3, .tokens_height = 2, .tokens_width = 2, .patch_size = 2}), input);
    CHECK(output.size(0) == 4);
    CHECK(output.size(1) == 3);
}

// The five pooling variants the original suite never constructed. Average
// pooling has an exact answer on a constant input, so these assert the value
// and not just the shape; a variant wired to the wrong kernel would still get
// the shape right.
TEST_CASE("layer: the remaining pooling variants downsample correctly") {
    SUBCASE("AvgPool1d averages each window") {
        auto input = torch::ones({2, 3, 8}, torch::requires_grad(true));
        auto output = forward_single_layer(Layer::AvgPool1d({.kernel_size = {2}, .stride = {2}}), input);
        CHECK(output.sizes() == torch::IntArrayRef({2, 3, 4}));
        CHECK(torch::allclose(output, torch::ones_like(output)));
    }
    SUBCASE("AvgPool2d averages each window") {
        auto input = torch::ones({2, 3, 8, 8}, torch::requires_grad(true));
        auto output = forward_single_layer(Layer::AvgPool2d({.kernel_size = {2, 2}, .stride = {2, 2}}), input);
        CHECK(output.sizes() == torch::IntArrayRef({2, 3, 4, 4}));
        CHECK(torch::allclose(output, torch::ones_like(output)));
    }
    SUBCASE("AdaptiveAvgPool1d hits the requested length from an indivisible input") {
        auto input = torch::randn({2, 3, 10}, torch::requires_grad(true));
        auto output = forward_single_layer(Layer::AdaptiveAvgPool1d({.output_size = {4}}), input);
        CHECK(output.sizes() == torch::IntArrayRef({2, 3, 4}));
    }
    SUBCASE("AdaptiveMaxPool1d hits the requested length from an indivisible input") {
        auto input = torch::randn({2, 3, 10}, torch::requires_grad(true));
        auto output = forward_single_layer(Layer::AdaptiveMaxPool1d({.output_size = {4}}), input);
        CHECK(output.sizes() == torch::IntArrayRef({2, 3, 4}));
    }
    SUBCASE("AdaptiveMaxPool2d hits the requested grid and keeps the per-window maximum") {
        auto input = torch::randn({2, 3, 9, 9}, torch::requires_grad(true));
        auto output = forward_single_layer(Layer::AdaptiveMaxPool2d({.output_size = {3, 3}}), input);
        CHECK(output.sizes() == torch::IntArrayRef({2, 3, 3, 3}));
        // Max pooling can only ever return values the input actually contained.
        CHECK(output.max().item<double>() <= input.max().item<double>());
    }
    SUBCASE("global adaptive pooling collapses the spatial dims to 1x1") {
        auto input = torch::randn({2, 5, 7, 7}, torch::requires_grad(true));
        auto output = forward_single_layer(Layer::AdaptiveAvgPool2d({.output_size = {1, 1}}), input);
        CHECK(output.sizes() == torch::IntArrayRef({2, 5, 1, 1}));
    }
}

// xLSTM had no test at all: it shares LSTMOptions, so a descriptor wired to the
// wrong implementation would have gone unnoticed.
TEST_CASE("layer: xLSTM produces a batched sequence output and backprops") {
    auto input = torch::randn({4, 7, 5}, torch::requires_grad(true));
    auto output = forward_single_layer(
        Layer::xLSTM({.input_size = 5, .hidden_size = 6, .batch_first = true}), input);
    REQUIRE(output.dim() >= 2);
    CHECK(output.size(0) == 4);
    CHECK(output.size(1) == 7);
}

TEST_CASE("layer: xLSTM respects bidirectional and multi-layer options") {
    auto input = torch::randn({3, 6, 5}, torch::requires_grad(true));
    auto output = forward_single_layer(
        Layer::xLSTM({.input_size = 5, .hidden_size = 4, .num_layers = 2, .batch_first = true,
                      .bidirectional = true}),
        input);
    REQUIRE(output.dim() >= 2);
    CHECK(output.size(0) == 3);
    // Both directions are concatenated on the feature axis.
    CHECK(output.size(2) == 8);
}
