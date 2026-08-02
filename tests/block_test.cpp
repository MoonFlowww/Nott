#include "test_prelude.hpp"

#include <torch/torch.h>
#include "../include/Nott.h"

using namespace Nott;

namespace {
    torch::Tensor forward_single_block(Block::Descriptor descriptor, torch::Tensor input) {
        Model model("block_smoke");
        model.add(std::move(descriptor));
        auto output = model.forward(input);
        REQUIRE(output.defined());
        CHECK(torch::isfinite(output).all().item<bool>());
        output.sum().backward();
        return output;
    }

    // Attention must only ever mix information across the sequence dimension of a single
    // sample, never across different samples in a batch. If batch and sequence got
    // confused inside attention (see the block/details/transformers/{bert,vision,
    // perceiver,longformer_xl}.hpp batch-first-vs-seq-first fixes), changing one sample's
    // *input* would leak into other samples' outputs.
    //
    // NB: a same-shape *permutation* check (swap two batch entries, expect the two
    // outputs to swap too) does NOT catch this bug class -- self-attention over any axis
    // is inherently permutation-equivariant along that axis, correct or swapped, so a
    // pure permutation looks identical either way. Perturbing one sample's *values* while
    // holding the others fixed is the property that actually distinguishes "batch axis
    // kept separate" from "batch axis silently attended over".
    void check_batch_independence(Block::Descriptor descriptor, torch::Tensor reference_input) {
        REQUIRE(reference_input.size(0) >= 2);
        Model model("block_axis_check");
        model.add(std::move(descriptor));
        model.eval(); // disable dropout so repeated forwards are deterministic

        torch::NoGradGuard guard;
        auto baseline = model.forward(reference_input);

        auto perturbed_input = reference_input.clone();
        perturbed_input[1] = torch::randn_like(perturbed_input[1]); // touch only sample 1

        auto perturbed_output = model.forward(perturbed_input);

        // Sample 0's output must be unaffected by sample 1's input changing.
        CHECK(torch::allclose(baseline[0], perturbed_output[0], 1e-4, 1e-4));
    }
}

TEST_CASE("block: Sequential chains layers end to end") {
    auto input = torch::randn({4, 8}, torch::requires_grad(true));
    auto output = forward_single_block(
        Block::Sequential({Layer::FC({8, 16, true}, Activation::ReLU), Layer::FC({16, 4, true})}), input);
    CHECK(output.sizes() == torch::IntArrayRef({4, 4}));
}

TEST_CASE("block: Residual block adds its input back when dims already match") {
    auto input = torch::randn({4, 8}, torch::requires_grad(true));
    auto output = forward_single_block(
        Block::Residual({Layer::FC({8, 8, true}, Activation::ReLU), Layer::FC({8, 8, true})}), input);
    CHECK(output.sizes() == torch::IntArrayRef({4, 8}));
}

TEST_CASE("block: Classic transformer encoder preserves [batch, seq, embed_dim]") {
    auto input = torch::randn({2, 5, 64}, torch::requires_grad(true));
    auto output = forward_single_block(
        Block::Transformer::Classic::Encoder({.layers = 1, .embed_dim = 64}), input);
    CHECK(output.sizes() == torch::IntArrayRef({2, 5, 64}));
}

// Atlas and Titans (block/details/transformers/atlas.hpp, titan.hpp) are advertised in
// docs/README.md as 2 of the "10 transformer architectures" but are empty stub headers
// (include guard + a paper citation, no code) and aren't in Block::Descriptor's variant
// or Model::add's dispatcher -- there is nothing to construct or test.
// No Atlas/Titan tests; add them once those headers have an actual implementation.

TEST_CASE("block: Classic transformer decoder preserves [batch, seq, embed_dim]") {
    auto input = torch::randn({2, 5, 64}, torch::requires_grad(true));
    auto output = forward_single_block(
        Block::Transformer::Classic::Decoder({.layers = 1, .embed_dim = 64}), input);
    CHECK(output.sizes() == torch::IntArrayRef({2, 5, 64}));
}

TEST_CASE("block: Mamba encoder preserves [batch, seq, embed_dim]") {
    auto input = torch::randn({2, 5, 64}, torch::requires_grad(true));
    auto output = forward_single_block(
        Block::Transformer::Mamba::Encoder({.layers = 1, .embed_dim = 64}), input);
    CHECK(output.sizes() == torch::IntArrayRef({2, 5, 64}));
}

TEST_CASE("block: PlusPlus transformer encoder preserves [batch, seq, embed_dim]") {
    auto input = torch::randn({2, 5, 64}, torch::requires_grad(true));
    auto output = forward_single_block(
        Block::Transformer::PlusPlus::Encoder({.layers = 1, .embed_dim = 64}), input);
    CHECK(output.sizes() == torch::IntArrayRef({2, 5, 64}));
}

TEST_CASE("block: PlusPlus transformer decoder preserves [batch, seq, embed_dim]") {
    auto input = torch::randn({2, 5, 64}, torch::requires_grad(true));
    auto output = forward_single_block(
        Block::Transformer::PlusPlus::Decoder({.layers = 1, .embed_dim = 64}), input);
    CHECK(output.sizes() == torch::IntArrayRef({2, 5, 64}));
}

TEST_CASE("block: BERT encoder preserves [batch, seq, embed_dim] and doesn't confuse batch with sequence") {
    // Regression test for a fixed bug: BertEncoderLayerImpl called libtorch's raw
    // torch::nn::MultiheadAttention (always seq-first) directly on batch-first tensors
    // with no transpose, silently swapping the batch and sequence axes inside attention.
    auto descriptor = [] {
        return Block::Transformer::Bert::Encoder(
            {.layers = 1, .embed_dim = 64, .attention = {.embed_dim = 64, .num_heads = 4}});
    };
    auto input = torch::randn({2, 5, 64}, torch::requires_grad(true));
    auto output = forward_single_block(descriptor(), input);
    CHECK(output.sizes() == torch::IntArrayRef({2, 5, 64}));
    check_batch_independence(descriptor(), input.detach());
}

TEST_CASE("block: LongformerXL encoder preserves [batch, seq, embed_dim] and doesn't confuse batch with sequence") {
    // Regression test for two fixed bugs: the additive attn_mask was passed into the
    // key_padding_mask argument slot (wrong parameter position), and -- like BERT above
    // -- the raw torch::nn::MultiheadAttention call had no batch-first/seq-first
    // transpose around it.
    auto descriptor = [] {
        return Block::Transformer::LongformerXL::Encoder(
            {.layers = 1, .embed_dim = 64, .attention = {.embed_dim = 64, .num_heads = 4}, .window_size = 8});
    };
    auto input = torch::randn({2, 16, 64}, torch::requires_grad(true));
    auto output = forward_single_block(descriptor(), input);
    CHECK(output.sizes() == torch::IntArrayRef({2, 16, 64}));
    check_batch_independence(descriptor(), input.detach());
}

TEST_CASE("block: Vision transformer encoder patch-embeds an image into a token sequence") {
    // 16x16 image, patch_size=8 -> a 2x2 grid of patches (4 tokens) + 1 class token = 5.
    // Regression test for a fixed bug: same batch-first/seq-first mismatch as BERT above,
    // in both the ViT and windowed-Swin attention branches.
    auto descriptor = [] {
        return Block::Transformer::Vision::Encoder(
            {.layers = 1,
             .embed_dim = 64,
             .attention = {.embed_dim = 64, .num_heads = 4},
             .patch_embedding = {.in_channels = 3, .embed_dim = 64, .patch_size = 8}});
    };
    auto input = torch::randn({2, 3, 16, 16}, torch::requires_grad(true));
    auto output = forward_single_block(descriptor(), input);
    CHECK(output.size(0) == 2);
    CHECK(output.size(2) == 64);
    check_batch_independence(descriptor(), input.detach());
}

TEST_CASE("block: Perceiver encoder compresses input into a fixed latent sequence") {
    // Perceiver cross-attends a long input into a small, fixed number of latents, so the
    // output sequence length is latent_slots, not the input's -- unlike the other blocks.
    // Regression test for a fixed bug: same batch-first/seq-first mismatch as BERT above,
    // in both the cross-attention and latent self-attention calls.
    auto descriptor = [] {
        return Block::Transformer::Perceiver::Encoder(
            {.layers = 1,
             .self_layers = 1,
             .latent_dim = 64,
             .input_dim = 64,
             .latent_slots = 4,
             .cross_attention = {.query_dim = 64, .key_dim = 64, .num_heads = 4},
             .self_attention = {.query_dim = 64, .key_dim = 64, .num_heads = 4}});
    };
    auto input = torch::randn({2, 20, 64}, torch::requires_grad(true));
    auto output = forward_single_block(descriptor(), input);
    CHECK(output.size(0) == 2);
    CHECK(output.size(1) == 4);
    CHECK(output.size(2) == 64);
    check_batch_independence(descriptor(), input.detach());
}

TEST_CASE("block: EBT encoder refines continuous predictions in place") {
    auto input = torch::randn({2, 8}, torch::requires_grad(true));
    auto output = forward_single_block(
        Block::Transformer::EBT::Encoder({.modality = {.type = Block::Transformer::EBT::ModalityType::Continuous,
                                                        .input_dim = 8,
                                                        .embed_dim = 8}}),
        input);
    CHECK(output.sizes() == torch::IntArrayRef({2, 8}));
}

TEST_CASE("block: EBT decoder refines continuous predictions with no context") {
    auto input = torch::randn({2, 8}, torch::requires_grad(true));
    auto output = forward_single_block(
        Block::Transformer::EBT::Decoder({.target = {.type = Block::Transformer::EBT::ModalityType::Continuous,
                                                      .input_dim = 8,
                                                      .embed_dim = 8}}),
        input);
    CHECK(output.sizes() == torch::IntArrayRef({2, 8}));
}
