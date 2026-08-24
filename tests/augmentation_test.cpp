#include "test_prelude.hpp"

#include <cmath>
#include <string>
#include <vector>

#include <torch/torch.h>

#include "../include/Nott.h"

using namespace Nott;

// Covers unchecked augmentation items from #13: Flip (true mirror), Cutout (masked
// region), CLAHE (contrast change), RandomBrightnessContrast (non-identity shift).

TEST_CASE("augment: Flip is a true pixel-level mirror on the requested axis") {
    // NCHW batch of two 3x4 images with a unique ramp so a wrong axis or a
    // shape-only check would still fail the allclose against torch::flip.
    auto inputs = torch::arange(0, 2 * 1 * 3 * 4, torch::kFloat32).reshape({2, 1, 3, 4});
    auto targets = torch::tensor({0, 1}, torch::kInt64);

    // frequency=1.0 selects every sample; show_progress=false keeps CI quiet.
    // Call shape matches examples/classification/images/cifar10.cpp.
    auto [augmented, aug_targets] =
        Data::Manipulation::Flip(inputs, targets, {{"x"}, 1.0, true, false});

    REQUIRE(augmented.sizes() == torch::IntArrayRef({4, 1, 3, 4}));
    REQUIRE(aug_targets.sizes() == torch::IntArrayRef({4}));

    auto flipped = augmented.slice(/*dim=*/0, /*start=*/2, /*end=*/4);
    auto expected = inputs.flip({-1});
    CHECK(torch::allclose(flipped, expected));
    // Original batch is preserved at the front of the concat.
    CHECK(torch::allclose(augmented.slice(0, 0, 2), inputs));
}

TEST_CASE("augment: Cutout zeroes the configured region at the expected location") {
    // Constant ones; frequency=1.0 so every sample is cut out and concatenated.
    auto inputs = torch::ones({2, 3, 8, 8}, torch::kFloat32);
    auto targets = torch::tensor({0, 1}, torch::kInt64);

    // offsets (y,x)=(2,3), sizes (h,w)=(2,3), fill RGB=0 (not random -1).
    auto [augmented, aug_targets] =
        Data::Manipulation::Cutout(inputs, targets, {{2, 3}, {2, 3}, {0, 0, 0}, 1.0, true, false});

    REQUIRE(augmented.sizes() == torch::IntArrayRef({4, 3, 8, 8}));
    auto cut = augmented.slice(/*dim=*/0, /*start=*/2, /*end=*/4);

    auto patch = cut.slice(/*dim=*/2, 2, 4).slice(/*dim=*/3, 3, 6);
    CHECK(torch::allclose(patch, torch::zeros_like(patch)));

    // Pixel outside the patch must remain 1 (proves we did not wipe the whole image).
    CHECK(cut.index({0, 0, 0, 0}).item<float>() == doctest::Approx(1.0f));
    CHECK(cut.index({0, 0, 7, 7}).item<float>() == doctest::Approx(1.0f));
}

TEST_CASE("augment: CLAHE changes the contrast distribution vs the input") {
    // Smooth gradient is a poor CLAHE target (already flat histogram locally);
    // a peaked tile plus a dark tile forces redistribution.
    auto inputs = torch::zeros({1, 1, 16, 16}, torch::kFloat32);
    inputs.slice(/*dim=*/2, 0, 8).slice(/*dim=*/3, 0, 8).fill_(0.9f);
    inputs.slice(/*dim=*/2, 8, 16).slice(/*dim=*/3, 8, 16).fill_(0.1f);
    auto targets = torch::tensor({0}, torch::kInt64);

    // frequency=1.0; bins=256; clip_limit>0; tile_grid 4x4.
    // Call shape mirrors the CLAHE example in dubai_segment.cpp comments.
    auto [augmented, aug_targets] =
        Data::Manipulation::CLAHE(inputs, targets, {256, 2.0, {4, 4}, 1.0, true, false});

    REQUIRE(augmented.size(0) == 2);
    auto clahe_out = augmented.slice(0, 1, 2);
    CHECK_FALSE(torch::allclose(clahe_out, inputs));
    // Stddev of intensities should move (histogram equalisation stretches contrast).
    const double in_std = inputs.std().item<double>();
    const double out_std = clahe_out.std().item<double>();
    CHECK(std::abs(out_std - in_std) > 1e-4);
}

TEST_CASE("augment: RandomBrightnessContrast produces a non-identity shift") {
    auto inputs = torch::full({2, 3, 8, 8}, 0.5, torch::kFloat32);
    auto targets = torch::tensor({0, 1}, torch::kInt64);

    // Large deltas + frequency=1.0 make identity extremely unlikely.
    auto [augmented, aug_targets] = Data::Manipulation::RandomBrightnessContrast(
        inputs, targets, {/*brightness_delta=*/0.4, /*contrast_delta=*/0.5, 1.0, true, false});

    REQUIRE(augmented.size(0) == 4);
    auto adjusted = augmented.slice(0, 2, 4);
    CHECK_FALSE(torch::allclose(adjusted, inputs));
    // Mean of adjusted batch should move away from the constant 0.5 input mean
    // (brightness/contrast both applied; either channel moves the global mean).
    const double mean_in = inputs.mean().item<double>();
    const double mean_out = adjusted.mean().item<double>();
    CHECK(std::abs(mean_out - mean_in) > 1e-4);
}