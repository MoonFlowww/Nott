#include "test_prelude.hpp"

#include <torch/torch.h>
#include "../src/loss/loss.hpp"

using namespace Nott;

namespace {
    void check_finite_scalar_and_backward(torch::Tensor loss, torch::Tensor pred) {
        REQUIRE(loss.defined());
        CHECK(torch::isfinite(loss).all().item<bool>());
        loss.sum().backward();
        REQUIRE(pred.grad().defined());
        CHECK(torch::isfinite(pred.grad()).all().item<bool>());
    }
}

TEST_CASE("loss: pointwise regression losses on [N,C] tensors") {
    auto pred = torch::randn({8, 4}, torch::requires_grad(true));
    auto target = torch::randn({8, 4});

    SUBCASE("MSE") { check_finite_scalar_and_backward(Loss::Details::compute(Loss::MSE(), pred, target), pred); }
    SUBCASE("MAE") { check_finite_scalar_and_backward(Loss::Details::compute(Loss::MAE(), pred, target), pred); }
    SUBCASE("SmoothL1") { check_finite_scalar_and_backward(Loss::Details::compute(Loss::SmoothL1(), pred, target), pred); }
}

TEST_CASE("loss: MSE and MAE of a tensor against itself are zero") {
    auto x = torch::randn({6, 3});
    CHECK(Loss::Details::compute(Loss::MSE(), x, x).item<double>() == doctest::Approx(0.0));
    CHECK(Loss::Details::compute(Loss::MAE(), x, x).item<double>() == doctest::Approx(0.0));
}

TEST_CASE("loss: classification losses on logits + class indices") {
    auto logits = torch::randn({8, 5}, torch::requires_grad(true));
    auto labels = torch::randint(0, 5, {8}, torch::kInt64);

    SUBCASE("CrossEntropy") {
        check_finite_scalar_and_backward(Loss::Details::compute(Loss::CrossEntropy(), logits, labels), logits);
    }
    SUBCASE("NegativeLogLikelihood on log-softmax input") {
        auto log_probs = torch::log_softmax(logits, /*dim=*/1);
        log_probs.retain_grad(); // non-leaf: needed for .grad() to populate after backward
        check_finite_scalar_and_backward(Loss::Details::compute(Loss::NegativeLogLikelihood(), log_probs, labels), log_probs);
    }
}

TEST_CASE("loss: BCEWithLogits on same-shape float target") {
    auto logits = torch::randn({8, 3}, torch::requires_grad(true));
    auto target = torch::randint(0, 2, {8, 3}).to(torch::kFloat32);
    check_finite_scalar_and_backward(Loss::Details::compute(Loss::BCEWithLogits(), logits, target), logits);
}

TEST_CASE("loss: KLDiv on log-prediction vs probability target") {
    auto logits = torch::randn({8, 5}, torch::requires_grad(true));
    auto log_pred = torch::log_softmax(logits, 1);
    log_pred.retain_grad(); // non-leaf: needed for .grad() to populate after backward
    auto target = torch::softmax(torch::randn({8, 5}), 1);
    check_finite_scalar_and_backward(Loss::Details::compute(Loss::KLDiv(), log_pred, target), log_pred);
}

TEST_CASE("loss: pair-based ranking losses on [N,2,D] tensors") {
    auto pred = torch::randn({6, 2, 4}, torch::requires_grad(true));

    SUBCASE("MarginRanking") {
        // MarginRanking compares input1/input2 elementwise, so target must match their
        // per-element shape [N, D], not just one label per sample.
        auto target = (torch::randint(0, 2, {6, 4}) * 2 - 1).to(torch::kFloat32); // {-1, 1}
        check_finite_scalar_and_backward(Loss::Details::compute(Loss::MarginRanking(), pred, target), pred);
    }
    SUBCASE("CosineEmbedding") {
        auto target = (torch::randint(0, 2, {6}) * 2 - 1).to(torch::kFloat32); // {-1, 1}, one per sample
        check_finite_scalar_and_backward(Loss::Details::compute(Loss::CosineEmbedding(), pred, target), pred);
    }
}

TEST_CASE("loss: segmentation-style losses on [N,C,H,W] probability maps") {
    auto pred = torch::rand({2, 3, 8, 8}, torch::requires_grad(true));
    auto target = torch::randint(0, 2, {2, 3, 8, 8}).to(torch::kFloat32);

    SUBCASE("Dice") { check_finite_scalar_and_backward(Loss::Details::compute(Loss::Dice(), pred, target), pred); }
    SUBCASE("Tversky") { check_finite_scalar_and_backward(Loss::Details::compute(Loss::Tversky(), pred, target), pred); }
    // Regression test for a fixed bug: LovaszSoftmaxInternal::lovasz_grad used to do an
    // in-place subtract between overlapping tensor views (`jaccard.narrow(0,1,n-1) -=
    // jaccard.narrow(0,0,n-1)`), which libtorch rejects as aliased memory. Fixed by
    // cloning the narrowed view before mutating it.
    SUBCASE("LovaszSoftmax") {
        check_finite_scalar_and_backward(Loss::Details::compute(Loss::LovaszSoftmax(), pred, target), pred);
    }
}
