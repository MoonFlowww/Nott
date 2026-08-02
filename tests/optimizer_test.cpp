#include "test_prelude.hpp"

#include <torch/torch.h>
#include <utility>
#include "../include/Nott.h"

using namespace Nott;

namespace {
    // y = 2x + 1 plus tiny noise: a convex regression problem every optimizer should handle.
    std::pair<torch::Tensor, torch::Tensor> toy_linear_regression(int64_t n = 64) {
        auto x = torch::randn({n, 1});
        auto y = 2.0 * x + 1.0 + torch::randn({n, 1}) * 0.01;
        return {x, y};
    }

    Model make_toy_model() {
        Model model("optimizer_smoke");
        model.add(Layer::FC({1, 8, true}, Activation::ReLU));
        model.add(Layer::FC({8, 1, true}));
        return model;
    }

    template <class OptimizerDescriptor>
    std::vector<double> train_and_collect_losses(const OptimizerDescriptor& descriptor, std::size_t epochs) {
        auto model = make_toy_model();
        model.set_optimizer(descriptor);
        model.set_loss(Loss::MSE());

        auto [x, y] = toy_linear_regression();
        TrainOptions options{};
        options.epoch = epochs;
        options.batch_size = 64;
        options.monitor = false;

        model.train(x, y, options);

        std::vector<double> losses;
        for (const auto& epoch : model.training_telemetry().epochs()) {
            losses.push_back(epoch.train_loss_value());
        }
        return losses;
    }
}

TEST_CASE("optimizer: every registered optimizer trains without diverging to NaN/Inf") {
    auto check_smoke = [](auto descriptor) {
        auto losses = train_and_collect_losses(descriptor, /*epochs=*/5);
        REQUIRE(losses.size() == 5);
        for (double loss : losses) {
            CHECK(std::isfinite(loss));
        }
    };

    SUBCASE("SGD") { check_smoke(Optimizer::SGD()); }
    SUBCASE("RMSprop") { check_smoke(Optimizer::RMSprop()); }
    SUBCASE("Adagrad") { check_smoke(Optimizer::Adagrad()); }
    SUBCASE("Adam") { check_smoke(Optimizer::Adam()); }
    SUBCASE("AdamW") { check_smoke(Optimizer::AdamW()); }
    SUBCASE("Lion") { check_smoke(Optimizer::Lion()); }
    SUBCASE("LAMB") { check_smoke(Optimizer::LAMB()); }
    SUBCASE("Adafactor") { check_smoke(Optimizer::Adafactor()); }
    SUBCASE("SophiaG") { check_smoke(Optimizer::SophiaG()); }
    SUBCASE("SophiaH") { check_smoke(Optimizer::SophiaH()); }
    SUBCASE("Muon") { check_smoke(Optimizer::Muon()); }
    SUBCASE("AdaMuon") { check_smoke(Optimizer::AdaMuon()); }
    SUBCASE("MuonManifold") { check_smoke(Optimizer::MuonManifold()); }
}

TEST_CASE("optimizer: well-understood optimizers actually reduce loss on the toy problem") {
    auto check_converges = [](auto descriptor) {
        auto losses = train_and_collect_losses(descriptor, /*epochs=*/100);
        REQUIRE(losses.size() == 100);
        CHECK(losses.back() < losses.front());
        CHECK(std::isfinite(losses.back()));
    };

    SUBCASE("SGD") { check_converges(Optimizer::SGD({.learning_rate = 0.1})); }
    SUBCASE("Adam") { check_converges(Optimizer::Adam({.learning_rate = 0.05})); }
    SUBCASE("AdamW") { check_converges(Optimizer::AdamW({.learning_rate = 0.05})); }
    SUBCASE("RMSprop") { check_converges(Optimizer::RMSprop({.learning_rate = 0.05})); }
    SUBCASE("Adagrad") { check_converges(Optimizer::Adagrad({.learning_rate = 0.3})); }
}

// The eight below were previously smoke-tested only (finite loss, no NaN), so a
// silently non-learning optimizer would still have passed. These assert the
// loss actually comes down. Learning rates are set per optimizer because their
// natural scales differ; Lion in particular needs a much smaller step than SGD.
TEST_CASE("optimizer: the remaining optimizers also reduce loss on the toy problem") {
    auto check_converges = [](auto descriptor) {
        auto losses = train_and_collect_losses(descriptor, /*epochs=*/100);
        REQUIRE(losses.size() == 100);
        CHECK(std::isfinite(losses.back()));
        CHECK(losses.back() < losses.front());
    };

    SUBCASE("Lion") { check_converges(Optimizer::Lion(Optimizer::LionOptions().lr(0.01))); }
    SUBCASE("LAMB") { check_converges(Optimizer::LAMB(Optimizer::LAMBOptions().lr(0.05))); }
    SUBCASE("Adafactor") { check_converges(Optimizer::Adafactor(Optimizer::AdafactorOptions().lr(0.05))); }
    SUBCASE("SophiaG") { check_converges(Optimizer::SophiaG(Optimizer::SophiaGOptions().lr(0.05))); }
    SUBCASE("SophiaH") { check_converges(Optimizer::SophiaH(Optimizer::SophiaHOptions().lr(0.05))); }
    SUBCASE("Muon") { check_converges(Optimizer::Muon(Optimizer::MuonOptions().lr(0.05))); }
    SUBCASE("AdaMuon") { check_converges(Optimizer::AdaMuon(Optimizer::AdaMuonOptions().lr(0.05))); }
    SUBCASE("MuonManifold") {
        check_converges(Optimizer::MuonManifold(Optimizer::MuonManifoldOptions().lr(0.05)));
    }
}
