#include "test_prelude.hpp"

#include <torch/torch.h>
#include "../include/Nott.h"

using namespace Nott;

// graph capture is not supported yet: requesting it must fail loudly, not run silently

TEST_CASE("graph capture: inference capture request throws") {
    if (!torch::cuda::is_available()) {
        return;
    }

    Model model("graph_capture_inference");
    model.add(Layer::FC({4, 3, true}, Activation::ReLU), "fc");
    model.links({
        LinkSpec{Port::Input("@input"), Port::Module("fc")},
        LinkSpec{Port::Module("fc"), Port::Output("@output")},
    }, /*enable_graph_capture=*/true);
    model.use_cuda(true);
    model.eval();

    auto input = torch::randn({2, 4}, torch::TensorOptions().device(torch::kCUDA));

    ForwardOptions options{};
    options.graph_mode = GraphMode::Capture;

    torch::NoGradGuard no_grad;
    CHECK_THROWS(model.forward(input, options));
}

TEST_CASE("graph capture: training capture request throws") {
    if (!torch::cuda::is_available()) {
        return;
    }

    Model model("graph_capture_training");
    model.add(Layer::FC({4, 3, true}, Activation::ReLU), "fc");
    model.links({
        LinkSpec{Port::Input("@input"), Port::Module("fc")},
        LinkSpec{Port::Module("fc"), Port::Output("@output")},
    }, /*enable_graph_capture=*/true);
    model.use_cuda(true);
    model.set_loss(Loss::MSE());
    model.set_optimizer(Optimizer::SGD({.learning_rate = 0.05}));

    auto inputs = torch::randn({16, 4}, torch::TensorOptions().device(torch::kCUDA));
    auto targets = torch::randn({16, 3}, torch::TensorOptions().device(torch::kCUDA));

    TrainOptions options{};
    options.epoch = 5;
    options.batch_size = 8;
    options.monitor = false;
    options.graph_mode = GraphMode::Capture;
    options.drop_last = true;

    CHECK_THROWS(model.train(inputs, targets, options));
}
