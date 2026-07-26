// End-to-end parity harness: trains CNN / UNet-with-skip / Transformer / LSTM on
// deterministic dummy data (and MNIST if present), prints per-epoch loss and wall
// time so the same binary built on two branches can be diffed. Uses only common
// public API so it compiles on master and on the feature branch unchanged.
#include "../../include/Nott.h"

#include <torch/torch.h>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace Nott;

namespace {

constexpr int64_t kSamples = 512;
constexpr int64_t kClasses = 10;
constexpr std::size_t kEpochs = 20;
constexpr std::size_t kWarmupEpochs = 8;  // drop these (cuDNN autotune, cache priming)
constexpr std::size_t kBatch = 64;
constexpr uint64_t kSeed = 1234;

struct DummyData {
    torch::Tensor inputs;
    torch::Tensor targets;
};

DummyData make_dummy(std::vector<int64_t> input_shape) {
    torch::manual_seed(kSeed);
    input_shape.insert(input_shape.begin(), kSamples);
    auto inputs = torch::randn(input_shape);
    auto targets = torch::randint(0, kClasses, {kSamples}, torch::TensorOptions().dtype(torch::kLong));
    return {inputs, targets};
}

void run(const std::string &label, Model &model, const DummyData &data, bool use_cuda,
         GraphMode graph_mode) {
    model.use_cuda(use_cuda);
    model.set_loss(Loss::CrossEntropy({}));
    model.set_optimizer(Optimizer::Adam({.learning_rate = 1e-3}));

    TrainOptions options{};
    options.epoch = kEpochs;
    options.batch_size = kBatch;
    options.shuffle = false;   // determinism: same data order every run
    options.monitor = false;
    options.graph_mode = graph_mode;
    options.drop_last = (graph_mode != GraphMode::Disabled);

    model.train(data.inputs, data.targets, options);

    // steady-state per-epoch: mean of epochs after warmup
    const auto &epochs = model.training_telemetry().epochs();
    double sum = 0.0;
    std::size_t count = 0;
    double final_loss = 0.0;
    for (std::size_t i = 0; i < epochs.size(); ++i) {
        if (i >= kWarmupEpochs) { sum += epochs[i].duration_seconds * 1000.0; ++count; }
        final_loss = epochs[i].train_loss_value();
    }
    const double steady_ms = count ? sum / static_cast<double>(count) : 0.0;

    // machine-parseable: KEY <tab> steady_ms <tab> final_loss
    std::cout << label << "\t" << std::fixed << std::setprecision(3) << steady_ms
              << "\t" << std::setprecision(6) << final_loss << "\n";
}

Model build_cnn() {
    Model model("cnn");
    model.add(Layer::Conv2d({3, 16, {3, 3}, {1, 1}, {1, 1}}, Activation::ReLU, Initialization::HeUniform));
    model.add(Layer::MaxPool2d({{2, 2}, {2, 2}}));
    model.add(Layer::Conv2d({16, 32, {3, 3}, {1, 1}, {1, 1}}, Activation::ReLU, Initialization::HeUniform));
    model.add(Layer::AdaptiveAvgPool2d({.output_size = {1, 1}}));
    model.add(Layer::Flatten());
    model.add(Layer::FC({32, kClasses}, Activation::Identity, Initialization::XavierUniform));
    return model;
}

// encoder-decoder with a channel-concat skip join, exercises links() + Stack join
Model build_unet() {
    Model model("unet");
    model.add(Layer::Conv2d({3, 16, {3, 3}, {1, 1}, {1, 1}}, Activation::ReLU, Initialization::HeUniform), "enc");
    model.add(Layer::Downsample({.scale = {2.0, 2.0}}), "down");
    model.add(Layer::Conv2d({16, 16, {3, 3}, {1, 1}, {1, 1}}, Activation::ReLU, Initialization::HeUniform), "bott");
    model.add(Layer::Upsample({.scale = {2.0, 2.0}}), "up");
    model.add(Layer::Conv2d({32, 16, {3, 3}, {1, 1}, {1, 1}}, Activation::ReLU, Initialization::HeUniform), "dec");
    model.add(Layer::AdaptiveAvgPool2d({.output_size = {1, 1}}), "gap");
    model.add(Layer::Flatten(), "flatten");
    model.add(Layer::FC({16, kClasses}, Activation::Identity, Initialization::XavierUniform), "head");

    model.links({
        LinkSpec{Port::Input("@input"), Port::Module("enc")},
        LinkSpec{Port::Module("enc"), Port::Module("down")},
        LinkSpec{Port::Module("down"), Port::Module("bott")},
        LinkSpec{Port::Module("bott"), Port::Module("up")},
        LinkSpec{Port::Module("enc"), Port::Join("skip", MergePolicy::Stack)},
        LinkSpec{Port::Module("up"), Port::Join("skip", MergePolicy::Stack)},
        LinkSpec{Port::Join("skip", MergePolicy::Stack), Port::Module("dec")},
        LinkSpec{Port::Module("dec"), Port::Module("gap")},
        LinkSpec{Port::Module("gap"), Port::Module("flatten")},
        LinkSpec{Port::Module("flatten"), Port::Module("head")},
        LinkSpec{Port::Module("head"), Port::Output("@output")},
    }, false);
    return model;
}

Model build_transformer() {
    Model model("transformer");
    Block::Transformer::Classic::EncoderOptions opts{};
    opts.layers = 2;
    opts.embed_dim = 32;
    model.add(Block::Transformer::Classic::Encoder(opts));
    model.add(Layer::Flatten());
    model.add(Layer::FC({16 * 32, kClasses}, Activation::Identity, Initialization::XavierUniform));
    return model;
}

// long-sequence transformer: attention is O(seq^2), so at seq=256 it is the
// dominant cost and the attention fusion should approach its microbench speedup
Model build_transformer_big(int64_t seq, int64_t embed) {
    Model model("transformer_big");
    Block::Transformer::Classic::EncoderOptions opts{};
    opts.layers = 4;
    opts.embed_dim = embed;
    model.add(Block::Transformer::Classic::Encoder(opts));
    model.add(Layer::Flatten());
    model.add(Layer::FC({seq * embed, kClasses}, Activation::Identity, Initialization::XavierUniform));
    return model;
}

Model build_lstm() {
    Model model("lstm");
    model.add(Layer::LSTM({.input_size = 8, .hidden_size = 16, .batch_first = true}));
    model.add(Layer::Flatten());
    model.add(Layer::FC({16 * 16, kClasses}, Activation::Identity, Initialization::XavierUniform));
    return model;
}

}

int main() {
    const bool use_cuda = torch::cuda::is_available();
    std::cout << "device: " << (use_cuda ? "cuda" : "cpu") << "\n";

    // absorb one-time CUDA context + cuDNN init so no timed run pays for it
    if (use_cuda) {
        auto w = torch::randn({8, 3, 32, 32}, torch::TensorOptions().device(torch::kCUDA));
        auto k = torch::randn({16, 3, 3, 3}, torch::TensorOptions().device(torch::kCUDA));
        (void) torch::conv2d(w, k).sum().item();
        torch::cuda::synchronize();
    }

    {
        auto data = make_dummy({3, 32, 32});
        auto m = build_cnn();
        run("cnn_disabled", m, data, use_cuda, GraphMode::Disabled);
    }
    {
        auto data = make_dummy({3, 32, 32});
        auto m = build_unet();
        run("unet", m, data, use_cuda, GraphMode::Disabled);
    }
    {
        auto data = make_dummy({16, 32});
        auto m = build_transformer();
        run("transformer_seq16", m, data, use_cuda, GraphMode::Disabled);
    }
    {
        auto data = make_dummy({64, 64});
        auto m = build_transformer_big(64, 64);
        run("transformer_seq64", m, data, use_cuda, GraphMode::Disabled);
    }
    {
        auto data = make_dummy({256, 64});
        auto m = build_transformer_big(256, 64);
        run("transformer_seq256", m, data, use_cuda, GraphMode::Disabled);
    }
    {
        auto data = make_dummy({16, 8});
        auto m = build_lstm();
        run("lstm", m, data, use_cuda, GraphMode::Disabled);
    }

    const std::string mnist_root = "/home/moonfloww/Projects/DATASETS/Image/MNIST/";
    try {
        auto [x1, y1, x2, y2] = Data::Load::MNIST(mnist_root, 0.1f, 0.0f, true);
        (void) x2; (void) y2;
        Model cnn("cnn_mnist"); // MNIST is 1-channel
        cnn.add(Layer::Conv2d({1, 16, {3, 3}, {1, 1}, {1, 1}}, Activation::ReLU, Initialization::HeUniform));
        cnn.add(Layer::MaxPool2d({{2, 2}, {2, 2}}));
        cnn.add(Layer::Conv2d({16, 32, {3, 3}, {1, 1}, {1, 1}}, Activation::ReLU, Initialization::HeUniform));
        cnn.add(Layer::AdaptiveAvgPool2d({.output_size = {1, 1}}));
        cnn.add(Layer::Flatten());
        cnn.add(Layer::FC({32, kClasses}, Activation::Identity, Initialization::XavierUniform));
        DummyData mnist{x1, y1};
        run("cnn_mnist", cnn, mnist, use_cuda, GraphMode::Disabled);
    } catch (const std::exception &e) {
        std::cout << "cnn        mnist SKIPPED (" << e.what() << ")\n";
    }

    return 0;
}
