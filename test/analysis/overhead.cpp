#include "Nott.h"

int main() {
    Nott::Model model("OverheadDetect");
    model.use_cuda(torch::cuda::is_available());


    model.add(Nott::Block::Sequential({
        Nott::Layer::Conv2d({1, 16, {3, 3}, {1, 1}, {1, 1}, {1, 1}, 1, false}, Nott::Activation::SiLU, Nott::Initialization::HeNormal),
        Nott::Layer::MaxPool2d({{2, 2}, {2, 2}})
    }));

    model.add(Nott::Block::Sequential({
        Nott::Layer::Conv2d({16, 32, {3, 3}, {1, 1}, {1, 1}, {1, 1}, 1, false}, Nott::Activation::SiLU, Nott::Initialization::HeNormal),
        Nott::Layer::MaxPool2d({{2, 2}, {2, 2}})
    }));

    model.add(Nott::Layer::AdaptiveAvgPool2d({{1, 1}})); // Result: 32x1x1
    model.add(Nott::Layer::Flatten());

    model.add(Nott::Layer::HardDropout({.probability = 0.2}));
    model.add(Nott::Layer::FC({32, 10, true}, Nott::Activation::Identity, Nott::Initialization::HeNormal));

    model.set_loss(Nott::Loss::CrossEntropy({}));
    model.set_optimizer(Nott::Optimizer::Adam({.learning_rate = 1e-3}));


    auto [x1, y1, x2, y2] = Nott::Data::Load::MNIST("/home/moonfloww/Projects/DATASETS/Image/MNIST/", 0.1f, 0.f, true);
    model.train(x1, y1, {.epoch=5, .batch_size=16, .buffer_vram = 2});

    return 0;
}