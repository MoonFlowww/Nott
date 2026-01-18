#include <iostream>
#include <cstddef>
#include <vector>
#include <torch/torch.h>
#include <utility>
#include "../../../include/Nott.h"

int main() {
    Nott::Model model("Debug_CIFAR");
    model.use_cuda(torch::cuda::is_available());

    const int64_t N = 200000;
    const int64_t B = std::pow(2,7);
    const int64_t epochs = 50;
    const int64_t steps_per_epoch = (N + B - 1) / B;

    
    model.add(Nott::Block::Sequential({
        Nott::Layer::Conv2d({3, 64, {3, 3}, {1, 1}, {1, 1}, {1, 1}, 1, false}, Nott::Activation::Identity, Nott::Initialization::HeNormal),
        Nott::Layer::BatchNorm2d({64, 1e-5, 0.1, true, true}, Nott::Activation::SiLU),
        Nott::Layer::Conv2d({64, 64, {3, 3}, {1, 1}, {1, 1}, {1, 1}, 1, true}, Nott::Activation::Identity, Nott::Initialization::HeNormal),
        Nott::Layer::BatchNorm2d({64, 1e-5, 0.1, true, true}, Nott::Activation::SiLU),
        Nott::Layer::MaxPool2d({{2, 2}, {2, 2}})
    }));

    model.add(Nott::Layer::HardDropout({ .probability = 0.3 }));

    model.add(Nott::Block::Residual({
        Nott::Layer::Conv2d({64, 64, {3, 3}, {1, 1}, {1, 1}, {1, 1}, 1, false}, Nott::Activation::Identity, Nott::Initialization::HeNormal),
        Nott::Layer::BatchNorm2d({64, 1e-5, 0.1, true, true}, Nott::Activation::SiLU),
        Nott::Layer::Conv2d({64, 64, {3, 3}, {1, 1}, {1, 1}, {1, 1}, 1, true}, Nott::Activation::Identity, Nott::Initialization::HeNormal)
    }, 3, {}, { .final_activation = Nott::Activation::SiLU }));

    model.add(Nott::Block::Residual({
        Nott::Layer::Conv2d({64, 128, {3, 3}, {2, 2}, {1, 1}, {1, 1}, 1, false}, Nott::Activation::Identity, Nott::Initialization::HeNormal),
        Nott::Layer::BatchNorm2d({128, 1e-5, 0.1, true, true}, Nott::Activation::SiLU),
        Nott::Layer::Conv2d({128, 128, {3, 3}, {1, 1}, {1, 1}, {1, 1}, 1, true}, Nott::Activation::Identity, Nott::Initialization::HeNormal)
    }, 1, {.projection = Nott::Layer::Conv2d({64, 128, {1, 1}, {2, 2}, {0, 0}, {1, 1}, 1, false}, Nott::Activation::Identity, Nott::Initialization::HeNormal)}, { .final_activation = Nott::Activation::SiLU }));

    model.add(Nott::Layer::HardDropout({ .probability = 0.3 }));

    model.add(Nott::Block::Sequential({
        Nott::Layer::Conv2d({128, 256, {3, 3}, {1, 1}, {1, 1}, {1, 1}, 1, false}, Nott::Activation::Identity, Nott::Initialization::HeNormal),
        Nott::Layer::BatchNorm2d({256, 1e-5, 0.1, true, true}, Nott::Activation::SiLU),
        Nott::Layer::AdaptiveAvgPool2d({{1, 1}})
    }));

    model.add(Nott::Block::Sequential({
        Nott::Layer::Conv2d({256, 128, {3, 3}, {1, 1}, {1, 1}, {1, 1}, 1, false}, Nott::Activation::Identity, Nott::Initialization::HeNormal),
        Nott::Layer::BatchNorm2d({128, 1e-5, 0.1, true, true}, Nott::Activation::SiLU),
        Nott::Layer::AdaptiveAvgPool2d({{1, 1}})
    }));

    model.add(Nott::Layer::Flatten());

    model.add(Nott::Layer::FC({128, 512, true}, Nott::Activation::SiLU, Nott::Initialization::HeNormal));
    model.add(Nott::Layer::HardDropout({.probability = 0.5}));
    model.add(Nott::Layer::FC({512, 10, true}, Nott::Activation::Identity, Nott::Initialization::HeNormal));


    model.set_optimizer(
        Nott::Optimizer::AdamW({.learning_rate=1e-4, .weight_decay=5e-4}),
            Nott::LrScheduler::CosineAnnealing({
            .T_max = static_cast<size_t>(epochs*0.85) * steps_per_epoch,
            .eta_min = 1e-5,
            .warmup_steps = 5*static_cast<size_t>(steps_per_epoch),
            .warmup_start_factor = 0.1
        })
    );

    model.set_loss(Nott::Loss::CrossEntropy({.label_smoothing=0.02f}));

    model.set_regularization({ Nott::Regularization::SWAG({
        .coefficient = 1e-3,
        .variance_epsilon = 1e-6,
        .start_step = static_cast<size_t>(0.85 * (steps_per_epoch*epochs)),
        .accumulation_stride = static_cast<size_t>(steps_per_epoch),
        .max_snapshots = 20,
    }), Nott::Regularization::VAT({5})});


    

    auto [x1, y1, x2, y2] = Nott::Data::Load::CIFAR10("/home/moonfloww/Projects/DATASETS/Image/CIFAR10", 1.f, 1.f, true);
    auto [validation_images, validation_labels] = Nott::Data::Manipulation::Fraction(x2, y2, 0.1f);
    Nott::Data::Check::Size(x1, "Raw");
    std::tie(x1, y1) = Nott::Data::Transform::Augmentation::CLAHE(x1, y1, {256, 2.f, {4,4}, 1.f, true});
    std::tie(x1, y1) = Nott::Data::Manipulation::Cutout(x1, y1, {{-1, -1}, {12, 12}, {-1,-1,-1}, .5f, false, false});
    std::tie(x1, y1) = Nott::Data::Manipulation::Cutout(x1, y1, {{-1, -1}, {12, 12}, {-1,-1,-1}, 1.f, false, false});
    std::tie(x1, y1) = Nott::Data::Manipulation::Cutout(x1, y1, {{-1, -1}, {6, 6}, {-1,-1,-1}, 1.f, false, false});
    std::tie(x1, y1) = Nott::Data::Manipulation::Cutout(x1, y1, {{-1, -1}, {6, 6}, {-1,-1,-1}, 1.f, false, false});
    std::tie(x1, y1) = Nott::Data::Manipulation::Shuffle(x1, y1);

    std::tie(x1, y1) = Nott::Data::Manipulation::Flip(x1, y1, {{"x"}, 1.f, true, false});
    std::tie(x1, y1) = Nott::Data::Manipulation::Shuffle(x1, y1);

    Nott::Data::Check::Size(x1, "Augmented");
    Nott::Plot::Data::Image(x1, {1,2,3,4,5,6,7,8,9}); //idx

    model.train(x1, y1, {
        .epoch = static_cast<std::size_t>(epochs),
        .batch_size = static_cast<std::size_t>(B),
        .shuffle = false,
        .restore_best_state = true,
        .test = std::vector<at::Tensor>{validation_images, validation_labels},
        .graph_mode = Nott::GraphMode::Capture,
        .enable_amp=true,
        .memory_format = torch::MemoryFormat::ChannelsLast}
    );


    model.evaluate(x2, y2, Nott::Evaluation::Classification,{
        Nott::Metric::Classification::Accuracy,
        Nott::Metric::Classification::Precision,
        Nott::Metric::Classification::Recall,
        Nott::Metric::Classification::F1,
        Nott::Metric::Classification::TruePositiveRate,
        Nott::Metric::Classification::TrueNegativeRate,
        Nott::Metric::Classification::Top1Error,
        Nott::Metric::Classification::ExpectedCalibrationError,
        Nott::Metric::Classification::MaximumCalibrationError,
        Nott::Metric::Classification::CohensKappa,
        Nott::Metric::Classification::LogLoss,
        Nott::Metric::Classification::BrierScore,
        Nott::Metric::Classification::Informedness,
    }, {.batch_size = 64});

    Nott::Plot::Render(model, Nott::Plot::Reliability::GradCAM({.samples = 4, .random = false, .normalize = true, .overlay = true}),
        validation_images,validation_labels);

    //Nott::Plot::Render(model, Nott::Plot::Reliability::LIME({.random = true, .normalize = true, .showWeights = true}),
    //    validation_images, validation_labels);

    return 0;
}


/*
 *Epoch [19/20] | Train loss: 0.445165 | Test loss: 0.542593 | ΔLoss: -0.001442 (∇) | duration: 23.38sec
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┓
┃ Evaluation: Classification ┃          ┃                    ┃
┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━━━┫
┃ Metric                     ┃  Macro   ┃ Weighted (support) ┃
┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━━━┫
┃ Accuracy                   ┃ 0.974560 ┃           0.974560 ┃
┃ Precision                  ┃ 0.872237 ┃           0.872237 ┃
┃ Recall                     ┃ 0.872800 ┃           0.872800 ┃
┃ F1 score                   ┃ 0.872213 ┃           0.872213 ┃
┃ True positive rate         ┃ 0.872800 ┃           0.872800 ┃
┃ True negative rate         ┃ 0.985867 ┃           0.985867 ┃
┃ Top-1 error                ┃ 0.127200 ┃           0.127200 ┃
┃ Expected calibration error ┃ 0.037887 ┃           0.037887 ┃
┃ Maximum calibration error  ┃ 0.148656 ┃           0.148656 ┃
┃ Cohen's kappa              ┃ 0.858667 ┃           0.858667 ┃
┃ Log loss                   ┃ 0.415918 ┃           0.415918 ┃
┃ Brier score                ┃ 0.190100 ┃           0.190100 ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━┻━━━━━━━━━━┻━━━━━━━━━━━━━━━━━━━━┛

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┓
┃ Per-class metrics          ┃          ┃          ┃          ┃          ┃          ┃          ┃          ┃          ┃          ┃          ┃
┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━┫
┃ Metric                     ┃  Label 0 ┃  Label 1 ┃  Label 2 ┃  Label 3 ┃  Label 4 ┃  Label 5 ┃  Label 6 ┃  Label 7 ┃  Label 8 ┃  Label 9 ┃
┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━┫
┃ Support                    ┃     1000 ┃     1000 ┃     1000 ┃     1000 ┃     1000 ┃     1000 ┃     1000 ┃     1000 ┃     1000 ┃     1000 ┃
┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━┫
┃ Accuracy                   ┃ 0.974100 ┃ 0.986300 ┃ 0.964800 ┃ 0.947300 ┃ 0.968900 ┃ 0.958100 ┃ 0.977600 ┃ 0.980000 ┃ 0.984100 ┃ 0.982600 ┃
┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━┫
┃ Precision                  ┃ 0.857971 ┃ 0.949948 ┃ 0.862416 ┃ 0.774044 ┃ 0.813467 ┃ 0.798561 ┃ 0.854662 ┃ 0.899202 ┃ 0.905497 ┃ 0.895594 ┃
┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━┫
┃ Recall                     ┃ 0.888000 ┃ 0.911000 ┃ 0.771000 ┃ 0.668000 ┃ 0.894000 ┃ 0.777000 ┃ 0.935000 ┃ 0.901000 ┃ 0.939000 ┃ 0.935000 ┃
┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━┫
┃ F1 score                   ┃ 0.872727 ┃ 0.930066 ┃ 0.814150 ┃ 0.717123 ┃ 0.851834 ┃ 0.787633 ┃ 0.893028 ┃ 0.900100 ┃ 0.921944 ┃ 0.914873 ┃
┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━┫
┃ True positive rate         ┃ 0.888000 ┃ 0.911000 ┃ 0.771000 ┃ 0.668000 ┃ 0.894000 ┃ 0.777000 ┃ 0.935000 ┃ 0.901000 ┃ 0.939000 ┃ 0.935000 ┃
┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━┫
┃ True negative rate         ┃ 0.983667 ┃ 0.994667 ┃ 0.986333 ┃ 0.978333 ┃ 0.977222 ┃ 0.978222 ┃ 0.982333 ┃ 0.988778 ┃ 0.989111 ┃ 0.987889 ┃
┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━┫
┃ Top-1 error                ┃ 0.025900 ┃ 0.013700 ┃ 0.035200 ┃ 0.052700 ┃ 0.031100 ┃ 0.041900 ┃ 0.022400 ┃ 0.020000 ┃ 0.015900 ┃ 0.017400 ┃
┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━┫
┃ Expected calibration error ┃ 0.005989 ┃ 0.005526 ┃ 0.013781 ┃ 0.016694 ┃ 0.008951 ┃ 0.010240 ┃ 0.008629 ┃ 0.003474 ┃ 0.005556 ┃ 0.006448 ┃
┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━┫
┃ Maximum calibration error  ┃ 0.147397 ┃ 0.320176 ┃ 0.127196 ┃ 0.141762 ┃ 0.144576 ┃ 0.177146 ┃ 0.325090 ┃ 0.217588 ┃ 0.454391 ┃ 0.250556 ┃
┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━┫
┃ Cohen's kappa              ┃      nan ┃      nan ┃      nan ┃      nan ┃      nan ┃      nan ┃      nan ┃      nan ┃      nan ┃      nan ┃
┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━┫
┃ Log loss                   ┃ 0.337757 ┃ 0.299296 ┃ 0.745857 ┃ 0.978922 ┃ 0.336989 ┃ 0.622666 ┃ 0.244309 ┃ 0.312370 ┃ 0.187604 ┃ 0.213017 ┃
┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━┫
┃ Brier score                ┃ 0.159042 ┃ 0.136714 ┃ 0.349428 ┃ 0.480833 ┃ 0.158239 ┃ 0.314499 ┃ 0.104114 ┃ 0.143320 ┃ 0.087378 ┃ 0.099287 ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━┻━━━━━━━━━━┻━━━━━━━━━━┻━━━━━━━━━━┻━━━━━━━━━━┻━━━━━━━━━━┻━━━━━━━━━━┻━━━━━━━━━━┻━━━━━━━━━━┻━━━━━━━━━━┻━━━━━━━━━━┛
*/
