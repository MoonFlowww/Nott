/**
 * @brief CIFAR-10 image classification with Vision Transformer (ViT) backbone.
 *
 * Architecture:
 *   Vision Transformer encoder (patch_size=4, 8 layers, 8 heads, embed_dim=256)
 *   -> Global mean pooling over token dimension
 *   -> FC(256 -> 128) + SiLU + Dropout(0.15)
 *   -> FC(128 -> 10) classification head
 *
 * Training regime:
 *   - Optimizer: AdamW(lr=5e-4, weight_decay=5e-2) + CosineAnnealing LR with 5-epoch warmup
 *   - Loss: CrossEntropy with label_smoothing=0.1
 *   - Regularization: L2(1e-4) + SWA ensemble averaging
 *   - 30 epochs, batch_size=128, AMP enabled, ChannelsLast memory format
 */

#include <iostream>
#include <cstddef>
#include <vector>
#include <torch/torch.h>
#include <utility>
#include <cmath>
#include "../../../include/Nott.h"

int main() {
    const bool use_cuda = torch::cuda::is_available();

    /* ---- Data loading ---- */
    auto [x1, y1, x2, y2] = Nott::Data::Load::CIFAR10(
        "/home/moonfloww/Projects/DATASETS/Image/CIFAR10/",
        1.0f, 1.0f, true
    );

    const int64_t N = x1.size(0);
    const int64_t B = 128;
    const int64_t epochs = 30;
    const int64_t steps_per_epoch = (N + B - 1) / B;

    /* ---- Data augmentation ---- */
    std::tie(x1, y1) = Nott::Data::Manipulation::Flip(x1, y1, {{"x"}, 0.5f, true, false});
    std::tie(x1, y1) = Nott::Data::Manipulation::Cutout(x1, y1, {{-1, -1}, {8, 8}, {-1, -1, -1}, 0.5f, true, false});
    std::tie(x1, y1) = Nott::Data::Manipulation::Shuffle(x1, y1);

    /* ---- Build model ---- */
    Nott::Model model("CIFAR10_ViT");
    model.use_cuda(use_cuda);

    /** ViT encoder: processes (B, 3, 32, 32) into (B, 65, 256) token sequence */
    Nott::Block::Transformer::Vision::EncoderOptions vit_opts{};
    vit_opts.layers = 8;
    vit_opts.embed_dim = 256;
    vit_opts.variant = Nott::Block::Transformer::Vision::Variant::ViT;
    vit_opts.attention.embed_dim = 256;
    vit_opts.attention.num_heads = 8;
    vit_opts.attention.dropout = 0.0;
    vit_opts.attention.bias = true;
    vit_opts.attention.batch_first = true;
    vit_opts.feed_forward.embed_dim = 256;
    vit_opts.feed_forward.mlp_ratio = 4.0;
    vit_opts.feed_forward.activation = Nott::Activation::GeLU;
    vit_opts.feed_forward.bias = true;
    vit_opts.layer_norm.eps = 1e-6;
    vit_opts.patch_embedding.in_channels = 3;
    vit_opts.patch_embedding.embed_dim = 256;
    vit_opts.patch_embedding.patch_size = 4;
    vit_opts.patch_embedding.add_class_token = true;
    vit_opts.patch_embedding.normalize = true;
    vit_opts.patch_embedding.dropout = 0.0;
    vit_opts.residual_dropout = 0.0;
    vit_opts.attention_dropout = 0.1;
    vit_opts.feed_forward_dropout = 0.1;
    vit_opts.pre_norm = true;
    vit_opts.final_layer_norm = true;
    model.add(Nott::Block::Transformer::Vision::Encoder(vit_opts), "vit");

    /** Global pooling over token dimension (dim=1) -> (B, 256) */
    model.add(Nott::Layer::Reduce({
        .op = Nott::Layer::ReduceOp::Mean,
        .dims = {1},
        .keep_dim = false
    }), "pool");

    /** Classification head */
    model.add(Nott::Layer::FC(
        {256, 128, true},
        Nott::Activation::SiLU,
        Nott::Initialization::HeNormal
    ), "fc1");

    model.add(Nott::Layer::HardDropout({.probability = 0.15}), "drop1");

    model.add(Nott::Layer::FC(
        {128, 10, true},
        Nott::Activation::Identity,
        Nott::Initialization::XavierUniform
    ), "logits");

    /* ---- Training configuration ---- */
    model.set_optimizer(
        Nott::Optimizer::AdamW({
            .learning_rate = 5e-4,
            .beta1 = 0.9,
            .beta2 = 0.999,
            .eps = 1e-8,
            .weight_decay = 5e-2,
            .amsgrad = false
        }),
        Nott::LrScheduler::CosineAnnealing({
            .T_max = static_cast<std::size_t>(epochs) * static_cast<std::size_t>(steps_per_epoch),
            .eta_min = 1e-6,
            .warmup_steps = 5 * static_cast<std::size_t>(steps_per_epoch),
            .warmup_start_factor = 0.1
        })
    );

    model.set_loss(Nott::Loss::CrossEntropy({.label_smoothing = 0.1f}));

    model.set_regularization({
        Nott::Regularization::L2({.coefficient = 1e-4}),
        Nott::Regularization::SWA({.coefficient = 1e-4}),
    });

    /* ---- Training ---- */
    Nott::Data::Check::Size(x1, "Train images");
    Nott::Data::Check::Size(y1, "Train labels");

    std::cout << "\nTraining CIFAR10_ViT for " << epochs << " epochs...\n";
    model.train(x1, y1, {
        .epoch = static_cast<std::size_t>(epochs),
        .batch_size = static_cast<std::size_t>(B),
        .shuffle = true,
        .restore_best_state = true,
        .test = std::vector<at::Tensor>{x2, y2},
        .graph_mode = Nott::GraphMode::Capture,
        .enable_amp = true,
        .memory_format = torch::MemoryFormat::ChannelsLast
    });

    /* ---- Evaluation ---- */
    std::cout << "\nEvaluating on test set...\n";
    model.evaluate(x2, y2, Nott::Evaluation::Classification, {
        Nott::Metric::Classification::Accuracy,
        Nott::Metric::Classification::Precision,
        Nott::Metric::Classification::Recall,
        Nott::Metric::Classification::F1,
        Nott::Metric::Classification::Top1Error,
        Nott::Metric::Classification::LogLoss,
        Nott::Metric::Classification::BrierScore,
        Nott::Metric::Classification::ExpectedCalibrationError,
        Nott::Metric::Classification::CohensKappa,
        Nott::Metric::Classification::Informedness,
    }, {.batch_size = 256});

    /* ---- Interpretability ---- */
    Nott::Plot::Render(model, Nott::Plot::Reliability::GradCAM({
        .samples = 8,
        .random = false,
        .normalize = true,
        .overlay = true
    }), x2, y2);

    return 0;
}
