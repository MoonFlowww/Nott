// Large CNN (U-Net, conv-only) on the Dubai satellite segmentation set. Reconstructs
// each tile's 3x3 original from its 9 cuts, re-cuts into fixed patches, trains per-pixel
// 6-class segmentation. Also measures the precision levers (fp32 vs AMP vs channels-last)
// on a genuinely heavy conv workload, since that is the real speed knob for CNNs.
#include "../../include/Nott.h"

#include <torch/torch.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <array>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace Nott;
namespace fs = std::filesystem;

namespace {

constexpr int kClasses = 6;
constexpr int kPatch = 128;

// class-index order, RGB (matches the mask PNGs, verified against actual pixels)
const std::array<std::array<uint8_t, 3>, kClasses> kPalette{{
    {60, 16, 152}, {110, 193, 228}, {132, 41, 246}, {155, 155, 155}, {226, 169, 41}, {254, 221, 58},
}};

cv::Mat load_rgb(const fs::path &p) {
    cv::Mat bgr = cv::imread(p.string(), cv::IMREAD_COLOR);
    if (bgr.empty()) throw std::runtime_error("failed to read " + p.string());
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    return rgb;
}

// parts 1..9 are a row-major 3x3 grid (rows share a height, all share a width)
cv::Mat reconstruct(const fs::path &dir, const std::string &ext) {
    std::array<cv::Mat, 9> part;
    for (int i = 0; i < 9; ++i) {
        char name[64];
        std::snprintf(name, sizeof(name), "image_part_%03d.%s", i + 1, ext.c_str());
        part[i] = load_rgb(dir / name);
    }
    cv::Mat rows[3];
    for (int r = 0; r < 3; ++r) {
        std::vector<cv::Mat> row{part[r * 3], part[r * 3 + 1], part[r * 3 + 2]};
        cv::hconcat(row, rows[r]);
    }
    cv::Mat full;
    cv::vconcat(std::vector<cv::Mat>{rows[0], rows[1], rows[2]}, full);
    return full;
}

torch::Tensor mat_to_chw(const cv::Mat &m) {  // uint8 HWC -> float CHW [0,1]
    auto t = torch::from_blob(m.data, {m.rows, m.cols, 3}, torch::kUInt8).clone();
    return t.permute({2, 0, 1}).to(torch::kFloat32).div_(255.0);
}

torch::Tensor mat_to_hwc_u8(const cv::Mat &m) {  // uint8 HWC as-is
    return torch::from_blob(m.data, {m.rows, m.cols, 3}, torch::kUInt8).clone();
}

// nearest-palette RGB -> class index, vectorized, on a stack of (N,P,P,3) uint8 masks
torch::Tensor masks_to_indices(const torch::Tensor &masks_u8) {
    const auto N = masks_u8.size(0);
    auto flat = masks_u8.reshape({N * kPatch * kPatch, 3}).to(torch::kInt32);
    std::vector<int32_t> pal;
    for (const auto &c : kPalette) { pal.push_back(c[0]); pal.push_back(c[1]); pal.push_back(c[2]); }
    auto palette = torch::tensor(pal, torch::kInt32).reshape({kClasses, 3});
    auto diff = flat.unsqueeze(1) - palette.unsqueeze(0);   // (M, C, 3)
    auto dist = diff.mul(diff).sum(-1);                     // (M, C)
    return dist.argmin(1).reshape({N, kPatch, kPatch}).to(torch::kLong);
}

struct Dataset {
    torch::Tensor images;   // (N,3,P,P) float
    torch::Tensor targets;  // (N,P,P) long
};

Dataset build_dataset(const fs::path &root) {
    std::vector<torch::Tensor> imgs, msks_u8;
    for (int tile = 1; tile <= 8; ++tile) {
        const fs::path tdir = root / ("Tile " + std::to_string(tile));
        cv::Mat img = reconstruct(tdir / "images", "jpg");
        cv::Mat msk = reconstruct(tdir / "masks", "png");
        const int H = std::min(img.rows, msk.rows), W = std::min(img.cols, msk.cols);
        for (int y = 0; y + kPatch <= H; y += kPatch) {
            for (int x = 0; x + kPatch <= W; x += kPatch) {
                cv::Rect roi(x, y, kPatch, kPatch);
                imgs.push_back(mat_to_chw(img(roi)));
                msks_u8.push_back(mat_to_hwc_u8(msk(roi)));
            }
        }
    }
    auto images = torch::stack(imgs);
    auto targets = masks_to_indices(torch::stack(msks_u8));
    return {images, targets};
}

// conv block: two 3x3 same-padding convs, ReLU
void add_block(Model &m, int64_t in, int64_t out, const std::string &a, const std::string &b) {
    m.add(Layer::Conv2d({in, out, {3, 3}, {1, 1}, {1, 1}}, Activation::ReLU, Initialization::HeUniform), a);
    m.add(Layer::Conv2d({out, out, {3, 3}, {1, 1}, {1, 1}}, Activation::ReLU, Initialization::HeUniform), b);
}

// large U-Net, conv only, skip connections via Stack (concat) joins
Model build_unet() {
    Model m("large_unet");
    add_block(m, 3, 64, "e1a", "e1b");
    m.add(Layer::MaxPool2d({{2, 2}, {2, 2}}), "p1");
    add_block(m, 64, 128, "e2a", "e2b");
    m.add(Layer::MaxPool2d({{2, 2}, {2, 2}}), "p2");
    add_block(m, 128, 256, "e3a", "e3b");
    m.add(Layer::MaxPool2d({{2, 2}, {2, 2}}), "p3");
    add_block(m, 256, 512, "b1", "b2");
    m.add(Layer::Upsample({.scale = {2.0, 2.0}}), "u3");
    add_block(m, 512 + 256, 256, "d3a", "d3b");
    m.add(Layer::Upsample({.scale = {2.0, 2.0}}), "u2");
    add_block(m, 256 + 128, 128, "d2a", "d2b");
    m.add(Layer::Upsample({.scale = {2.0, 2.0}}), "u1");
    add_block(m, 128 + 64, 64, "d1a", "d1b");
    m.add(Layer::Conv2d({64, kClasses, {1, 1}, {1, 1}, {0, 0}}, Activation::Identity, Initialization::XavierUniform), "out");

    auto J = [](const char *n) { return Port::Join(n, MergePolicy::Stack); };
    m.links({
        LinkSpec{Port::Input("@input"), Port::Module("e1a")},
        LinkSpec{Port::Module("e1a"), Port::Module("e1b")},
        LinkSpec{Port::Module("e1b"), Port::Module("p1")},
        LinkSpec{Port::Module("p1"), Port::Module("e2a")},
        LinkSpec{Port::Module("e2a"), Port::Module("e2b")},
        LinkSpec{Port::Module("e2b"), Port::Module("p2")},
        LinkSpec{Port::Module("p2"), Port::Module("e3a")},
        LinkSpec{Port::Module("e3a"), Port::Module("e3b")},
        LinkSpec{Port::Module("e3b"), Port::Module("p3")},
        LinkSpec{Port::Module("p3"), Port::Module("b1")},
        LinkSpec{Port::Module("b1"), Port::Module("b2")},
        LinkSpec{Port::Module("b2"), Port::Module("u3")},
        LinkSpec{Port::Module("u3"), J("j3")},
        LinkSpec{Port::Module("e3b"), J("j3")},
        LinkSpec{J("j3"), Port::Module("d3a")},
        LinkSpec{Port::Module("d3a"), Port::Module("d3b")},
        LinkSpec{Port::Module("d3b"), Port::Module("u2")},
        LinkSpec{Port::Module("u2"), J("j2")},
        LinkSpec{Port::Module("e2b"), J("j2")},
        LinkSpec{J("j2"), Port::Module("d2a")},
        LinkSpec{Port::Module("d2a"), Port::Module("d2b")},
        LinkSpec{Port::Module("d2b"), Port::Module("u1")},
        LinkSpec{Port::Module("u1"), J("j1")},
        LinkSpec{Port::Module("e1b"), J("j1")},
        LinkSpec{J("j1"), Port::Module("d1a")},
        LinkSpec{Port::Module("d1a"), Port::Module("d1b")},
        LinkSpec{Port::Module("d1b"), Port::Module("out")},
        LinkSpec{Port::Module("out"), Port::Output("@output")},
    }, false);
    return m;
}

void train_cfg(const std::string &label, const Dataset &d, bool use_cuda,
               bool amp, torch::MemoryFormat fmt, bool tf32_benchmark) {
    torch::manual_seed(7);  // identical init across configs so losses are comparable
    auto m = build_unet();
    m.use_cuda(use_cuda);
    if (tf32_benchmark) m.set_precision(/*allow_tf32=*/true, /*benchmark_cudnn=*/true);
    m.set_loss(Loss::CrossEntropy({}));
    m.set_optimizer(Optimizer::Adam({.learning_rate = 1e-3}));

    TrainOptions o{};
    o.epoch = 6;
    o.batch_size = 8;
    o.shuffle = false;
    o.monitor = false;
    o.enable_amp = amp;
    o.memory_format = fmt;

    m.train(d.images, d.targets, o);

    const auto &ep = m.training_telemetry().epochs();
    double sum = 0.0; std::size_t n = 0;
    for (std::size_t i = 0; i < ep.size(); ++i) if (i >= 2) { sum += ep[i].duration_seconds * 1000.0; ++n; }
    std::cout << std::left << std::setw(26) << label
              << " steady ms/epoch " << std::fixed << std::setprecision(1) << (n ? sum / n : 0.0)
              << " | first_loss " << std::setprecision(4) << ep.front().train_loss_value()
              << " last_loss " << ep.back().train_loss_value() << "\n";
}

}

int main() {
    const bool use_cuda = torch::cuda::is_available();
    const fs::path root = "/home/moonfloww/Projects/DATASETS/Image/Satellite/DubaiSegmentationImages";

    std::cout << "loading + reconstructing + patching...\n";
    auto full = build_dataset(root);
    // subsample for fast iteration on the precision-lever comparison
    const int64_t N = std::min<int64_t>(640, full.images.size(0));
    Dataset data{full.images.narrow(0, 0, N).contiguous(), full.targets.narrow(0, 0, N).contiguous()};
    std::cout << "device: " << (use_cuda ? "cuda" : "cpu")
              << " | samples (subset): " << data.images.size(0) << " of " << full.images.size(0)
              << " | image " << data.images.size(1) << "x" << data.images.size(2) << "x" << data.images.size(3)
              << " | classes present: " << std::get<0>(torch::_unique(data.targets)).numel() << "\n\n";

    train_cfg("fp32 default",         data, use_cuda, false, torch::MemoryFormat::Contiguous,   false);
    train_cfg("fp32 tf32+benchmark",  data, use_cuda, false, torch::MemoryFormat::Contiguous,   true);
    train_cfg("amp",                  data, use_cuda, true,  torch::MemoryFormat::Contiguous,   true);
    train_cfg("amp + channels-last",  data, use_cuda, true,  torch::MemoryFormat::ChannelsLast, true);

    return 0;
}
