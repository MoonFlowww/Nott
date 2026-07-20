#include "../../include/Nott.h"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <tuple>
#include <vector>
#include <string>

namespace {
    using namespace torch::indexing;

    struct ToyDataset {
        torch::Tensor x;  // [N, 1, 2]
        torch::Tensor y;  // [N]
    };

    ToyDataset make_spiral_dataset(int64_t samples_per_class, int64_t num_classes, double noise_std = 0.2) {
        const int64_t total_samples = samples_per_class * num_classes;
        auto input_opts = torch::TensorOptions().dtype(torch::kFloat32);
        auto label_opts = torch::TensorOptions().dtype(torch::kLong);

        auto x = torch::zeros({total_samples, 2}, input_opts);
        auto y = torch::zeros({total_samples}, label_opts);

        for (int64_t class_idx = 0; class_idx < num_classes; ++class_idx) {
            const auto start = class_idx * samples_per_class;
            const auto end = start + samples_per_class;
            auto slice = Slice(start, end);

            auto r = torch::linspace(0.0, 1.0, samples_per_class, input_opts);
            auto theta = torch::linspace(class_idx * 4.0, (class_idx + 1) * 4.0, samples_per_class, input_opts);
            theta += torch::randn({samples_per_class}, input_opts) * noise_std;

            x.index_put_({slice, 0}, r * torch::sin(theta));
            x.index_put_({slice, 1}, r * torch::cos(theta));
            y.index_put_({slice}, torch::full({samples_per_class}, class_idx, label_opts));
        }

        x = x.view({total_samples, 1, 2});
        return {x, y};
    }

    void print_vector(const std::vector<double>& values, const std::string& label, std::size_t max_items = 10) {
        std::cout << label;
        if (values.empty()) {
            std::cout << " <empty>\n";
            return;
        }

        const auto limit = std::min(values.size(), max_items);
        std::cout << " [";
        for (std::size_t i = 0; i < limit; ++i) {
            std::cout << std::fixed << std::setprecision(4) << values[i];
            if (i + 1 < limit) {
                std::cout << ", ";
            }
        }
        if (limit < values.size()) {
            std::cout << ", ...";
        }
        std::cout << "]\n";
    }
}

int main() {
    Nott::Model model("");
    const auto device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    model.use_cuda(torch::cuda::is_available());

    const int64_t hidden_dim1 = 64;
    const int64_t hidden_dim2 = 64;
    const int64_t num_classes = 3;
    model.add(Nott::Layer::Flatten());
    model.add(Nott::Layer::FC({.in_features = 2, .out_features = hidden_dim1}, Nott::Activation::GeLU,
                              Nott::Initialization::HeUniform));
    model.add(Nott::Layer::FC({.in_features = hidden_dim1, .out_features = hidden_dim2}, Nott::Activation::GeLU,
                              Nott::Initialization::HeUniform));
    model.add(Nott::Layer::FC({.in_features = hidden_dim2, .out_features = num_classes}, Nott::Activation::Identity,
                              Nott::Initialization::HeUniform));
    model.to(device);

    const int64_t samples_per_class = std::pow(2,12);
    auto [x, y] = make_spiral_dataset(samples_per_class, num_classes);
    auto [x2, y2] = make_spiral_dataset(16, num_classes);

    x = x.to(device);
    y = y.to(device);
    model.set_loss(Nott::Loss::CrossEntropy({.label_smoothing=0.02f}));
    model.set_optimizer(Nott::Optimizer::AdamW({.learning_rate=0.0004}));
    model.train(x, y, {.epoch = 20, .batch_size=64, .restore_best_state=true, .test=std::vector<at::Tensor>{x2, y2}});
    Nott::NTK::Options options;
    options.kernel_type = Nott::NTK::KernelType::NTK;
    options.output_mode = Nott::NTK::OutputMode::SumOutputs;
    options.approximation = Nott::NTK::Approximation::Exact;
    options.memory_mode = Nott::NTK::MemoryMode::FullGPU;
    options.max_samples = x.size(0);
    options.center_kernel = true;
    options.normalize_diag = false;
    options.compute_eigs = true;
    options.top_k_eigs = 12;
    options.ridge_lambda = 1e-3;
    options.estimate_lr_bound = true;
    options.run_kernel_regression = true;// solves KRR with the NTK as the kernel
    options.stream = &std::cout;
    options.print_summary = false;

    auto [xf1, yf1] = Nott::Data::Manipulation::Fraction(x, y, 0.1f);
    auto result = Nott::NTK::Compute(model, xf1, yf1, options);

    const auto& s = result.stats;
    std::cout << "\n[NTK] ==================================================\n";

    // ---------- A. Setup / configuration ----------
    std::cout << "[NTK] Setup / configuration\n";
    std::cout << "  Model name         : " << s.model_name << "\n";
    std::cout << "  Samples used       : " << s.n_samples << "\n";
    std::cout << "  Effective params   : " << s.n_params_effective << "\n";
    std::cout << "  Kernel type        : " << to_string(s.kernel_type)   << "\n";
    std::cout << "  Output mode        : " << to_string(s.output_mode)   << "\n";
    std::cout << "  Centered / normdiag: " << std::boolalpha << s.centered << " / " << s.normalized_diag << std::noboolalpha << "\n";
    std::cout << "  Device / dtype     : " << s.kernel_device << " / " << s.kernel_dtype << "\n";
    std::cout << "  Approximation mode : " << to_string(s.approximation_mode) << " (feature_dim_used=" << s.feature_dim_used << ")\n";
    std::cout << "  Random seed        : " << s.random_seed << "\n";

    if (!s.sample_indices.empty()) {
        const auto min_idx = *std::ranges::min_element(s.sample_indices.begin(), s.sample_indices.end());
        const auto max_idx = *std::ranges::max_element(s.sample_indices.begin(), s.sample_indices.end());
        std::cout << "  Sample indices     : count=" << s.sample_indices.size()
                  << ", range=[" << min_idx << ", " << max_idx << "]\n";
    }

    std::cout << "\n";

    // ---------- B. Spectrum / capacity / conditioning ----------
    std::cout << "[NTK] Spectrum / capacity / conditioning\n";
    std::cout << "  Trace              : " << s.trace << "\n";
    std::cout << "  Frobenius norm     : " << s.frobenius_norm << "\n";
    std::cout << "  Rank (est./eff./stab): " << s.rank_estimate << " / " << s.effective_rank << " / " << s.stable_rank << "\n";
    std::cout << "  Lambda max / min+  : " << s.lambda_max << " / " << s.lambda_min_positive << "\n";
    std::cout << "  Condition number   : " << s.condition_number << "\n";
    std::cout << "  Spectral gap       : " << s.spectral_gap << "\n";

    print_vector(s.top_eigenvalues,        "  Top eigenvalues");
    print_vector(s.eigenvalue_percentiles, "  Eigenvalue percentiles");

    std::cout << "\n";

    // ---------- C. Geometry / data structure ----------
    std::cout << "[NTK] Geometry / similarity structure\n";
    std::cout << "  Diag   mean / std  : " << s.diag_mean    << " / " << s.diag_std    << "\n";
    std::cout << "  Off-diag mean / std: " << s.offdiag_mean << " / " << s.offdiag_std << "\n";

    if (!std::isnan(s.within_class_mean) && !std::isnan(s.between_class_mean)) {
        std::cout << "  Within-class  mean / std : " << s.within_class_mean  << " / " << s.within_class_std  << "\n";
        std::cout << "  Between-class mean / std : " << s.between_class_mean << " / " << s.between_class_std << "\n";
        std::cout << "  Class separation score   : " << s.class_separation_score << "\n";
    }

    print_vector(s.per_class_self_sim, "  Per-class self-similarity");
    std::cout << "\n";

    // ---------- D. Dynamics / learning rate ----------
    std::cout << "[NTK] Dynamics / learning-rate diagnostics\n";
    std::cout << "  LR critical        : " << s.lr_crit << "\n";
    std::cout << "  LR safe            : " << s.lr_safe << "\n";

    if (s.lr_recommended_range.first > 0.0 || s.lr_recommended_range.second > 0.0) {
        std::cout << "  Suggested LR range : [" << s.lr_recommended_range.first  << ", " << s.lr_recommended_range.second << "]\n";
    }

    std::cout << "\n";

    // ---------- E. KRR / generalization proxy ----------
    if (s.krr_ran) {
        std::cout << "[NTK] Kernel ridge regression (lazy infinite-width proxy)\n";
        std::cout << "  Ridge lambda       : " << s.ridge_lambda_used << "\n";
        std::cout << "  Train loss (KRR)   : " << s.train_loss_krr << "\n";
        std::cout << "  Train accuracy (KRR): " << s.train_accuracy_krr * 100.0 << "%\n";
        std::cout << "  Val loss (KRR)     : " << s.val_loss_krr << "\n";
        std::cout << "  Val accuracy (KRR) : " << s.val_accuracy_krr * 100.0 << "%\n";
        print_vector(s.per_class_val_accuracy_krr, "  Per-class val acc (KRR, %)", /*scale=*/100.0);
        std::cout << "  Label alignment    : " << s.label_alignment << "\n";
        std::cout << "\n";
    }

    // ---------- F. Numerical sanity / implementation ----------
    std::cout << "[NTK] Numerical sanity / implementation\n";
    std::cout << "  PSD checked        : " << std::boolalpha << s.is_psd_checked << std::noboolalpha << "\n";
    std::cout << "  Max symmetry error : " << s.symmetry_error_max << "\n";
    std::cout << "  Max abs entry      : " << s.max_abs_entry << "\n";
    std::cout << "  CG iterations      : " << s.cg_iterations << "\n";
    std::cout << "  CG residual (abs / rel): " << s.cg_final_residual << " / " << s.cg_relative_residual << "\n";
    std::cout << "  CG converged       : " << std::boolalpha << s.cg_converged << std::noboolalpha << "\n";

    std::cout << "\n";

    // ---------- G. Raw kernel info ----------
    if (result.kernel.defined()) {
        auto kernel_cpu = result.kernel.to(torch::kCPU);
        std::cout << "[NTK] Kernel matrix shape: " << kernel_cpu.sizes() << "\n";
    }

    std::cout << "[NTK] ==================================================\n";

    return 0;
}
