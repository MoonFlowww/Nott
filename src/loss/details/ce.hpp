#ifndef Nott_CE_HPP
#define Nott_CE_HPP
#include <optional>
#include <stdexcept>
#include <torch/torch.h>
#include <vector>

#include "reduction.hpp"

namespace Nott::Loss::Details {
    struct CrossEntropyOptions {
        Reduction reduction{Reduction::Mean};
        std::vector<double> weight{};
        double label_smoothing{0.0};
    };

    struct CrossEntropyDescriptor {
        CrossEntropyOptions options{};

        // Pre-built options for the no-weight case - avoids reconstruction every step.
        mutable std::optional<torch::nn::functional::CrossEntropyFuncOptions> cached_opts{};

        [[nodiscard]] const torch::nn::functional::CrossEntropyFuncOptions& base_opts() const {
            if (!cached_opts) {
                auto opts = torch::nn::functional::CrossEntropyFuncOptions{};
                opts = opts.reduction(to_torch_reduction<torch::nn::functional::CrossEntropyFuncOptions>(options.reduction));
                opts = opts.label_smoothing(options.label_smoothing);
                cached_opts = std::move(opts);
            }
            return *cached_opts;
        }
    };

    inline torch::Tensor compute(const CrossEntropyDescriptor& descriptor,
                                 const torch::Tensor& prediction,
                                 const torch::Tensor& target,
                                 const std::optional<torch::Tensor>& weight = std::nullopt) {
        if (descriptor.options.weight.empty() && !weight.has_value()) {
            return torch::nn::functional::cross_entropy(prediction, target, descriptor.base_opts());
        }

        auto opts = descriptor.base_opts();
        if (!descriptor.options.weight.empty()) {
            opts = opts.weight(torch::tensor(
                descriptor.options.weight,
                torch::TensorOptions().dtype(prediction.scalar_type()).device(prediction.device())));
        } else if (weight.has_value() && weight->defined()) {
            opts = opts.weight(weight->to(prediction.device(), prediction.scalar_type()));
        }
        return torch::nn::functional::cross_entropy(prediction, target, opts);
    }
}
#endif //Nott_CE_HPP