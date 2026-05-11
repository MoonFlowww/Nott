#ifndef Nott_LOSS_KL_HPP
#define Nott_LOSS_KL_HPP
#include <optional>
#include <stdexcept>

#include <torch/torch.h>

#include "reduction.hpp"

namespace Nott::Loss::Details {
    struct KLDivOptions {
        Reduction reduction{Reduction::Mean};
        bool log_target{false};
        bool use_batch_mean{false};
        int64_t log_softmax_dim{1};
        bool prediction_is_log{false};
    };

    struct KLDivDescriptor {
        KLDivOptions options{};
    };

    inline torch::Tensor compute(const KLDivDescriptor& descriptor, const torch::Tensor& prediction, const torch::Tensor& target,
                                 const std::optional<torch::Tensor>& weight = std::nullopt)  {
        if (weight && weight->defined()) {
            throw std::invalid_argument("KLDiv loss does not support weighted reduction.");
        }

        auto input = prediction;
        if (!descriptor.options.prediction_is_log) {
            auto log_softmax_opts = torch::nn::functional::LogSoftmaxFuncOptions(descriptor.options.log_softmax_dim);
            input = torch::nn::functional::log_softmax(prediction, log_softmax_opts);
        }

        auto tgt = target.to(input.device(), input.scalar_type());

        /// If target is given as class indices (int64), convert to one-hot prob
        if (tgt.dtype() == torch::kLong) {
            const auto dim = descriptor.options.log_softmax_dim;
            const auto num_classes = input.size(dim);
            tgt = torch::one_hot(tgt, num_classes).to(input.scalar_type());
            tgt = tgt / tgt.sum(dim, /*keepdim=*/true).clamp_min(1e-12);
        }

        auto opts = torch::nn::functional::KLDivFuncOptions{};
        if (descriptor.options.use_batch_mean) {
            opts = opts.reduction(torch::kBatchMean);
        } else {
            opts = opts.reduction(to_torch_reduction<torch::nn::functional::KLDivFuncOptions>(descriptor.options.reduction));
        }
        opts = opts.log_target(descriptor.options.log_target);

        return torch::nn::functional::kl_div(input, tgt, opts);
    }
}
#endif //Nott_LOSS_KL_HPP