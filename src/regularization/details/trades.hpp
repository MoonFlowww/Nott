#ifndef Nott_REGULARIZATION_DETAILS_TRADES_HPP
#define Nott_REGULARIZATION_DETAILS_TRADES_HPP
/// "Theoretically Principled Trade-off between Robustness and Accuracy" (TRADES) https://arxiv.org/pdf/1901.08573
#include <torch/torch.h>

#include "common.hpp"

namespace Nott::Regularization::Details {

    struct TRADESOptions {
        double coefficient{0.0};
    };

    struct TRADESDescriptor {
        TRADESOptions options{};
    };

    [[nodiscard]] inline torch::Tensor penalty(const TRADESDescriptor& descriptor, const torch::Tensor& logits)
    {
        (void)descriptor;
        return detail::zeros_like_optional(logits);
    }

    [[nodiscard]] inline torch::Tensor penalty(const TRADESDescriptor& descriptor,
                                               const torch::Tensor& logits,
                                               const torch::Tensor& adversarial_logits)
    {
        const auto& options = descriptor.options;
        if (options.coefficient == 0.0 || !logits.defined() || !adversarial_logits.defined()) {
            const auto& reference = logits.defined() ? logits : adversarial_logits;
            return detail::zeros_like_optional(reference);
        }

        auto p = torch::softmax(logits, -1);
        auto log_p = torch::log_softmax(logits, -1);
        auto log_q = torch::log_softmax(adversarial_logits, -1);
        auto kl = (p * (log_p - log_q)).sum(-1).mean();
        return kl.mul(options.coefficient);
    }

}

#endif // Nott_REGULARIZATION_DETAILS_TRADES_HPP