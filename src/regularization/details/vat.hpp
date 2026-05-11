#ifndef Nott_REGULARIZATION_DETAILS_VAT_HPP
#define Nott_REGULARIZATION_DETAILS_VAT_HPP
/// "Virtual Adversarial Training: A Regularization Method for Supervised and Semi-supervised Learning" https://arxiv.org/pdf/1704.03976
#include <torch/torch.h>

#include "common.hpp"

namespace Nott::Regularization::Details {

    struct VATOptions {
        double coefficient{0.0};
    };

    struct VATDescriptor {
        VATOptions options{};
    };

    [[nodiscard]] inline torch::Tensor penalty(const VATDescriptor& descriptor, const torch::Tensor& logits)
    {
        (void)descriptor;
        return detail::zeros_like_optional(logits);
    }

    [[nodiscard]] inline torch::Tensor penalty(const VATDescriptor& descriptor,
                                               const torch::Tensor& logits,
                                               const torch::Tensor& perturbed_logits)
    {
        const auto& options = descriptor.options;
        if (options.coefficient == 0.0 || !logits.defined() || !perturbed_logits.defined()) {
            const auto& reference = logits.defined() ? logits : perturbed_logits;
            return detail::zeros_like_optional(reference);
        }

        auto p = torch::softmax(logits, -1);
        auto log_p = torch::log_softmax(logits, -1);
        auto log_q = torch::log_softmax(perturbed_logits, -1);
        auto kl = (p * (log_p - log_q)).sum(-1).mean();
        return kl.mul(options.coefficient);
    }

}

#endif // Nott_REGULARIZATION_DETAILS_VAT_HPP