#ifndef Nott_KERNEL_HPP
#define Nott_KERNEL_HPP
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

#include <torch/torch.h>

#include "../attention.hpp"

namespace Nott::Attention::Details {
    class ScaledDotProductKernelImpl : public torch::nn::Module {
    public:
        explicit ScaledDotProductKernelImpl(double dropout = 0.0,
                                            ::Nott::Attention::Variant variant = ::Nott::Attention::Variant::Full)
            : variant_(variant),
              dropout_(register_module("dropout", torch::nn::Dropout(torch::nn::DropoutOptions(dropout)))) {}

        torch::Tensor forward(const torch::Tensor& query,
                              const torch::Tensor& key,
                              const torch::Tensor& value,
                              const torch::Tensor& attn_mask = {},
                              const torch::Tensor& key_padding_mask = {})
        {
            // no-mask fast path: skip building any mask tensor, let SDPA's is_causal do it
            const bool has_key_padding_mask = key_padding_mask.defined() && key_padding_mask.numel() > 0;
            const bool has_attn_mask = attn_mask.defined() && attn_mask.numel() > 0;
            if (!has_key_padding_mask && !has_attn_mask) {
                const double dropout_p = is_training() ? dropout_->options.p() : 0.0;
                return at::scaled_dot_product_attention(
                    query, key, value,
                    /*attn_mask=*/std::nullopt,
                    dropout_p,
                    /*is_causal=*/variant_ == ::Nott::Attention::Variant::Causal);
            }

            // combined is additive (same convention this class's attn_mask already used, and
            // what SDPA's float attn_mask expects), so key_padding_mask/causal fold into it
            // instead of running the old eager matmul+softmax+matmul path.
            const auto batch_size = query.size(0);
            const auto num_heads = query.size(1);
            const auto target_len = query.size(2);
            const auto source_len = key.size(2);
            auto combined = torch::zeros({batch_size, num_heads, target_len, source_len}, query.options());

            if (has_key_padding_mask) {
                auto mask = key_padding_mask.to(torch::kBool).unsqueeze(1).unsqueeze(2);
                combined = combined.masked_fill(mask, -std::numeric_limits<float>::infinity());
            }

            if (has_attn_mask) {
                auto mask = attn_mask;

                const auto make_size_mismatch_error = [&](const std::string& reason) {
                    throw std::invalid_argument("Attention mask dimensions mismatch: " + reason +
                                                ". Expected (batch=" + std::to_string(batch_size) +
                                                ", heads=" + std::to_string(num_heads) +
                                                ", target=" + std::to_string(target_len) +
                                                ", source=" + std::to_string(source_len) + ") but got " +
                                                std::to_string(mask.dim()) + "D mask.");
                };

                switch (mask.dim()) {
                case 2:
                    if (mask.size(0) != target_len || mask.size(1) != source_len) {
                        make_size_mismatch_error("2D mask must match target and source dimensions");
                    }
                    mask = mask.unsqueeze(0).unsqueeze(0);
                    break;
                case 3:
                    if (mask.size(1) != target_len || mask.size(2) != source_len) {
                        make_size_mismatch_error("3D mask must match target and source dimensions in the last two axes");
                    }

                    if (mask.size(0) == batch_size) {
                        mask = mask.unsqueeze(1);
                    } else if (mask.size(0) == batch_size * num_heads) {
                        mask = mask.view({batch_size, num_heads, target_len, source_len});
                    } else {
                        make_size_mismatch_error("3D mask batch dimension must equal batch_size or batch_size * num_heads");
                    }
                    break;
                case 4:
                    if (mask.size(0) != batch_size || mask.size(1) != num_heads || mask.size(2) != target_len ||
                        mask.size(3) != source_len) {
                        make_size_mismatch_error("4D mask must match (batch, heads, target, source)");
                    }
                    break;
                default:
                    throw std::invalid_argument("Unsupported attention mask dimensionality: " + std::to_string(mask.dim()));
                }
                combined = combined + mask.to(combined.dtype());
            }

            if (variant_ == ::Nott::Attention::Variant::Causal) {
                auto causal_mask = torch::ones({target_len, source_len}, combined.options().dtype(torch::kBool)).triu(1);
                combined = combined.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), -std::numeric_limits<float>::infinity());
            }

            const double dropout_p = is_training() ? dropout_->options.p() : 0.0;
            return at::scaled_dot_product_attention(
                query, key, value,
                combined,
                dropout_p,
                /*is_causal=*/false);
        }

    private:
        ::Nott::Attention::Variant variant_{::Nott::Attention::Variant::Full};
        torch::nn::Dropout dropout_{nullptr};
    };

    TORCH_MODULE(ScaledDotProductKernel);
}
#endif //Nott_KERNEL_HPP