#ifndef Nott_COSINE_EMBEDDING_HPP
#define Nott_COSINE_EMBEDDING_HPP

#include <optional>
#include <stdexcept>
#include <torch/torch.h>

#include "reduction.hpp"
/// TODO: rework
/// - using single path : std::visit([&](const auto& d){ return Loss::Details::compute(d, prediction, target, weight); }, *loss_descriptor_);
///      --> Cos Embedd expect 2 inputs
namespace Nott::Loss::Details {

    struct CosineEmbeddingOptions {
        Reduction reduction{Reduction::Mean};
        double margin{0.0};
    };

    struct CosineEmbeddingDescriptor {
        CosineEmbeddingOptions options{};
    };

    /// Compat signature to match Model::compute_loss(...):
    /// prediction must contain a pair: [B, 2, D...] or [2, B, D...]
    inline torch::Tensor compute(const CosineEmbeddingDescriptor& descriptor,
                                 const torch::Tensor& prediction,
                                 const torch::Tensor& target,
                                 const std::optional<torch::Tensor>& weight = std::nullopt) {
        if (weight && weight->defined()) {
            throw std::invalid_argument("CosineEmbedding loss does not support weighted reduction.");
        }

        TORCH_CHECK(prediction.dim() >= 2,
                    "CosineEmbedding expects a pair tensor with a dimension of size 2, got ", prediction.sizes());

        int64_t pair_axis = -1;
        if (prediction.size(1) == 2) {
            pair_axis = 1;                           // [B, 2, D...]
        } else if (prediction.size(0) == 2) {
            pair_axis = 0;                           // [2, B, D...]
        } else {
            TORCH_CHECK(false,
                        "CosineEmbedding expects prediction shaped [B, 2, D...] or [2, B, D...], got ",
                        prediction.sizes());
        }

        auto x1 = prediction.select(pair_axis, 0).contiguous();
        auto x2 = prediction.select(pair_axis, 1).contiguous();

        auto opts = torch::nn::functional::CosineEmbeddingLossFuncOptions{}
                        .margin(descriptor.options.margin)
                        .reduction(to_torch_reduction<torch::nn::CosineEmbeddingLossOptions>(
                            descriptor.options.reduction));

        /// y should be 1D with {-1, +1}
        auto y = target.to(x1.device(), x1.scalar_type());
        if (y.dim() != 1) {
            y = y.reshape({-1});
        }

        return torch::nn::functional::cosine_embedding_loss(x1, x2, y, opts);
    }

}

#endif // Nott_COSINE_EMBEDDING_HPP
