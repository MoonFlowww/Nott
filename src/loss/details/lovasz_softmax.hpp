#ifndef Nott_LOSS_LOVASZ_SOFTMAX_HPP
#define Nott_LOSS_LOVASZ_SOFTMAX_HPP

#include <optional>
#include <stdexcept>
#include <vector>

#include <torch/torch.h>

#include "reduction.hpp"

namespace Nott::Loss::Details {

    namespace LovaszSoftmaxInternal {
        inline torch::Tensor lovasz_grad(torch::Tensor sorted_ground_truth)
        {
            auto length = sorted_ground_truth.size(0);
            auto gts = sorted_ground_truth.sum();
            auto intersection = gts - sorted_ground_truth.cumsum(0);
            auto union_ = gts + (1.0 - sorted_ground_truth).cumsum(0);
            auto jaccard = 1.0 - intersection / union_.clamp_min(1e-12);

            if (length > 1) {
                // narrow() returns a view: without the clone, this in-place subtract reads
                // and writes overlapping memory (jaccard[1:] aliases jaccard[0:-1]).
                auto jaccard_shifted = jaccard.narrow(0, 1, length - 1).clone();
                jaccard_shifted -= jaccard.narrow(0, 0, length - 1);
                jaccard.narrow(0, 1, length - 1).copy_(jaccard_shifted);
            }
            return jaccard;
        }

        inline std::pair<torch::Tensor, torch::Tensor> flatten_probas(torch::Tensor probas,
                                                                      torch::Tensor labels,
                                                                      int64_t ignore_index)
        {
            if (probas.dim() < 2) {
                throw std::invalid_argument("Lovasz-Softmax expects predictions with shape [N, C, ...].");
            }

            const auto classes = probas.size(1);

            std::vector<int64_t> permute_dims;
            permute_dims.reserve(probas.dim());
            permute_dims.push_back(0);
            for (int64_t dim = 2; dim < probas.dim(); ++dim) {
                permute_dims.push_back(dim);
            }
            permute_dims.push_back(1);

            auto permuted = probas.permute(permute_dims).contiguous();
            auto probas_flat = permuted.reshape({-1, classes});
            auto labels_flat = labels.reshape({-1});

            if (ignore_index >= 0) {
                auto mask = labels_flat != ignore_index;
                if (!mask.any().item<bool>()) {
                    return {
                        torch::empty({0, classes}, probas.options()),
                        torch::empty({0}, labels.options().dtype(torch::kLong))};
                }
                probas_flat = probas_flat.index({mask});
                labels_flat = labels_flat.index({mask});
            }

            return {probas_flat, labels_flat.to(torch::kLong)};
        }

        inline torch::Tensor lovasz_softmax_flat(torch::Tensor probas,
                                                 torch::Tensor labels,
                                                 bool include_background,
                                                 bool only_present_classes,
                                                 int64_t ignore_index)
        {
            const auto classes = probas.size(1);
            auto [probas_flat, labels_flat] = flatten_probas(probas, labels, ignore_index);

            if (probas_flat.numel() == 0 || labels_flat.numel() == 0) {
                return torch::zeros({}, probas.options());
            }

            std::vector<torch::Tensor> class_losses;
            class_losses.reserve(classes);

            for (int64_t class_index = 0; class_index < classes; ++class_index) {
                if (!include_background && class_index == 0) {
                    continue;
                }

                auto fg = labels_flat.eq(class_index);
                if (only_present_classes && !fg.any().item<bool>()) {
                    continue;
                }

                auto class_pred = probas_flat.select(1, class_index);
                auto errors = (fg.to(class_pred.options().dtype()) - class_pred).abs();
                auto order = torch::argsort(errors, /*dim=*/0, /*descending=*/true);
                auto sorted_errors = errors.index_select(0, order);
                auto sorted_fg = fg.to(class_pred.options().dtype()).index_select(0, order);
                auto grad = lovasz_grad(sorted_fg);
                class_losses.push_back(torch::dot(sorted_errors, grad));
            }

            if (class_losses.empty()) {
                return torch::zeros({}, probas.options());
            }

            return torch::stack(class_losses).mean();
        }
    } // namespace LovaszSoftmaxInternal

    struct LovaszSoftmaxOptions {
        Reduction reduction{Reduction::Mean};
        bool per_image{false};
        int64_t ignore_index{-1};
        bool apply_softmax{true};
        bool include_background{true};
        bool only_present_classes{true};
    };

    struct LovaszSoftmaxDescriptor {
        LovaszSoftmaxOptions options{};
    };

    inline torch::Tensor compute(const LovaszSoftmaxDescriptor& descriptor,
                                 const torch::Tensor& prediction,
                                 const torch::Tensor& target,
                                 const std::optional<torch::Tensor>& weight = std::nullopt)
    {
        if (!prediction.defined() || !target.defined()) {
            throw std::invalid_argument("Lovasz-Softmax loss requires defined prediction and target tensors.");
        }

        auto pred = prediction;
        if (descriptor.options.apply_softmax) {
            pred = torch::nn::functional::softmax(pred, torch::nn::functional::SoftmaxFuncOptions(1));
        }

        auto tgt = target.to(pred.device());
        if (tgt.dim() == pred.dim()) {
            tgt = tgt.argmax(1);
        }

        std::vector<torch::Tensor> losses;
        losses.reserve(descriptor.options.per_image ? pred.size(0) : 1);

        auto compute_sample = [&](const torch::Tensor& sample_pred, const torch::Tensor& sample_target) {
            auto loss = LovaszSoftmaxInternal::lovasz_softmax_flat(
                sample_pred,
                sample_target,
                descriptor.options.include_background,
                descriptor.options.only_present_classes,
                descriptor.options.ignore_index);
            if (loss.defined()) {
                losses.push_back(loss);
            }
        };

        if (descriptor.options.per_image) {
            for (int64_t index = 0; index < pred.size(0); ++index) {
                compute_sample(pred.index({index}).unsqueeze(0), tgt.index({index}).unsqueeze(0));
            }
        } else {
            compute_sample(pred, tgt);
        }

        if (losses.empty()) {
            return torch::zeros({}, pred.options());
        }

        auto stacked = torch::stack(losses);
        if (weight && weight->defined()) {
            return apply_reduction_weighted(stacked, *weight, descriptor.options.reduction);
        }
        return apply_reduction(stacked, descriptor.options.reduction);
    }

}

#endif // Nott_LOSS_LOVASZ_SOFTMAX_HPP