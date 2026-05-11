#ifndef Nott_MSE_HPP
#define Nott_MSE_HPP

#include <optional>
#include <stdexcept>
#include <torch/torch.h>
#include "helper.hpp"

namespace Nott::Loss::Details {

    namespace F = torch::nn::functional;

    struct MSEOptions {
        Reduction reduction{Reduction::Mean};
        std::vector<double> weight{};
    };

    struct MSEDescriptor {
        MSEOptions options{};
    };


    inline torch::Tensor compute(const MSEDescriptor& descriptor,
                                 const torch::Tensor& prediction,
                                 const torch::Tensor& target,
                                 const std::optional<torch::Tensor>& weight = std::nullopt)
    {
        auto per_elem = F::mse_loss(
            prediction,
            target,
            F::MSELossFuncOptions().reduction(torch::kNone)
        );

        std::optional<torch::Tensor> weight_tensor{};
        if (!descriptor.options.weight.empty()) {
            auto tensor = torch::tensor(
                descriptor.options.weight,
                torch::TensorOptions().dtype(prediction.scalar_type()).device(prediction.device()));
            if (tensor.numel() == per_elem.numel()) {
                tensor = tensor.reshape(per_elem.sizes());
            }
            weight_tensor = tensor;
        } else if (weight && weight->defined()) {
            weight_tensor = weight->to(prediction.device(), prediction.scalar_type());
        }

        if (!weight_tensor) {
            /// Pure LibTorch MSE
            return F::mse_loss(
                prediction,
                target,
                torch::nn::functional::MSELossFuncOptions().reduction(to_torch_reduction<torch::nn::functional::MSELossFuncOptions>(descriptor.options.reduction))
            );
        }

        return apply_reduction_weighted(per_elem, *weight_tensor, descriptor.options.reduction);
    }


}




#endif // Nott_MSE_HPP
