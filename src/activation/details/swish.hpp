#ifndef Nott_SWISH_HPP
#define Nott_SWISH_HPP
/// "Searching for Activation Functions" (original SiLU/Swish proposal) https://arxiv.org/pdf/1710.05941
#include <torch/torch.h>

#include <utility>

#include "../activation.hpp"

namespace Nott::Activation::Details {

    struct Swish {
        [[nodiscard]] torch::Tensor operator()(torch::Tensor input) const {
            auto tensor = std::move(input);
            return tensor * torch::sigmoid(tensor);
        }
    };

}

#endif //Nott_SWISH_HPP