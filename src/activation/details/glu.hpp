#ifndef Nott_GLU_HPP
#define Nott_GLU_HPP
/// "Language Modeling with Gated Convolutional Networks" (GLU activation) https://arxiv.org/pdf/1612.08083

#include <torch/torch.h>

#include <utility>

#include "../activation.hpp"

namespace Nott::Activation::Details {

    struct GLU {
        [[nodiscard]] torch::Tensor operator()(torch::Tensor input) const {
            if (input.dim() == 0) {
                return input;
            }
            const auto dim = input.dim() - 1;
            return torch::glu(std::move(input), dim);
        }
    };

}

#endif //Nott_GLU_HPP