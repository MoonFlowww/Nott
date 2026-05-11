#ifndef Nott_SWIGLU_HPP
#define Nott_SWIGLU_HPP
/// "PaLM: Scaling Language Modeling with Pathways" (introduces SwiGLU) https://arxiv.org/pdf/2204.02311
#include <torch/torch.h>

#include <utility>
#include <vector>

#include "../activation.hpp"

namespace Nott::Activation::Details {

    struct SwiGLU {
        [[nodiscard]] torch::Tensor operator()(torch::Tensor input) const {
            if (input.dim() == 0) {
                return input;
            }
            const auto dim = input.dim() - 1;
            auto parts = input.chunk(2, dim);
            if (parts.size() < 2) {
                return input;
            }
            return torch::silu(parts[0]) * parts[1];
        }
    };

}

#endif //Nott_SWIGLU_HPP