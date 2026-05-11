#ifndef Nott_MISH_HPP
#define Nott_MISH_HPP
/// "Mish: A Self Regularized Non-Monotonic Neural Activation Function" https://arxiv.org/pdf/1908.08681
#include <torch/torch.h>

#include <utility>

#include "../activation.hpp"

namespace Nott::Activation::Details {

    struct Mish {
        [[nodiscard]] torch::Tensor operator()(torch::Tensor input) const {
            auto tensor = std::move(input);
            return tensor * torch::tanh(torch::softplus(tensor));
        }
    };

}

#endif //Nott_MISH_HPP