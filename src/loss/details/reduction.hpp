#ifndef Nott_LOSS_REDUCTION_HPP
#define Nott_LOSS_REDUCTION_HPP

#include <torch/torch.h>
#include <type_traits>

namespace Nott::Loss::Details {

    enum class Reduction { Mean, Sum, None };

    /// Use: to_torch_reduction<torch::nn::functional::KLDivFuncOptions>(Reduction::Mean)
    template <typename Options>
    inline typename Options::reduction_t to_torch_reduction(Reduction r) {
        using RT = typename Options::reduction_t;
        /// Helpful compile-time check: Options must expose nested reduction_t
        static_assert(!std::is_void_v<RT>, "Options must define nested type 'reduction_t'");

        switch (r) {
            case Reduction::Sum:  return RT{torch::kSum};
            case Reduction::None: return RT{torch::kNone};
            case Reduction::Mean:
            default:              return RT{torch::kMean};
        }
    }
    inline torch::Tensor apply_reduction(torch::Tensor loss, Reduction reduction) { // Non-torch base
        switch (reduction) {
            case Reduction::None:
                return loss;
            case Reduction::Sum:
                return loss.sum();
            case Reduction::Mean:
            default:
                return loss.mean();
        }
    }
    template <typename Options>
    inline typename Options::reduction_t to_torch_reduction(const Options&, Reduction r) {
        return to_torch_reduction<Options>(r);
    }

}

#endif // Nott_LOSS_REDUCTION_HPP
