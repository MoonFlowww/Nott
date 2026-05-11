#ifndef Nott_DATA_TRANSFORM_NORMALIZATION_EHLERS_HPP
#define Nott_DATA_TRANSFORM_NORMALIZATION_EHLERS_HPP

#include <torch/torch.h>
#include "common.hpp"


namespace Nott::Data::Transform::Normalization {

    /// Fisher transform on a bounded oscillator x in (-1,1).
    namespace Options {
        struct FisherOptions { double clamp = 0.999; };
    }

    inline at::Tensor FisherTransform(const at::Tensor& x_in, const Options::FisherOptions o = {}) {
        auto x = Details::to_float32(x_in).clamp(-static_cast<float>(o.clamp), static_cast<float>(o.clamp));
        /// 0.5 * ln((1+x)/(1-x)) = atanh(x)
        return 0.5f * (at::log1p(x) - at::log1p(-x));
    }

    inline at::Tensor InverseFisher(const at::Tensor& y_in) {
        auto y = Details::to_float32(y_in);
        return at::tanh(y); // tanh is the inverse of Fisher
    }


    /// TODO: Implement Ehler loops for 2d timeseries
}

#endif // Nott_DATA_TRANSFORM_NORMALIZATION_EHLERS_HPP
