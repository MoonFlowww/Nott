#ifndef Nott_DATA_TRANSFORM_NORMALIZATION_POWER_HPP
#define Nott_DATA_TRANSFORM_NORMALIZATION_POWER_HPP

#include <torch/torch.h>
#include <cmath>
#include "common.hpp"

namespace Nott::Data::Transform::Normalization {

    namespace Options{
        struct BoxCoxOptions {
            double lambda = 0.0;   // 0 -> log
            double shift  = 0.0;   // ensure x + shift > 0
            double eps    = 1e-12;
        };
        struct YeoJohnsonOptions {
            double lambda = 0.0; // covers negatives
            double eps = 1e-12;
        };
    }

    /// y = ((x+s)^lambda - 1) / lambda, lambda!=0;  log(x+s) if lambda==0
    inline at::Tensor BoxCox(const at::Tensor& x_in, const Options::BoxCoxOptions o = {}) {
        auto x = Details::to_float32(x_in) + static_cast<float>(o.shift);
        auto xp = x.clamp_min(o.eps);
        if (std::abs(o.lambda) < 1e-12) {
            return at::log(xp);
        } else {
            auto y = at::pow(xp, static_cast<float>(o.lambda)) - 1.0;
            return y / static_cast<float>(o.lambda);
        }
    }

    inline at::Tensor YeoJohnson(const at::Tensor& x_in, const Options::YeoJohnsonOptions o = {}) {
        auto x = Details::to_float32(x_in);
        auto pos = (x >= 0);
        auto xp = at::where(pos, x + 1.0, 1.0 - x);

        at::Tensor y_pos, y_neg;
        if (std::abs(o.lambda) < 1e-12) {
            y_pos =  at::log(xp);
            y_neg = -at::log(xp);
        } else {
            y_pos = (at::pow(xp, static_cast<float>(o.lambda)) - 1.0) / static_cast<float>(o.lambda);
            y_neg = -(at::pow(xp, static_cast<float>(2.0 - o.lambda)) - 1.0) / static_cast<float>(2.0 - o.lambda);
        }
        return at::where(pos, y_pos, y_neg);
    }

    /// Signed power: sign(x)*|x|^p (safe for fractional p)
    inline at::Tensor SignedPower(const at::Tensor& x_in, double p) {
        auto x = Details::to_float32(x_in);
        return x.sign() * at::pow(x.abs(), static_cast<float>(p));
    }

}

#endif // Nott_DATA_TRANSFORM_NORMALIZATION_POWER_HPP
