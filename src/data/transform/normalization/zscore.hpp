#ifndef Nott_DATA_TRANSFORM_NORMALIZATION_ZSCORE_HPP
#define Nott_DATA_TRANSFORM_NORMALIZATION_ZSCORE_HPP

#include <torch/torch.h>
#include <algorithm>
#include "common.hpp"

namespace Nott::Data::Transform::Normalization {
    namespace Options {
        struct ZscoreOptions {
            int64_t lag = 32;          // window size if forward_only
            int64_t temporal_dim = 0;  // which axis is time
            bool forward_only = true;  // rolling vs static (fit on head[0:lag])
            double eps = 1e-12;
        };
        struct EWZscoreOptions {
            double alpha = 0.1;        // EW factor in (0,1]
            int64_t temporal_dim = 0;
            double eps = 1e-12;
        };
        struct StandardizeToTargetOptions {
            int64_t temporal_dim = 0;
            double target_mean = 0.0;
            double target_std  = 1.0;
            bool forward_only = false; // if true, use rolling stats
            int64_t lag = 128;
            double eps = 1e-12;
        };
        struct DemeanOptions {
            int64_t temporal_dim = 0;
            bool forward_only = false;
            int64_t lag = 128; // if forward_only, use rolling mean
        };
        struct RobustZscoreOptions {
            int64_t temporal_dim = 0;
            bool static_baseline = true; // fit on head[0:lag] if true, else whole series
            int64_t lag = 256;           // used when static_baseline
            double c_mad = 1.4826;       // to estimate sigma from MAD
            double eps = 1e-12;
        };
    }

    inline at::Tensor Zscore(const at::Tensor& x_in, const Options::ZscoreOptions o) {
        using namespace Details;
        check_temporal_dim(x_in, o.temporal_dim);

        const auto plan = make_move_time_last_plan(x_in, o.temporal_dim);
        auto x0 = move_time_last(to_float32(x_in), plan);
        const int64_t T = x0.size(-1);

        auto y = at::empty_like(x0);
        if (o.forward_only) {
            for (int64_t t = 0; t < T; ++t) {
                const int64_t start = std::max<int64_t>(0, t + 1 - o.lag);
                const int64_t len   = (t + 1) - start;

                auto win = x0.narrow(/*dim=*/-1, /*start=*/start, /*length=*/len);
                auto m = win.mean(-1, /*keepdim=*/true);
                auto s = clamp_std(win.std(-1, /*unbiased=*/false, /*keepdim=*/true), o.eps);

                auto xt = x0.select(/*dim=*/-1, /*index=*/t).unsqueeze(-1);
                auto zt = (xt - m) / s; // shape: [..., 1]

                y.select(-1, t).copy_(zt.squeeze(-1));
            }
        } else {
            TORCH_CHECK(o.lag > 0 && o.lag < T, "lag must be in (0, T) for static mode");
            auto head = x0.narrow(-1, 0, o.lag);
            auto m = head.mean(-1, /*keepdim=*/true);
            auto s = clamp_std(head.std(-1, /*unbiased=*/false, /*keepdim=*/true), o.eps);
            y = (x0 - m) / s;
        }
        return move_time_restore(y, plan);
    }

    /// Exponentially-weighted z-score (one-pass, leakage-safe)
    inline at::Tensor EWZscore(const at::Tensor& x_in, const Options::EWZscoreOptions o) {
        using namespace Details;
        TORCH_CHECK(o.alpha > 0.0 && o.alpha <= 1.0, "alpha must be in (0,1]");
        check_temporal_dim(x_in, o.temporal_dim);

        const auto plan = make_move_time_last_plan(x_in, o.temporal_dim);
        auto x0 = move_time_last(to_float32(x_in), plan);
        const int64_t T = x0.size(-1);
        auto y = at::empty_like(x0);

        /// Initialize from first point
        auto m_prev = x0.select(-1, 0);
        auto v_prev = at::zeros_like(m_prev);

        /// t=0 -> z=0 by convention
        y.select(-1, 0).zero_();

        const double a = o.alpha;
        for (int64_t t = 1; t < T; ++t) {
            auto xt = x0.select(-1, t);
            auto delta = xt - m_prev;
            auto m_t = m_prev + a * delta;

            /// EW variance (stable form)
            auto v_t = (1.0 - a) * (v_prev + a * delta * (xt - m_t));
            auto s_t = clamp_std(v_t.sqrt(), o.eps);
            auto zt  = (xt - m_t) / s_t;

            y.select(-1, t).copy_(zt);
            m_prev = m_t;
            v_prev = v_t;
        }
        return move_time_restore(y, plan);
    }

    /// Robust Z using median/MAD (static baseline to avoid heavy rolling medians)
    inline at::Tensor RobustZscore(const at::Tensor& x_in, const Options::RobustZscoreOptions opt) {
        using namespace Details;
        check_temporal_dim(x_in, opt.temporal_dim);

        const auto plan = make_move_time_last_plan(x_in, opt.temporal_dim);
        auto x0 = move_time_last(to_float32(x_in), plan);
        const int64_t T = x0.size(-1);

        at::Tensor base;
        if (opt.static_baseline) {
            TORCH_CHECK(opt.lag > 1 && opt.lag <= T, "lag must be in (1, T]");
            base = x0.narrow(-1, 0, opt.lag);
        } else {
            base = x0; // use entire series for baseline (leakage if used online)
        }

        /// Median via kthvalue (works on last dim)
        auto med = std::get<0>(base.kthvalue((base.size(-1) + 1) / 2, -1, /*keepdim=*/true));
        auto mad = (base - med).abs();
        mad = std::get<0>(mad.kthvalue((mad.size(-1) + 1) / 2, -1, /*keepdim=*/true));

        auto sigma = clamp_std(opt.c_mad * mad, opt.eps);
        auto z = (x0 - med) / sigma;
        return move_time_restore(z, plan);
    }

    inline at::Tensor Demean(const at::Tensor& x_in, const Options::DemeanOptions opt) {
        using namespace Details;
        check_temporal_dim(x_in, opt.temporal_dim);
        const auto plan = make_move_time_last_plan(x_in, opt.temporal_dim);
        auto x0 = move_time_last(to_float32(x_in), plan);
        const int64_t T = x0.size(-1);
        auto y = at::empty_like(x0);

        if (opt.forward_only) {
            for (int64_t t = 0; t < T; ++t) {
                const int64_t start = std::max<int64_t>(0, t + 1 - opt.lag);
                const int64_t len   = (t + 1) - start;

                auto m = x0.narrow(-1, start, len).mean(-1, /*keepdim=*/true);
                auto xt = x0.select(-1, t).unsqueeze(-1);
                y.select(-1, t).copy_((xt - m).squeeze(-1));
            }
        } else {
            auto m = x0.mean(-1, /*keepdim=*/true);
            y = x0 - m;
        }
        return move_time_restore(y, plan);
    }

    inline at::Tensor StandardizeToTarget(const at::Tensor& x_in, const Options::StandardizeToTargetOptions opt) {
        using namespace Details;
        auto z = Zscore(x_in, ::Nott::Data::Transform::Normalization::Options::ZscoreOptions{
            .lag = opt.lag,
            .temporal_dim = opt.temporal_dim,
            .forward_only = opt.forward_only,
            .eps = opt.eps
        });
        return z * static_cast<float>(opt.target_std) + static_cast<float>(opt.target_mean);
    }

}
#endif // Nott_DATA_TRANSFORM_NORMALIZATION_ZSCORE_HPP
