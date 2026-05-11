#ifndef Nott_DATA_TRANSFORM_NORMALIZATION_COMMON_HPP
#define Nott_DATA_TRANSFORM_NORMALIZATION_COMMON_HPP
#include <torch/torch.h>
#include <vector>
#include <algorithm>
#include <stdexcept>

namespace Nott::Data::Transform::Normalization::Details {
    inline void check_temporal_dim(const at::Tensor& x, int64_t temporal_dim) {
        TORCH_CHECK(x.defined(), "Input tensor must be defined");
        TORCH_CHECK(temporal_dim >= 0 && temporal_dim < x.dim(), "temporal_dim out of range: ", temporal_dim, " for x.dim()=", x.dim());
        TORCH_CHECK(x.size(temporal_dim) >= 2, "Need at least 2 steps along temporal_dim");
    }

    inline at::Tensor to_float32(const at::Tensor& x) {
        return x.scalar_type() == at::kFloat ? x : x.to(at::kFloat);
    }

    namespace Options {
        struct PermutePlan {
            std::vector<int64_t> perm;     // to move time -> last
            std::vector<int64_t> invperm;  // to restore
            bool already_last{false};
        };
    }

    inline Options::PermutePlan make_move_time_last_plan(const at::Tensor& x, int64_t temporal_dim) {
        const int64_t D = x.dim();
        Options::PermutePlan plan{};
        if (temporal_dim == D - 1) {
            plan.already_last = true;
            return plan;
        }
        plan.perm.reserve(D);
        for (int64_t d = 0; d < D; ++d) if (d != temporal_dim) plan.perm.push_back(d);
        plan.perm.push_back(temporal_dim);

        plan.invperm.resize(D);
        for (int64_t i = 0; i < D; ++i) plan.invperm[plan.perm[i]] = i;
        return plan;
    }

    inline at::Tensor move_time_last(const at::Tensor& x, const Options::PermutePlan& plan) {
        return plan.already_last ? x : x.permute(plan.perm).contiguous();
    }

    inline at::Tensor move_time_restore(const at::Tensor& y, const Options::PermutePlan& plan) {
        return plan.already_last ? y : y.permute(plan.invperm).contiguous();
    }

    inline at::Tensor flatten_batch_time_last(const at::Tensor& x_last, int64_t* N_out, int64_t* T_out) {
        const int64_t D = x_last.dim();
        const int64_t T = x_last.size(D - 1);
        int64_t N = 1;
        for (int64_t i = 0; i < D - 1; ++i) N *= x_last.size(i);
        if (N == 1) {
            *N_out = 1; *T_out = T;
            return x_last.reshape({1, T});
        } else {
            *N_out = N; *T_out = T;
            return x_last.reshape({N, T});
        }
    }

    inline at::Tensor unflatten_batch_time_last(const at::Tensor& y_2d, const at::IntArrayRef& original_sizes, const Options::PermutePlan& plan) {
        std::vector<int64_t> new_sizes(original_sizes.begin(), original_sizes.end());
        auto y_last = (y_2d.size(0) == 1)
            ? y_2d.reshape({new_sizes.back()})
            : y_2d.reshape(new_sizes);
        return move_time_restore(y_last, plan);
    }

    inline at::Tensor clamp_std(const at::Tensor& s, double eps) {
        return s.clamp_min(eps);
    }

    inline at::Tensor safe_logit(const at::Tensor& u, double eps=1e-6) {
        auto v = u.clamp(eps, 1.0 - eps);
        return torch::log(v) - torch::log1p(-v);
    }

    inline at::Tensor safe_probit(const at::Tensor& u, double eps=1e-6) {
        auto v = u.clamp(eps, 1.0 - eps);
        /// probit(u) = sqrt(2) * erfinv(2u-1)
        return std::sqrt(2.0) * at::erfinv(2.0 * v - 1.0);
    }
}
#endif //Nott_DATA_TRANSFORM_NORMALIZATION_COMMON_HPP