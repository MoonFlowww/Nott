#ifndef Nott_LRSCHEDULER_COSINEANNEALING_HPP
#define Nott_LRSCHEDULER_COSINEANNEALING_HPP
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>
/// "SGDR: Stochastic Gradient Descent with Warm Restarts" (cosine annealing) https://arxiv.org/pdf/1608.03983
#include <torch/torch.h>

#include "common.hpp"

namespace Nott::LrScheduler::Details {
    struct CosineAnnealingOptions {
        std::size_t T_max{1};
        double eta_min{0.0};
        std::size_t warmup_steps{0};
        double warmup_start_factor{0.0};
    };

    struct CosineAnnealingDescriptor {
        CosineAnnealingOptions options{};
    };

    class CosineAnnealingScheduler final : public Scheduler {
    public:
        CosineAnnealingScheduler(torch::optim::Optimizer& optimizer, CosineAnnealingOptions options)
            : optimizer_(optimizer),
              options_(std::move(options)),
              base_lrs_(capture_base_lrs(optimizer)),
              step_count_(0) {
            if (options_.T_max == 0) {
                throw std::invalid_argument("CosineAnnealingScheduler requires T_max to be greater than zero.");
            }
            if (options_.warmup_start_factor < 0.0 || options_.warmup_start_factor > 1.0) {
                throw std::invalid_argument("CosineAnnealingScheduler warmup_start_factor must be within [0, 1].");
            }

            apply(step_count_);
        }

        void step() override {
            if (step_count_ < std::numeric_limits<std::size_t>::max()) {
                ++step_count_;
            }
            apply(step_count_);
        }

    private:
        void apply(std::size_t step) {
            auto& param_groups = optimizer_.param_groups();
            if (base_lrs_.size() != param_groups.size()) {
                throw std::runtime_error("Optimizer param group count changed after scheduler creation.");
            }

            for (std::size_t index = 0; index < param_groups.size(); ++index) {
                auto& group = param_groups[index];
                const auto lr = compute_lr(base_lrs_[index], step);
                group.options().set_lr(lr);
            }
        }

        [[nodiscard]] double compute_lr(double base_lr, std::size_t step) const {
            if (options_.warmup_steps > 0 && step < options_.warmup_steps) {
                const double progress = static_cast<double>(step) / static_cast<double>(options_.warmup_steps);
                const double factor = options_.warmup_start_factor + (1.0 - options_.warmup_start_factor) * progress;
                return base_lr * factor;
            }

            const std::size_t effective_step = step > options_.warmup_steps ? step - options_.warmup_steps : 0;
            const std::size_t effective_T_max = std::max<std::size_t>(1, options_.T_max);
            const double clamped_step = static_cast<double>(std::min<std::size_t>(effective_step, effective_T_max));
            constexpr double kPi = 3.14159265358979323846;
            const double cosine = std::cos(kPi * clamped_step / static_cast<double>(effective_T_max));
            return options_.eta_min + (base_lr - options_.eta_min) * (1.0 + cosine) * 0.5;
        }

        static std::vector<double> capture_base_lrs(torch::optim::Optimizer& optimizer) {
            std::vector<double> base_lrs;
            base_lrs.reserve(optimizer.param_groups().size());
            for (auto& group : optimizer.param_groups()) {
                base_lrs.push_back(group.options().get_lr());
            }
            return base_lrs;
        }

        torch::optim::Optimizer& optimizer_;
        CosineAnnealingOptions options_{};
        std::vector<double> base_lrs_{};
        std::size_t step_count_{};
    };
}

#endif //Nott_LRSCHEDULER_COSINEANNEALING_HPP