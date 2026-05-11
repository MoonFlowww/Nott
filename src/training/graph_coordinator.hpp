#ifndef Nott_GRAPH_COORDINATOR_HPP
#define Nott_GRAPH_COORDINATOR_HPP
/**
 * @file graph_coordinator.hpp
 * @brief Per-training-run CUDA graph capture/replay state machine.
 *
 * Lives at run_epochs() scope.  One instance per training run.
 * Tracks whether the graph has been captured for the current batch shape
 * and coordinates the Capture -> Pending -> Ready -> Replay transitions.
 */

#include <cstdint>
#include <vector>
#include <torch/torch.h>

namespace Nott { enum class GraphMode; }

namespace Nott::Training {

struct GraphModeCoordinator {
    enum class Status : uint8_t { NeverCaptured, Pending, Ready };

    GraphMode requested;
    Status    status = Status::NeverCaptured;

private:
    /// Lightweight batch-shape descriptor, avoids pulling in Model::BatchSignature.
    struct TensorDesc {
        torch::Device              device{torch::kCPU};
        torch::ScalarType          dtype{torch::kFloat32};
        std::vector<int64_t>       shape{};

        [[nodiscard]] bool matches(const torch::Tensor& t) const noexcept {
            if (t.device() != device || t.scalar_type() != dtype) return false;
            const auto sizes = t.sizes();
            if (static_cast<std::size_t>(sizes.size()) != shape.size()) return false;
            for (std::size_t i = 0; i < shape.size(); ++i)
                if (sizes[static_cast<int64_t>(i)] != shape[i]) return false;
            return true;
        }
    };

    TensorDesc inputs_sig_{};
    TensorDesc targets_sig_{};
    bool       has_sig_{false};

    static TensorDesc describe(const torch::Tensor& t) {
        TensorDesc d;
        d.device = t.device();
        d.dtype  = t.scalar_type();
        const auto s = t.sizes();
        d.shape.assign(s.begin(), s.end());
        return d;
    }

    [[nodiscard]] bool sig_matches(
        const torch::Tensor& inputs,
        const torch::Tensor& targets) const noexcept
    {
        return has_sig_ && inputs_sig_.matches(inputs) && targets_sig_.matches(targets);
    }

    void set_sig(const torch::Tensor& inputs, const torch::Tensor& targets) {
        inputs_sig_  = describe(inputs);
        targets_sig_ = describe(targets);
        has_sig_     = true;
    }

public:
    explicit GraphModeCoordinator(GraphMode mode) : requested(mode) {}

    /**
     * @brief Resolve the effective GraphMode for this batch.
     *
     * On shape change, resets the model's graph cache and returns Capture.
     * Otherwise returns Capture (Pending) or Replay (Ready).
     *
     * @tparam ModelT  Requires reset_graph_shape_cache(GraphMode).
     */
    template<typename ModelT>
    [[nodiscard]] GraphMode resolve(
        ModelT&              model,
        const torch::Tensor& inputs,
        const torch::Tensor& targets)
    {
        if (requested != GraphMode::Capture) return requested;

        if (!sig_matches(inputs, targets)) {
            set_sig(inputs, targets);
            status = Status::Pending;
            model.reset_graph_shape_cache(GraphMode::Capture);
            return GraphMode::Capture;
        }
        return (status == Status::Pending) ? GraphMode::Capture : GraphMode::Replay;
    }

    void on_captured() noexcept { status = Status::Ready; }

    template<typename ModelT>
    void on_replay_failed(ModelT& model) {
        status = Status::Pending;
        model.reset_graph_shape_cache(GraphMode::Capture);
    }
};

} // namespace Nott::Training
#endif // Nott_GRAPH_COORDINATOR_HPP
