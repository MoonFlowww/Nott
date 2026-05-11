#ifndef Nott_TRAINING_POLICY_HPP
#define Nott_TRAINING_POLICY_HPP
/** @file training_policy.hpp
 *  @brief Runtime config collapsing 4-way template combinatorics.
 *
 *  Does NOT include any Nott headers to avoid namespace nesting issues
 *  when included outside namespace Nott.  GraphMode is forward-declared;
 *  the definition comes from common/streaming.hpp (included by core.hpp).
 */

#include <cstddef>
#include <ostream>
#include <torch/torch.h>

namespace Nott { enum class GraphMode; struct TrainOptions; }

namespace Nott::Training {

struct TrainingPolicy {
    std::size_t epochs{10};
    std::size_t batch_size{32};
    bool         shuffle{true};
    std::size_t buffer_vram{0};
    bool         prefetch_available{false};
    GraphMode    graph_mode{GraphMode::Disabled};  // fwd-declared, default works
    bool         amp_enabled{false};
    bool         is_cuda{false};
    torch::MemoryFormat memory_format{torch::MemoryFormat::Contiguous};
    bool         channels_last_applicable{false};
    bool         regularization_active{false};
    bool         monitor{false};
    bool         restore_best_state{false};
    std::ostream* stream{nullptr};
    bool         fold{false};

    [[nodiscard]] bool use_buffer()   const noexcept { return buffer_vram > 0 && is_cuda; }
    [[nodiscard]] bool use_prefetch() const noexcept { return prefetch_available && is_cuda; }
    [[nodiscard]] bool graph_enabled() const noexcept { return graph_mode != GraphMode::Disabled; }
};

/**
 * @brief Build a TrainingPolicy from a TrainOptions-compatible struct.
 *
 * Accepts any @p Opts that exposes the same public fields as TrainOptions.
 * @p eff_graph_mode must already be resolved (Disabled if not supported by device).
 * @p channels_last_applicable must already account for device, conv layers, and input dims.
 */
template<typename Opts>
inline TrainingPolicy make_training_policy(
    const Opts&        opts,
    bool               is_cuda,
    bool               /*has_conv*/,        // used by caller to compute channels_last_applicable
    bool               has_reg,
    bool               prefetch_possible,
    bool               channels_last_applicable,
    GraphMode          eff_graph_mode)
{
    TrainingPolicy p;
    p.epochs                = opts.epoch;
    p.batch_size            = opts.batch_size;
    p.shuffle               = opts.shuffle;
    p.buffer_vram           = opts.buffer_vram;
    p.graph_mode            = eff_graph_mode;
    p.amp_enabled           = opts.enable_amp && is_cuda;
    p.memory_format         = channels_last_applicable
                              ? torch::MemoryFormat::ChannelsLast
                              : torch::MemoryFormat::Contiguous;
    p.regularization_active = has_reg;
    p.is_cuda               = is_cuda;
    p.prefetch_available    = prefetch_possible;
    p.monitor               = opts.monitor && opts.stream != nullptr;
    p.restore_best_state    = opts.restore_best_state;
    p.stream                = opts.stream;
    p.fold                  = opts.fold;
    return p;
}

} // namespace Nott::Training
#endif
