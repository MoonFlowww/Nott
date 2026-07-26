#ifndef Nott_COMMON_STREAMING_HPP
#define Nott_COMMON_STREAMING_HPP

#include <cstddef>
#include <memory>
#include <optional>

#include <torch/torch.h>
#ifdef TORCH_CUDA_AVAILABLE
#include <torch/cuda.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAStream.h>
#endif

namespace Nott {

    enum class GraphMode {
        Disabled,
        Capture,
        Replay
    };

    struct ForwardOptions {
        std::optional<std::size_t> max_chunk_size{};
        GraphMode graph_mode{GraphMode::Disabled};  // Graph capture/replay disables chunking; pad/drop to maintain static shapes.

        [[nodiscard]] bool buffering_enabled() const noexcept
        {
            return graph_mode == GraphMode::Disabled && max_chunk_size.has_value() && *max_chunk_size > 0;
        }

        [[nodiscard]] bool graph_capture_requested() const noexcept
        {
            return graph_mode == GraphMode::Capture;
        }

        [[nodiscard]] bool graph_replay_requested() const noexcept
        {
            return graph_mode == GraphMode::Replay;
        }
    };

    struct StreamingOptions {
        std::size_t batch_size{0};
        std::size_t buffer_batches{0};
        std::optional<std::size_t> forward_chunk_size{};
    };

    struct DeferredHostTensor {
        torch::Tensor host_tensor{};
#ifdef TORCH_CUDA_AVAILABLE
        mutable std::shared_ptr<at::cuda::CUDAEvent> ready_event{};
        int device_index{-1};
#endif

        DeferredHostTensor() = default;

        static DeferredHostTensor from_tensor(torch::Tensor tensor, bool non_blocking = false)
        {
            DeferredHostTensor deferred{};

            if (!tensor.defined()) {
                return deferred;
            }

            if (!tensor.device().is_cuda()) {
                deferred.host_tensor = std::move(tensor);
                return deferred;
            }

#ifdef TORCH_CUDA_AVAILABLE
            const auto device_index = tensor.device().index();
            auto host_copy = tensor.to(torch::kCPU, tensor.scalar_type(), non_blocking);
            deferred.host_tensor = std::move(host_copy);
            deferred.device_index = device_index;

            if (non_blocking) {
                auto stream = at::cuda::getCurrentCUDAStream(device_index);
                auto event = std::make_shared<at::cuda::CUDAEvent>();
                event->record(stream);
                deferred.ready_event = std::move(event);
            }
            return deferred;
#else
            deferred.host_tensor = tensor.to(torch::kCPU);
            return deferred;
#endif
        }

        [[nodiscard]] bool defined() const noexcept { return host_tensor.defined(); }

        torch::Tensor materialize() const
        {
#ifdef TORCH_CUDA_AVAILABLE
            if (ready_event) {
                if (!ready_event->query()) {
                    ready_event->synchronize();
                }
                ready_event.reset();
            }
#endif
            return host_tensor;
        }
    };

    struct StreamingBatch {
        torch::Tensor inputs{};
        torch::Tensor targets{};
        DeferredHostTensor reference_targets{};
    };

    struct AsyncPinnedTensor {
        torch::Tensor tensor{};
#ifdef TORCH_CUDA_AVAILABLE
        mutable std::shared_ptr<at::cuda::CUDAEvent> ready_event{};
#endif

        [[nodiscard]] bool defined() const noexcept { return tensor.defined(); }

        [[nodiscard]] bool is_ready() const
        {
#ifdef TORCH_CUDA_AVAILABLE
            if (ready_event) {
                if (!ready_event->query()) {
                    return false;
                }
                ready_event.reset();
            }
#endif
            return tensor.defined();
        }

        torch::Tensor materialize() const
        {
#ifdef TORCH_CUDA_AVAILABLE
            if (ready_event) {
                if (!ready_event->query()) {
                    ready_event->synchronize();
                }
                ready_event.reset();
            }
#endif
            return tensor;
        }
    };

    inline AsyncPinnedTensor async_pin_memory(torch::Tensor tensor)
    {
        AsyncPinnedTensor pinned{};

        if (!tensor.defined()) {
            return pinned;
        }

        if (!tensor.device().is_cpu()) {
            pinned.tensor = std::move(tensor);
            return pinned;
        }

        if (tensor.is_pinned()) {
            pinned.tensor = std::move(tensor);
            return pinned;
        }

#ifdef TORCH_CUDA_AVAILABLE
        if (torch::cuda::is_available()) {
            const auto memory_format = tensor.suggest_memory_format();
            auto options = tensor.options().device(torch::kCPU).pinned_memory(true);
            auto host_copy = torch::empty(tensor.sizes(), options, memory_format);
            const bool non_blocking = true;
            auto stream = at::cuda::getCurrentCUDAStream();
            host_copy.copy_(tensor, non_blocking);
            auto event = std::make_shared<at::cuda::CUDAEvent>();
            event->record(stream);
            pinned.tensor = std::move(host_copy);
            pinned.ready_event = std::move(event);
            return pinned;
        }
#endif

        // only reached with no usable GPU: pinning impossible (CPU build) or pointless.
        pinned.tensor = std::move(tensor);
        return pinned;
    }


}

#endif  // Nott_COMMON_STREAMING_HPP