#ifndef Nott_DEFERRED_SCALAR_HPP
#define Nott_DEFERRED_SCALAR_HPP
/**
 * @file deferred_scalar.hpp
 * @brief Non-blocking GPU→CPU scalar transfer.
 *
 * Call from_tensor() immediately after a GPU op, then materialize() later.
 * On CPU, from_tensor() is a synchronous no-op.
 */

#include <memory>
#include <optional>
#include <torch/torch.h>

#ifdef TORCH_CUDA_AVAILABLE
#include <ATen/cuda/CUDAEvent.h>
#include <ATen/cuda/CUDAStream.h>
#endif

namespace Nott::Training {

struct DeferredScalar {
    torch::Tensor host_tensor{};
#ifdef TORCH_CUDA_AVAILABLE
    mutable std::shared_ptr<at::cuda::CUDAEvent> ready_event{};
    int device_index{-1};
#endif
    mutable std::optional<double> cached_value{};

    DeferredScalar() = default;

    [[nodiscard]] static DeferredScalar from_tensor(
        torch::Tensor        tensor,
        const torch::Device& device)
    {
        DeferredScalar s{};
        if (!tensor.defined()) return s;

        tensor = tensor.detach();
        if (tensor.scalar_type() != torch::kFloat64)
            tensor = tensor.to(torch::kFloat64);

#ifdef TORCH_CUDA_AVAILABLE
        if (device.is_cuda()) {
            const auto idx = device.index();
            auto stream    = at::cuda::getCurrentCUDAStream(idx);
            auto host_copy = tensor.to(torch::kCPU, torch::kFloat64, /*non_blocking=*/true);
            auto event     = std::make_shared<at::cuda::CUDAEvent>();
            event->record(stream);
            s.host_tensor  = std::move(host_copy);
            s.ready_event  = std::move(event);
            s.device_index = idx;
            return s;
        }
#else
        (void) device;
#endif
        s.host_tensor = tensor.to(torch::kCPU, torch::kFloat64);
        return s;
    }

    [[nodiscard]] bool defined() const noexcept { return host_tensor.defined(); }

    [[nodiscard]] bool is_ready() const {
        if (!host_tensor.defined()) return false;
#ifdef TORCH_CUDA_AVAILABLE
        if (ready_event) {
            if (!ready_event->query()) return false;
            ready_event.reset();
        }
#endif
        if (!cached_value) cached_value = host_tensor.item<double>();
        return true;
    }

    double materialize() const {
        if (!host_tensor.defined()) return 0.0;
#ifdef TORCH_CUDA_AVAILABLE
        if (ready_event) {
            if (!ready_event->query()) ready_event->synchronize();
            ready_event.reset();
        }
#endif
        if (!cached_value) cached_value = host_tensor.item<double>();
        return *cached_value;
    }
};

} // namespace Nott::Training
#endif // Nott_DEFERRED_SCALAR_HPP
