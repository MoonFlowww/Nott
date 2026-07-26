# Training Loop

Nott's `Model::train` orchestrates dataset streaming, optimisation, and telemetry
capture. You can either pass a packed dataset (`std::vector` of `{inputs,
targets}` pairs) or raw tensors. Training requires an optimizer and a loss
descriptor to be set beforehand (see [Docs/Optimizer](../optimizer/README.md) and
[Docs/Loss](../loss/README.md)).

## TrainOptions

`TrainOptions` controls runtime behaviour:

| Field | Purpose |
| --- | --- |
| `epoch` | Number of epochs to run; zero short-circuits the loop. |
| `batch_size` | Mini-batch size. Must be non-zero. |
| `shuffle` | Shuffle dataset between epochs. |
| `buffer_vram` | When > 0, keep the dataset on the host and stream batches to the GPU on a side stream, overlapping the copy with compute. Requires CUDA. See the guidance below for when this actually helps. |
| `monitor` / `stream` | Enable console logging through `Utils::Terminal` when `stream` is non-null. |
| `restore_best_state` | Keep a shadow copy of parameters and restore the epoch with the lowest validation/test loss. |
| `validation` / `test` | Optional `{inputs, targets}` tensors evaluated at the end of each epoch. Validation is used if test is absent. |
| `graph_mode` | `GraphMode::Disabled` (default), `Capture`, or `Replay`. Capture/Replay are unsupported and throw (see below); leave `Disabled`. |
| `enable_amp` | Automatic mixed precision (bf16 autocast) on CUDA. Runs conv/matmul in bf16 on tensor cores while keeping fp32 master weights; no effect on CPU. See guidance below. |
| `memory_format` | Request `torch::MemoryFormat::ChannelsLast`. Applied only when the model has convolutional layers and runs on CUDA with 4D inputs; ignored (no-op) otherwise. |

Validation/test splits are supplied as `std::vector<torch::Tensor>{inputs,
targets}` to preserve ownership and allow Nott to reuse contiguous host buffers.

## Performance and precision features

The GPU fast paths below compile in only when a CUDA toolkit is present (CMake
defines `TORCH_CUDA_AVAILABLE`). On a CPU only build they compile out and training
still runs correctly, just without the extras.

### Mixed precision (`enable_amp`)

Turns on bf16 autocast on CUDA: convolutions and matmuls run in bf16 on the GPU
tensor cores while parameters and the optimizer stay fp32. bf16 keeps the fp32
exponent range, so no loss scaling is involved.

- **Use it** for models dominated by convolutions or matmuls (CNNs, UNets,
  transformers) on an Ampere class GPU or newer (SM 8.0+, such as the RTX 30/40
  series). Expect roughly 1.5x to 2x per step, largest when paired with
  `ChannelsLast` (see below).
- **Skip it** on CPU (no effect), on very small models where the step is not
  compute bound, and when training is sensitive to reduced mantissa precision
  (bf16 has about 3 decimal digits). If a model trains cleanly in fp32 but
  diverges with AMP, stay on fp32.

### Channels last (`memory_format = ChannelsLast`)

Stores 4D activations and conv weights in NHWC so convolutions hit tensor cores.
Applied only when the model has convolutional layers, runs on CUDA, and inputs are
4D; otherwise it is silently ignored.

- **Use it** for convolutional models on CUDA, especially together with AMP.
  That pairing is where the largest speedup shows up (about 2x on the sample UNet
  over fp32 NCHW).
- **Skip it** for models without convolutions (FC, RNN, or transformer only),
  where it does nothing, and on CPU.

### Input prefetch (`buffer_vram > 0`)

Keeps the dataset on the host and copies each batch to the GPU on a side stream so
the transfer overlaps with the previous batch compute.

- **Use it** when the host to device copy is a real fraction of the step: large
  inputs with a comparatively cheap model, or a dataset too big to keep resident
  in VRAM so it must be streamed each epoch.
- **Skip it** when the whole dataset already fits in VRAM and the model is compute
  bound (the common case): the copy is then a tiny fraction of the step and the
  overlap saves nothing. It is always correct, just not always worth enabling.

### CUDA graph capture (`graph_mode`)

**Currently unsupported.** Requesting `Capture` or `Replay` throws. Leave
`graph_mode` at `Disabled`. The enum stays in the API for a future capture safe
implementation, which needs a rework of how the training step threads gradients
through fixed buffers. When it works it only helps launch bound models (many small
kernels), not the compute bound models that are the common case.

## Telemetry and monitoring

`Model::training_telemetry()` exposes:

- `EpochSnapshot`: epoch index, deferred train/test loss scalars, deferred
  step latency, improvement flags, elapsed time, and learning rate snapshots.
- `DatasetLossSnapshot`: detailed metrics for validation/test sweeps when
  requested.

These values remain on the host and lazily materialise GPU tensors, making them
cheap to log or feed into [Docs/Plot](../plot/README.md). When `monitor` is `true`,
progress is streamed to the provided `std::ostream` with non-blocking CUDA event
handling to avoid stalling the training loop.

### Manual training loop ("semi Nott")

Nott keeps the underlying LibTorch modules exposed, so you can orchestrate a
training loop manually when you need custom control flow (curriculum learning,
reinforcement updates, mixed dataloaders, etc.). The model still constructs the
network graph and owns the optimizer/loss descriptors, but you steer the forward
and backward passes yourself:

```cpp
Nott::Model model("ManualMLP");
model.add(Nott::Layer::FC({784, 512, /*bias=*/true}, Nott::Activation::ReLU));
model.add(Nott::Layer::FC({512, 256, /*bias=*/true}, Nott::Activation::ReLU));
model.add(Nott::Layer::FC({256, 10, /*bias=*/true}, Nott::Activation::Identity));

model.set_optimizer(Nott::Optimizer::Adam({.learning_rate = 1e-3}));
const auto cross_entropy = Nott::Loss::CrossEntropy({});
model.set_loss(cross_entropy);

for (auto epoch = 0; epoch < max_epochs; ++epoch) {
    for (auto [inputs, targets] : minibatches) {
        inputs = inputs.to(model.device());
        targets = targets.to(model.device());

        model.zero_grad();                                 // clear stale gradients
        auto logits = model.forward(inputs);               // forward pass through the DAG
        auto loss = Nott::Loss::Details::compute(          // reuse Nott loss helpers
            cross_entropy, logits, targets);
        loss.backward();                                   // populate gradients
        model.step();                                      // step optimizer (+ scheduler if set)
    }
}
```
When your minibatches originate on the host, you can overlap page-locking with
compute by calling `Nott::async_pin_memory` before transferring the tensors to
the device:

```cpp
auto stage_for_device = [&](torch::Tensor tensor) {
    auto pinned = Nott::async_pin_memory(std::move(tensor));
    auto host   = pinned.materialize();
    const bool non_blocking = model.device().is_cuda() && host.is_pinned();
    return host.to(model.device(), host.scalar_type(), non_blocking);
};

auto inputs  = stage_for_device(minibatch_inputs);
auto targets = stage_for_device(minibatch_targets);
```
- **Forward.** `model.forward` executes the compiled graph exactly as
  `Model::train` would; this keeps AMP and calibration hooks available when you
  enable them via `ForwardOptions`.
- **Backward.** Loss tensors expose the standard `.backward()` API. You can mix
  Nott descriptors with custom reductions, attach gradient hooks, or integrate
  reinforcement learning signals before calling backward.
- **Optimizer step.** `model.step()` respects global and local optimizers plus
  schedulers. Use it when you still want Nott to handle learning-rate policies;
  call `model.optimizer().step()` directly only if you intend to bypass
  scheduler bookkeeping.

Prefer the manual path when you need tight integration with bespoke data
pipelines, gradient accumulation strategies, or debugging hooks that do not fit
inside `Model::train`. The abstractions remain the same, you reuse layer builders,
loss factories, optimizers, and telemetry APIs, while keeping full control over
loop structure.


## Advanced hooks

- **Staging observer.** `Model::set_staging_observer` lets you inspect every
  batch transferred to the device (for debugging augmentations or data quality).
- **Memory format.** Before the first epoch, Nott propagates the requested memory
  format to convolutional layers and residual projections so weight tensors match
  the layout of incoming batches.
- **Regularization integration.** Regularisation descriptors registered via
  [Docs/Regularization](../regularization/README.md) are evaluated inside the training
  step; penalties participate in AMP and CUDA graph capture.

---

Combine `TrainOptions` with [Docs/LrScheduler](../lrscheduler/README.md) and
per-module [Docs/Local](../local/README.md) overrides to craft complex optimisation
schemes. After training, persist the state with [Docs/Save & Load](../saveload/README.md).