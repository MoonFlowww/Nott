# Learning Rate Schedulers

Schedulers in `Nott::LrScheduler` are thin descriptors layered on top of LibTorch
optimizers. Pair them with [Docs/Optimizer](../optimizer/README.md) descriptors via
`Model::set_optimizer` or `LocalConfig` to control the learning rate schedule per
module.

## Cosine Annealing with Warmup

`LrScheduler::CosineAnnealing` mirrors the widely used cosine decay policy with
an optional linear warmup stage. The options struct exposes:

- `T_max`, Number of scheduler steps covering one cosine period. Must be
  positive.
- `eta_min`, Minimum learning rate reached at the trough. Defaults to `0.0`.
- `warmup_steps`, Number of linear warmup iterations executed before the cosine
  cycle begins.
- `warmup_start_factor`, Initial scale applied to the base learning rate when
  warmup is enabled (0 -> start from zero, 1 -> start from the base rate).

The scheduler validates that optimizer parameter groups remain constant after
construction and updates each group's `lr` in lockstep. Use it alongside the
telemetry exposed by [Docs/Training](../training/README.md) to plot learning-rate
trajectories with [Docs/Plot](../plot/README.md).

When you advance the scheduler once per optimizer update, express `T_max` and
`warmup_steps` in the same units as your training loop (for example,
`steps_per_epoch`) so a single cosine cycle or warmup period covers the intended
number of batches.

```cpp
Nott::Model model("CosineWarmup");

model.set_optimizer(
    Nott::Optimizer::AdamW({
        .learning_rate = 3e-4,
        .beta1 = 0.9,
        .beta2 = 0.999,
        .eps = 1e-8,
        .weight_decay = 1e-2,
        .amsgrad = false,
    }),
    Nott::LrScheduler::CosineAnnealing({
        .T_max = steps_per_epoch * epochs,
        .eta_min = 3e-6,
        .warmup_steps = 5 * steps_per_epoch,
        .warmup_start_factor = 0.1,
    })
);
```

---

Additional schedulers can be added by implementing `Details::Scheduler` and
expanding the `Descriptor` variant, following the same pattern as
`CosineAnnealing`.