# Plotting Utilities

`Nott::Plot` exposes lightweight wrappers around the statistics captured during
[Docs/Training](../training/README.md) and [Docs/Evaluation](../evaluation/README.md).
Plots rely on the Gnuplot bindings shipped in `src/utils/gnuplot.hpp`; ensure
`gnuplot` is available in your environment when rendering to the desktop.

## Training curves

`Plot::Training::Loss` visualises loss trajectories (and optionally learning
rates) stored inside `Model::training_telemetry()`.

```cpp
auto descriptor = Nott::Plot::Training::Loss({
    .learningRate = true,
    .smoothing = true,
    .smoothingWindow = 5,
    .logScale = false,
});
Nott::Plot::Training::Render(model, descriptor, train_losses, val_losses, lrs);
```

- **`learningRate`** toggles overlaying the step-wise learning rate on a
  secondary axis.
- **`smoothing` / `smoothingWindow`** apply a moving average before plotting,
  helping highlight long-term trends.
- **`logScale`** switches the y-axis to log-space for easier inspection of rapid
  loss drops.

You can pass only the tensors you recorded-validation losses and learning rates
are optional.

## Reliability and interpretability

The `Plot::Reliability` namespace bundles detectors and explanation tools for
classification models:

- **DET / ROC / PR / Youdens**, Consume logits (or precomputed probability
  series) and render Detection Error Tradeoff, ROC, Precision-Recall, and Youden
  index curves. Options allow log scaling (`adjustScale`) and custom subplot
  styling.
- **GradCAM / LIME**, Generate visual explanations by projecting feature
  importance back onto the input. Supply tensors, and the helper handles model
  hooks internally.

Each plotting function accepts either tensors (the model is queried on the fly)
or explicit vectors of probabilities/targets, making it easy to visualise
results exported from [Evaluation](../evaluation/README.md) reports.

---

Plots are side-effect free: they do not mutate the model, and they respect the
current device placement. Combine them with [Save & Load](../saveload/README.md)
to attach visual diagnostics to checkpoints.