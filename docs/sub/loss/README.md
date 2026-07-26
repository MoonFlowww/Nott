# Loss Descriptors

Loss descriptors wrap LibTorch criterion implementations so they can be stored
inside model graphs, attached to [Docs/Local](../local/README.md) scopes, and reused
across [Docs/Training](../training/README.md) sessions. Every helper in
`Nott::Loss` returns a stateless descriptor; the runtime picks the correct
implementation at compile time through `std::variant` visitation.

## Shared conventions

- `Reduction` controls how element-wise losses collapse to scalars. Supported
  values are `Mean`, `Sum`, and `None`.
- Many descriptors expose `use_weight` or `use_pos_weight` toggles. When set to
  `true`, you must provide the corresponding tensor to `Model::set_loss` or
  `Model::compute_loss`.
- All implementations validate tensor shapes and throw informative errors when
  sample counts mismatch, which helps diagnose data/target plumbing issues.

## Catalogue

| Helper | Key options | Notes |
| --- | --- | --- |
| `Loss::MSE` | `reduction`, `use_weight` | Falls back to manual reduction when weights are provided. |
| `Loss::MAE` | `reduction`, `use_weight` | Absolute error counterpart to MSE. |
| `Loss::SmoothL1` | `reduction`, `beta`, `use_weight` | Hybrid between L1 and L2 with configurable transition. |
| `Loss::CrossEntropy` | `reduction`, `use_weight`, `label_smoothing` | Supports class weighting and label smoothing with runtime validation. |
| `Loss::BCEWithLogits` | `reduction`, `use_weight`, `use_pos_weight` | Combines sigmoid and binary cross-entropy for numerical stability. |
| `Loss::NegativeLogLikelihood` | `reduction`, `ignore_index` | Works with log-probabilities and optional class masking. |
| `Loss::KLDiv` | `reduction`, `log_target` | Computes KL divergence against targets that can be log-probabilities. |
| `Loss::CosineEmbedding` | `margin` | Measures cosine similarity between paired embeddings with +/-1 labels. |
| `Loss::MarginRanking` | `margin` | Compares ordered pairs of scores. |
| `Loss::Dice` | `reduction`, `smooth`, `exponent`, `clamp_predictions` | Segmentation-friendly Sorensen-Dice overlap with optional per-sample weights. |
| `Loss::Tversky` | `reduction`, `alpha`, `beta`, `smooth` | Generalises Dice/Jaccard with asymmetric penalties for FP vs FN counts. |
| `Loss::LovaszSoftmax` | `reduction`, `per_image`, `ignore_index`, `apply_softmax`, `include_background`, `only_present_classes` | Optimises a convex Lovasz hinge surrogate of the Jaccard index for dense multi-class segmentation. |

`Loss::CrossEntropy` covers the common multi-class classification scenario where logits are compared against integer class indices. Use it when your output layer represents mutually-exclusive classes and you want a numerically stable softmax + negative-log-likelihood pipeline. The options map directly onto LibTorch's `torch::nn::functional::CrossEntropyFuncOptions`: enable `label_smoothing` to soften hard targets, flip `use_weight` to `true` when class imbalance requires weighting, and adjust `reduction` to control aggregation.

```cpp
#include <torch/torch.h>
#include <Nott.h>

Nott::Model model{/* module configuration */};

model.set_loss(Nott::Loss::CrossEntropy({
    .reduction = Nott::Loss::Reduction::Mean,
    .use_weight = true,
    .label_smoothing = 0.05
}));

torch::Tensor class_weights = torch::tensor({1.0f, 1.5f, 0.75f}, torch::TensorOptions{}.dtype(torch::kFloat32));
torch::Tensor logits = /*forward pass logits*/;
torch::Tensor targets = /*class indices*/;

auto loss = model.compute_loss(logits, targets, class_weights);
```

When `use_weight` is `true`, pass a tensor of per-class weights to `Model::compute_loss` (or to training utilities that forward the optional argument) so the descriptor can propagate it to LibTorch. Omitting the tensor while `use_weight` is enabled will trigger the runtime validation baked into the descriptor's `compute` function, ensuring configuration errors are surfaced immediately.


Descriptors can be stored globally via `Model::set_loss()` or attached to
individual layers through `LocalConfig` when different heads require distinct
criteria. During `Model::save`, all descriptors (and their option structs) are
serialised into `architecture.json`, ensuring checkpoints restore identical
losses after a [Docs/Save & Load](../saveload/README.md) round-trip.

---

Combine these losses with [Docs/Regularization](../regularization/README.md) and
[Docs/Optimizer](../optimizer/README.md) descriptors to fully specify the training
objective.