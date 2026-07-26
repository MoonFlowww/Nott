# Parameter Initialization

Nott exposes a small set of ready-made initialization descriptors that can be
attached to any layer or block descriptor. Each descriptor is a thin wrapper
around the corresponding LibTorch initializer and is completely stateless, you
select the strategy by passing one of the `Nott::Initialization::*` constants
when building your model.

```cpp
model.add(Nott::Layer::FC({...}, Nott::Activation::ReLU, Nott::Initialization::HeUniform));
```

## Available descriptors

| Descriptor | Effect | Notes |
|------------|--------|-------|
| `Default` | Leaves the module untouched. The layer keeps LibTorch's built-in defaults. | Useful when the layer already performs its own initialization. |
| `XavierNormal` | Calls `torch::nn::init::xavier_normal_` on the weight tensor and zeros the bias (if present). | Suitable for tanh / sigmoid activations or any layer expecting a symmetric weight distribution. |
| `XavierUniform` | Calls `torch::nn::init::xavier_uniform_` on the weight tensor and zeros the bias (if present). | Same usage as `XavierNormal` but with a uniform distribution. |
| `HeNormal` | Calls `torch::nn::init::kaiming_normal_` with `fan_in` mode and ReLU non-linearity assumptions, then zeros the bias (if present). | Recommended for ReLU-family activations. |
| `HeUniform` | Calls `torch::nn::init::kaiming_uniform_` with `fan_in` mode and ReLU non-linearity assumptions, then zeros the bias (if present). | ReLU-oriented uniform variant. |
| `ZeroBias` | Leaves weights untouched and sets any defined bias tensor to zeros. | Handy when you want predictable bias values without changing the weight initializer. |
| `Dirac` | Applies `torch::nn::init::dirac_` to weight tensors of dimension >= 3 and zeros the bias (if present). | Ideal for convolutional layers that should initially behave like identity mappings. |
| `Lyapunov` | Applies `torch::nn::init::orthogonal_` to the weight tensor and zeros the bias (if present). | Produces orthogonal matrices, often used for recurrent or state-space layers. |

All descriptors are parameterless; if you need a custom initializer you can
always run the desired LibTorch routine directly before the training loop. The
helper automatically guards bias tensors and skips operations that are not
supported by the module (e.g. `Dirac` on 2-D weights quietly falls back to no
change).

#### HyperLinks related:
- [Docs/Introduction](../../README.md) (`model.add()`)
- [Docs/Activations](../activation/README.md) (`Nott::Activation::*`)

