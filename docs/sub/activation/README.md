# Activation catalogue

The activation module exposes a `Descriptor` wrapper over the `Activation::Type`
enumeration. Each helper constant (for example `Activation::GeLU`) simply
pre-fills that descriptor so it can be threaded through layer builders or
blocks, you
select the strategy by passing one of the `Nott::Activation::*` constants
when building your model.

```cpp
model.add(Nott::Layer::FC({...}, Nott::Activation::ReLU, Nott::Initialization::HeUniform));
```

## Available activations

| Name | Descriptor constant | Definition | Parameters / Notes |
| --- | --- | --- | --- |
| Identity | `Activation::Identity` | Returns the input tensor unchanged. | No parameters. Acts as a pass-through for linear layers. |
| ReLU | `Activation::ReLU` | Applies Torch's `torch::relu`. | No configurable parameters; negative values are clamped to zero. |
| Sigmoid | `Activation::Sigmoid` | Applies the logistic sigmoid element-wise. | No parameters. |
| Tanh | `Activation::Tanh` | Applies the hyperbolic tangent element-wise. | No parameters. |
| Leaky ReLU | `Activation::LeakyReLU` | Delegates to `torch::leaky_relu`. | Uses Torch's default negative slope (0.01) because no custom value is provided. |
| Softmax | `Activation::Softmax` | Runs `torch::softmax` over the last tensor dimension; scalars are returned unchanged. | No exposed parameters; the reduction axis is automatically chosen as the last dimension when rank > 0. |
| SiLU | `Activation::SiLU` | Applies Torch's SiLU (`x * sigma(x)`). | No additional parameters. |
| GeLU | `Activation::GeLU` | Applies the Gaussian Error Linear Unit (`torch::gelu`). | No parameters. |
| GLU | `Activation::GLU` | Splits the tensor in half along the last dimension and applies `torch::glu`; only true scalars bypass the split. | No parameters. |
| SwiGLU | `Activation::SwiGLU` | Chunks the tensor into two parts along the last dimension, applies SiLU to the first half, and multiplies by the second. | No parameters. |
| dSiLU | `Activation::dSiLU` | Computes the derivative of SiLU: `sigma(x) * (1 + x * (1 - sigma(x)))`. | No parameters. |
| PSiLU | `Activation::PSiLU` | Parametric SiLU variant implemented as `x * sigma(betax)` with fixed `beta = 1.5`. | The slope parameter beta is hard-coded to 1.5; there is no runtime override. |
| Mish | `Activation::Mish` | Applies `x * tanh(softplus(x))`. | No parameters. |
| Swish | `Activation::Swish` | Applies `x * sigma(x)` using `torch::sigmoid`. | No parameters. |

Most activations try to return the original tensor when a meaningful
computation cannot be performed (for example, GLU/SwiGLU on scalars), but shape
validation is still recommended when wiring them into layer pipelines.


#### HyperLinks related:
- [Docs/Introduction](../../README.md) (`model.add()`)
- [Docs/Initialization](../initialization/README.md) (`Nott::Initialization::*`)