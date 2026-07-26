# Local Configuration and Custom Routing

Nott allows every layer or block to carry its own training scope (optimizer, loss,
regularization) while also exposing a routing DSL to wire complex data flows
between modules. This document summarises the local configuration knobs and shows
how to remap the forward graph with `Model::links`.

## Local definitions

### `Nott::LocalConfig`
`Nott::LocalConfig` bundles three optional pieces of configuration that can be
attached to a module descriptor:

| Field | Type | Effect |
| --- | --- | --- |
| `optimizer` | `std::optional<Nott::Optimizer::Descriptor>` | Replaces the model-wide optimizer for the owning module. |
| `loss` | `std::optional<Nott::Loss::Descriptor>` | Serialised with the architecture for future use. |
| `regularization` | `std::vector<Nott::Regularization::Descriptor>` | Registers penalties that only touch the owning module's parameters. |

The structure lives in `src/common/local.hpp` and is fully serialised when a
model is saved, so local scopes survive round-trips through
`architecture.json`. [Src/Local: LocalConfig struct](../../../src/common/local.hpp#L12-L16) [Src/Save & Load: Serialization logic](../../../src/common/save_load.hpp#L1945-L1976)

### Attaching a local scope
All layer factories expose the `LocalConfig` as the last parameter, and block
helpers mirror that convention. Passing a default-constructed `LocalConfig`
keeps the global behaviour, while filling any field activates the override on
that specific descriptor.[Src/Layer: Layer factory overloads](../../../src/layer/layer.hpp#L92-L138) [code: Block factory overloads](../../../src/block/block.hpp#L48-L80)

```cpp
Nott::LocalConfig bottleneck_scope{
    .optimizer = Nott::Optimizer::AdamW({.learning_rate = 3e-4f}),
    .regularization = {/*vector field*/
        Nott::Regularization::L2({.coefficient=0.f})
    }
};

model.add(
    Nott::Layer::FC({1024, 256, /*bias*/true},
        Nott::Activation::GeLU,
        Nott::Initialization::HeNormal,
        bottleneck_scope), "bottleneck");
```

The example above builds a layer that retains the global loss and regularizers,
yet it replaces the model-wide optimizer for the `bottleneck` parameters and
adds an extra DropConnect penalty only for that layer. Global optimizers and
regularization declared via `Model::set_*` continue to govern every layer that
does not opt into its own override, so shared defaults remain intact while the
local scope tailors behaviour where needed. For a catalogue of descriptors that
can populate these scopes see [Docs/Optimizer](../optimizer/README.md) and
[Docs/Regularization](../regularization/README.md).

### Layer and block semantics
* **Sequential blocks.** When a `Block::Sequential` descriptor declares a local
  optimizer, the framework either wraps the entire block as a single module so
  the optimizer can own all parameters, or propagates the same `LocalConfig`
  into child layers that do not have their own optimizer override. This prevents
  multiple optimizers from fighting over the same parameters.[code: Sequential block handling](../../../src/core.hpp#L526-L568)
* **Residual blocks.** Residual descriptors simply preserve their `LocalConfig`
  and expose it as a single module in the compiled graph.[code: Residual block handling](../../../src/core.hpp#L571-L586)

### Local optimizers
`Model::set_optimizer` inspects every registered layer. If a layer requests a
local optimizer, its parameters are removed from the global parameter pool and a
separate optimizer instance is built just for that layer. Attempting to attach a
local optimizer to a module without registered parameters raises a
`logic_error`, avoiding silent misconfiguration.[Src/Core: Local optimizer setup](../../../src/core.hpp#L1422-L1465)

Local optimizers participate in `zero_grad()` and `step()` alongside the global
optimizer, so the training loop clears and updates all optimizers in lock-step
without additional user code. [Src/Core: Optimizer stepping integration](../../../src/core.hpp#L2281-L2294)

### Local regularization
When a layer is registered, the engine captures its trainable parameters and
creates regularization bindings for every descriptor stored in its local
configuration. Later on, `compute_regularization_penalty()` accumulates the
contributions from both global and per-layer bindings, keeping CUDA graph safety
checks intact. [Src/Core: Regularization bindings](../../../src/core.hpp#L2976-L3012) [Src/Core: Penalty accumulation](../../../src/core.hpp#L3261-L3271) [Src/Core: Regularization descriptors](../../../src/core.hpp#L1545-L1636)

### Local losses
Loss descriptors recorded inside `LocalConfig` are currently persisted through
save/load, but the runtime still evaluates a single model-level loss via
`Model::compute_loss`. This means per-layer loss overrides are not consumed
during training yet, even though they remain part of the serialised
configuration. [Src/Save & Load: Loss serialization](../../../src/common/save_load.hpp#L1945-L1976) [Src/Core: Loss evaluation](../../../src/core.hpp#L2253-L2262)
