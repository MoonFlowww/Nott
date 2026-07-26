# Block Containers and Transformer Blueprints

The `Nott::Block` namespace bundles higher-level containers that stitch layers
into reusable macro-architectures. Blocks honour the same activation,
initialization, and [Docs/Local](../local/README.md) overrides accepted by individual
[Docs/Layers](../layer/README.md), letting you mix coarse and fine-grained
configuration inside a model. Every descriptor returned by these helpers can be
fed directly to `Model::add()`.

## Sequential

`Block::Sequential` stores an ordered list of layer descriptors and replays them
in a simple feed-forward fashion. Activations declared on each layer run in
between steps, matching `torch::nn::Sequential` semantics. You can pass either
an `std::initializer_list` or `std::vector` of descriptors when constructing the
block. Attach a `LocalConfig` to scope optimizers, losses, or regularisation to
the entire block. This is a convenient way to fuse patterns such as
`Conv2d -> BatchNorm2d -> Activation` while keeping a single local training scope.

Build a residual-flavoured sequential block and register it on a model with the following pattern:

Sequential Block:
```cpp
Nott::Model model("resnet_tiny");

model.add(Nott::Block::Sequential({
    Nott::Layer::Conv2d(
        Nott::Layer::Conv2dOptions{
            .in_channels = 3,
            .out_channels = 64,
            .kernel_size = {3, 3},
            .stride = {1, 1},
            .padding = {1, 1}
        },
        Nott::Activation::ReLU),
    Nott::Layer::BatchNorm2d(Nott::Layer::BatchNorm2dOptions{.num_features = 64}),
    Nott::Layer::Conv2d(
        Nott::Layer::Conv2dOptions{
            .in_channels = 64,
            .out_channels = 64,
            .kernel_size = {3, 3},
            .stride = {1, 1},
            .padding = {1, 1}
        },
        Nott::Activation::ReLU)
}));
```
Residual Block:
```cpp
model.add(Nott::Block::Residual({
    Nott::Layer::Conv2d(
        Nott::Layer::Conv2dOptions{
            .in_channels = 64,
            .out_channels = 64,
            .kernel_size = {3, 3},
            .stride = {1, 1},
            .padding = {1, 1}
        },
        Nott::Activation::ReLU),
    Nott::Layer::Conv2d(
        Nott::Layer::Conv2dOptions{
            .in_channels = 64,
            .out_channels = 64,
            .kernel_size = {3, 3},
            .stride = {1, 1},
            .padding = {1, 1}
        })
}, /*repeats=*/2, {/*skip*/}));
```

Sequential blocks forward descriptors exactly once in declaration order. To loop a motif multiple times, wrap it in a `Block::Residual` and configure its `repeats` argument as shown above.

## Residual

`Block::Residual` composes skip-connected stacks. Key options include:

- `layers`, ordered descriptors executed inside the residual branch.
- `repeats`, number of times the branch is unrolled; must be positive.
- `skip`, enable projection shortcuts via `use_projection` or supply a custom
  layer descriptor. Projection layers also accept their own activation hooks.
- `output`, configure the post-addition activation (`final_activation`) and
  dropout probability applied after merging the skip connection.

Residual blocks validate tensor shapes and throw descriptive errors when the
skip path does not match the branch output, guiding you towards inserting a
projection. Convolutional weights automatically adopt the tensor memory format
requested by `TrainOptions::memory_format`.

## Transformer Families

Transformer descriptors generate complete encoder/decoder stacks backed by the
attention implementations in `src/block/details/transformers`. Each family
exposes an options struct with sensible defaults and produces the per-layer
components (`Attention`, feed-forward MLPs, dropout, positional encoding) for
you. All transformers operate on sequences in `(batch, tokens, embed)` order.

### Classic

`Transformer::Classic::Encoder/Decoder` describe the canonical Vaswani-style
architecture. Notable knobs:

- `layers`, number of repeated blocks.
- `embed_dim`, hidden width shared by attention heads and feed-forward layers.
- `attention`, multi-head configuration (`num_heads`, `dropout`, `bias`,
  `variant`, `batch_first`).
- `feed_forward`, two-layer MLP with `mlp_ratio`, `bias`, activation, and
  [Initialization](../initialization/README.md) choices.
- `positional_encoding`, sinusoidal/learned positional encoders or a no-op.
- `dropout`, residual dropout applied after each sublayer.

The decoder adds a cross-attention stack and mirrors the encoder feed-forward
logic.

### Mamba

`Transformer::Mamba::Encoder` implements selective state-space layers with
gating-inspired feed-forward stages. Options cover `layers`, `embed_dim`, the
state-space backbone (convolution kernel size, state dimension, activation), and
dropout applied both after SSM updates and dense projections. Use this for long
sequence modelling with linear-time scaling.

### EBT (Energy Based Transformer)

`Transformer::EBT::Encoder/Decoder` add reversible residual links and balanced
branch widths. Options expand upon the classic transformer by letting you tune
`mlp_ratio`, gating activations, and separate dropout values for attention and
feed-forward submodules. The decoder ships with tied encoder-decoder attention
shapes to simplify machine translation setups.

### Transformer++

`Transformer::PlusPlus::Encoder/Decoder` bundle modern improvements (pre-layer
normalisation, SwiGLU activations, depth-scaled dropout). The options struct
lets you pick the normalisation placement, enable rotary positional embeddings,
and adjust the SwiGLU coefficient that rescales the hidden dimension before the
split.

### BERT

`Transformer::Bert::Encoder` reproduces the `bert-base` style stack with
segmentation embeddings, token-type IDs, and layer norm epsilon controls. You
can toggle intermediate dropout, embedding layer norm placement, and the
classification head pooler.

### Vision Transformer

`Transformer::Vision::Encoder` tokenises images through a configurable patch
embedding (`patch_size`, `stride`, `padding`, `conv_bias`) before applying a
classic encoder. Options expose the image resolution, class-token usage, and the
pooling policy (class token vs. mean pooling) for the final representation.

### Perceiver

`Transformer::Perceiver::Encoder` builds cross-attention and latent transformer
modules. Key settings include the latent array size, number of latent updates,
`num_latent_heads`, and whether to share weights across depth. This descriptor
is ideal for multimodal data thanks to its cross-attention front-end.

### Longformer-XL

`Transformer::LongformerXL::Encoder` mixes sliding-window attention with
periodic global tokens. Options control window size, dilation, global token
frequency, and gradient checkpointing for memory savings. Use it for long-form
NLP tasks where quadratic attention would be prohibitive.

---

All blocks emit descriptors compatible with [Docs/Links](../links/README.md), so you
can reroute skip connections or merge transformer branches with `Model::links`
when building complex graphs.
