# Attention Modules

The `Nott::Attention` namespace provides descriptors that plug into transformer
blocks and other components needing multi-head attention. This document covers
the available descriptor, its option set, and the runtime module exposed under
`Nott::Attention::Details`.

## Descriptor: `Nott::Attention::MultiHead(...)`

`MultiHead` returns a descriptor wrapping `MultiHeadOptions`. The descriptor is
consumed by transformer blocks (see `src/block/details/transformers`) and can
also be used to instantiate the low-level module manually.

### `MultiHeadOptions`

- `embed_dim` *(int64, required)*, size of the model/embedding dimension.
  Must be positive and divisible by `num_heads`.
- `num_heads` *(int64, default: `1`)*, number of parallel attention heads.
  Must be positive.
- `dropout` *(double, default: `0.0`)*, dropout probability applied to the
  attention weights and the post-projection output.
- `bias` *(bool, default: `true`)*, include learnable biases in the query, key,
  value, and output projections.
- `add_bias_kv` *(bool, default: `false`)*, reserved for parity with LibTorch's
  API; currently not consumed by the implementation.
- `add_zero_attn` *(bool, default: `false`)*, placeholder for future zero
  attention padding support.
- `batch_first` *(bool, default: `true`)*, expect inputs shaped as
  `{batch, sequence, embed_dim}`. When `false`, inputs should be
  `{sequence, batch, embed_dim}`.
- `variant` *(enum, default: `Variant::Full`)*, choose between full attention
  and a causal (upper-triangularly masked) variant.

## Runtime Module: `Nott::Attention::Details::MultiHeadAttention`

The descriptor is materialised into a LibTorch module through
`Nott::Attention::Details::MultiHeadAttention`. The module validates the option
set (positive dimensions and `embed_dim % num_heads == 0`) before registering
four linear projections (`q_proj`, `k_proj`, `v_proj`, `out_proj`), a
`ScaledDotProductKernel`, and an output dropout layer.

### Forward Contract

Given tensors shaped according to `batch_first`:

1. Query, key, and value are linearly projected to `(batch, heads, sequence,
   head_dim)` and fed to the scaled dot-product kernel.
2. The kernel computes attention weights with `1/sqrt(head_dim)` scaling,
   optional key padding masks, additive attention masks, and the selected
   variant.
3. Dropped-out attention weights are multiplied with the value projections and
   fused back through the output projection and dropout.

### Mask Handling

- `key_padding_mask` is broadcast across heads and targets; masked positions are
  filled with `-inf` before the softmax.
- `attn_mask` accepts 2D `(target, source)`, 3D `(batch, target, source)` or
  `(batch * heads, target, source)`, and 4D `(batch, heads, target, source)`
  shapes. Masks are validated and expanded to match the score tensor before
  being added elementwise.
- Selecting `Variant::Causal` adds a strict upper-triangular mask that prevents
  a position from attending to future tokens.

## Manual Usage Example

```cpp
using namespace Nott;
auto descriptor = Attention::MultiHead({
    .embed_dim = 512,
    .num_heads = 8,
    .dropout = 0.1,
    .bias = true,
    .batch_first = true,
    .variant = Attention::Variant::Causal,
});

Attention::Details::MultiHeadAttentionOptions impl_options{};
impl_options.embed_dim = descriptor.options.embed_dim;
impl_options.num_heads = descriptor.options.num_heads;
impl_options.dropout = descriptor.options.dropout;
impl_options.bias = descriptor.options.bias;
impl_options.batch_first = descriptor.options.batch_first;
impl_options.variant = descriptor.options.variant;

auto attention = Attention::Details::MultiHeadAttention(impl_options);
auto output = attention->forward(query, key, value, attn_mask, key_padding_mask);
```

The transformer blocks in `Nott::Block::Transformer*` perform this mapping for
you, but the descriptor and implementation are exposed for custom wiring. [Docs/Block & Transformer](../block/README.md)