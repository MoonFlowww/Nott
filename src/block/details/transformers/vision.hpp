#ifndef Nott_VISION_HPP
#define Nott_VISION_HPP
/// "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" Dosovitskiy https://arxiv.org/pdf/2010.11929
/// Vision transformer blocks with optional Swin-style shifted window attention for hierarchical visual modeling.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <cmath>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

#include <torch/torch.h>

#include "classic.hpp"

#include "../../../activation/activation.hpp"
#include "../../../activation/apply.hpp"
#include "../../../attention/attention.hpp"
#include "../../../attention/details/head.hpp"

namespace Nott::Block::Details::Transformer::Vision {
    using PositionalEncodingType = ::Nott::Layer::Details::PositionalEncodingType;
    using PositionalEncodingOptions = ::Nott::Layer::Details::PositionalEncodingOptions;

    enum class Variant {
        ViT,
        Swin,
    };

    struct AttentionOptions {
        std::int64_t embed_dim{768};
        std::int64_t num_heads{12};
        double dropout{0.0};
        bool bias{true};
        bool batch_first{true};
        ::Nott::Attention::Variant variant{::Nott::Attention::Variant::Full};
    };

    struct FeedForwardOptions {
        std::int64_t embed_dim{768};
        double mlp_ratio{4.0};
        ::Nott::Activation::Descriptor activation{::Nott::Activation::GeLU};
        bool bias{true};
    };

    struct LayerNormOptions {
        double eps{1e-6};
        bool elementwise_affine{true};
    };

    struct PatchEmbeddingOptions {
        std::int64_t in_channels{3};
        std::int64_t embed_dim{768};
        std::int64_t patch_size{16};
        bool add_class_token{true};
        bool normalize{true};
        double dropout{0.0};
    };

    struct WindowOptions {
        std::int64_t size{7};
        bool shift{false};
    };

    struct EncoderLayerDescriptor {
        AttentionOptions attention{};
        FeedForwardOptions feed_forward{};
        WindowOptions window{};
    };

    struct EncoderOptions {
        std::size_t layers{12};
        std::int64_t embed_dim{768};
        Variant variant{Variant::ViT};
        AttentionOptions attention{};
        FeedForwardOptions feed_forward{};
        LayerNormOptions layer_norm{};
        PatchEmbeddingOptions patch_embedding{};
        WindowOptions window{};
        PositionalEncodingOptions positional_encoding{};
        double residual_dropout{0.0};
        double attention_dropout{0.0};
        double feed_forward_dropout{0.0};
        bool pre_norm{true};
        bool final_layer_norm{true};
    };

    struct EncoderDescriptor {
        EncoderOptions options{};
        std::vector<EncoderLayerDescriptor> layers{};
    };

    [[nodiscard]] inline auto Encoder(const EncoderOptions& options) -> EncoderDescriptor
    {
        if (options.embed_dim <= 0) {
            throw std::invalid_argument("Vision transformer requires a positive embedding dimension.");
        }
        if (options.layers == 0) {
            throw std::invalid_argument("Vision transformer requires at least one layer.");
        }
        if (options.patch_embedding.patch_size <= 0) {
            throw std::invalid_argument("Vision transformer patch size must be positive.");
        }

        EncoderDescriptor descriptor{};
        descriptor.options = options;

        auto attention = options.attention;
        attention.embed_dim = options.embed_dim;

        auto feed_forward = options.feed_forward;
        feed_forward.embed_dim = options.embed_dim;

        descriptor.layers.reserve(options.layers);
        for (std::size_t index = 0; index < options.layers; ++index) {
            EncoderLayerDescriptor layer{};
            layer.attention = attention;
            layer.feed_forward = feed_forward;
            layer.window = options.window;
            if (options.variant == Variant::Swin && layer.window.size <= 0) {
                layer.window.size = 7;
            }
            if (options.variant == Variant::Swin) {
                layer.attention.batch_first = true;
            }
            descriptor.layers.emplace_back(std::move(layer));
        }

        return descriptor;
    }

    namespace Detail {
        using ::Nott::Block::Details::Transformer::Classic::Detail::normalise_to_sequence;
        using ::Nott::Block::Details::Transformer::Classic::Detail::restore_from_sequence;
        using ClassicPositionalEncoding = ::Nott::Block::Details::Transformer::Classic::Detail::PositionalEncoding;

        class PatchEmbedImpl : public torch::nn::Module {
        public:
            explicit PatchEmbedImpl(PatchEmbeddingOptions options)
                : options_(std::move(options))
            {
                if (options_.embed_dim <= 0) {
                    throw std::invalid_argument("Patch embedding requires a positive embedding dimension.");
                }
                if (options_.patch_size <= 0) {
                    throw std::invalid_argument("Patch embedding requires a positive patch size.");
                }

                conv_ = register_module(
                    "projection",
                    torch::nn::Conv2d(torch::nn::Conv2dOptions(options_.in_channels,
                                                               options_.embed_dim,
                                                               options_.patch_size)
                                          .stride(options_.patch_size)));

                if (options_.normalize) {
                    auto norm_options = torch::nn::LayerNormOptions(std::vector<int64_t>{options_.embed_dim})
                                             .eps(1e-6)
                                             .elementwise_affine(true);
                    norm_ = register_module("norm", torch::nn::LayerNorm(norm_options));
                }

                if (options_.add_class_token) {
                    class_token_ = register_parameter(
                        "class_token", torch::zeros({1, 1, options_.embed_dim}));
                }

                if (options_.dropout > 0.0) {
                    dropout_ = register_module(
                        "dropout",
                        torch::nn::Dropout(torch::nn::DropoutOptions(options_.dropout)));
                }
            }

            torch::Tensor forward(torch::Tensor input)
            {
                if (!input.defined()) {
                    return input;
                }
                if (input.dim() != 4) {
                    throw std::invalid_argument("Vision transformer expects image tensor of shape (B, C, H, W).");
                }

                auto projected = conv_->forward(input);
                last_grid_ = {projected.size(2), projected.size(3)};
                projected = projected.flatten(2).transpose(1, 2);

                if (norm_) {
                    projected = norm_->forward(projected);
                }

                const auto batch = projected.size(0);
                if (class_token_.defined()) {
                    auto cls = class_token_.expand({batch, class_token_.size(1), class_token_.size(2)});
                    projected = torch::cat({cls, projected}, 1);
                }

                if (dropout_) {
                    projected = dropout_->forward(projected);
                }

                return projected;
            }

            [[nodiscard]] auto grid_shape() const noexcept -> std::pair<std::int64_t, std::int64_t>
            {
                return last_grid_;
            }

            [[nodiscard]] auto has_class_token() const noexcept -> bool
            {
                return class_token_.defined();
            }

        private:
            PatchEmbeddingOptions options_{};
            torch::nn::Conv2d conv_{nullptr};
            torch::nn::LayerNorm norm_{nullptr};
            torch::nn::Dropout dropout_{nullptr};
            torch::Tensor class_token_{};
            std::pair<std::int64_t, std::int64_t> last_grid_{0, 0};
        };

        TORCH_MODULE(PatchEmbed);

        inline torch::Tensor window_partition(torch::Tensor tokens, std::int64_t window_size)
        {
            const auto batch = tokens.size(0);
            const auto height = tokens.size(1);
            const auto width = tokens.size(2);
            const auto embed = tokens.size(3);

            if (height % window_size != 0 || width % window_size != 0) {
                throw std::invalid_argument("Window attention requires height and width divisible by window size.");
            }

            tokens = tokens.view({batch,
                                   height / window_size,
                                   window_size,
                                   width / window_size,
                                   window_size,
                                   embed});
            tokens = tokens.permute({0, 1, 3, 2, 4, 5}).contiguous();
            tokens = tokens.view({-1, window_size * window_size, embed});
            return tokens;
        }

        inline torch::Tensor window_reverse(torch::Tensor windows,
                                            std::int64_t window_size,
                                            std::int64_t height,
                                            std::int64_t width)
        {
            const auto batch = windows.size(0) / ((height * width) / (window_size * window_size));
            const auto embed = windows.size(2);

            windows = windows.view({batch,
                                     height / window_size,
                                     width / window_size,
                                     window_size,
                                     window_size,
                                     embed});
            windows = windows.permute({0, 1, 3, 2, 4, 5}).contiguous();
            windows = windows.view({batch, height, width, embed});
            return windows;
        }

        class VisionEncoderLayerImpl : public torch::nn::Module {
        public:
            VisionEncoderLayerImpl(EncoderLayerDescriptor descriptor,
                                   const EncoderOptions& options,
                                   bool has_class_token)
                : options_(options),
                  descriptor_(std::move(descriptor)),
                  has_class_token_(has_class_token)
            {
                auto norm_options = torch::nn::LayerNormOptions(std::vector<int64_t>{options_.embed_dim})
                                         .eps(options_.layer_norm.eps)
                                         .elementwise_affine(options_.layer_norm.elementwise_affine);
                norm1_ = register_module("norm1", torch::nn::LayerNorm(norm_options));
                norm2_ = register_module("norm2", torch::nn::LayerNorm(norm_options));

                auto attn_options = torch::nn::MultiheadAttentionOptions(options_.embed_dim,
                                                                         descriptor_.attention.num_heads);
                attn_options.dropout(descriptor_.attention.dropout);
                attn_options.bias(descriptor_.attention.bias);
                attention_ = register_module("attention", torch::nn::MultiheadAttention(attn_options));

                hidden_dim_ = static_cast<std::int64_t>(std::llround(
                    descriptor_.feed_forward.mlp_ratio * static_cast<double>(options_.embed_dim)));
                if (hidden_dim_ <= 0) {
                    hidden_dim_ = options_.embed_dim;
                }

                auto linear1_options = torch::nn::LinearOptions(options_.embed_dim, hidden_dim_)
                                            .bias(descriptor_.feed_forward.bias);
                auto linear2_options = torch::nn::LinearOptions(hidden_dim_, options_.embed_dim)
                                            .bias(descriptor_.feed_forward.bias);

                linear1_ = register_module("linear1", torch::nn::Linear(linear1_options));
                linear2_ = register_module("linear2", torch::nn::Linear(linear2_options));

                if (options_.attention_dropout > 0.0) {
                    attention_dropout_ = register_module(
                        "attention_dropout",
                        torch::nn::Dropout(torch::nn::DropoutOptions(options_.attention_dropout)));
                }
                if (options_.residual_dropout > 0.0) {
                    residual_dropout_ = register_module(
                        "residual_dropout",
                        torch::nn::Dropout(torch::nn::DropoutOptions(options_.residual_dropout)));
                }
                if (options_.feed_forward_dropout > 0.0) {
                    feed_forward_dropout_ = register_module(
                        "feed_forward_dropout",
                        torch::nn::Dropout(torch::nn::DropoutOptions(options_.feed_forward_dropout)));
                }
            }

            torch::Tensor forward(torch::Tensor input,
                                  std::int64_t height,
                                  std::int64_t width,
                                  const torch::Tensor& attn_mask = {},
                                  const torch::Tensor& key_padding_mask = {})
            {
                if (!input.defined()) {
                    return input;
                }
                if (input.dim() != 3) {
                    throw std::invalid_argument("Vision transformer layer expects 3D input (B, N, C).");
                }

                auto residual = input;
                auto attn_input = options_.pre_norm ? norm1_->forward(input) : input;

                torch::Tensor cls_token{};
                torch::Tensor tokens = attn_input;
                if (options_.variant == Variant::Swin && has_class_token_) {
                    cls_token = tokens.narrow(1, 0, 1);
                    tokens = tokens.narrow(1, 1, tokens.size(1) - 1);
                }

                torch::Tensor attn_output;
                if (options_.variant == Variant::ViT) {
                    c10::optional<torch::Tensor> attention_mask{};
                    if (attn_mask.defined()) {
                        attention_mask = attn_mask;
                    }
                    c10::optional<torch::Tensor> padding_mask{};
                    if (key_padding_mask.defined()) {
                        padding_mask = key_padding_mask;
                    }
                    // torch::nn::MultiheadAttentionImpl has no batch_first option and
                    // always expects [seq, batch, embed]; attn_input here is batch-first
                    // [batch, seq, embed]. Without transposing in and back out, batch and
                    // sequence silently swap roles inside attention instead of crashing.
                    auto attn_input_seq_first = attn_input.transpose(0, 1);
                    auto attention = attention_->forward(
                        attn_input_seq_first,
                        attn_input_seq_first,
                        attn_input_seq_first,
                        padding_mask.value_or(torch::Tensor{}), // optional
                        true,
                        attention_mask.value_or(torch::Tensor{})); // optional
                        attn_output = std::get<0>(attention).transpose(0, 1);

                } else {
                    if (height <= 0 || width <= 0) {
                        throw std::invalid_argument("Swin transformer layer requires spatial height and width.");
                    }
                    auto tokens_2d = tokens.view({tokens.size(0), height, width, options_.embed_dim});
                    if (descriptor_.window.shift) {
                        const auto shift = descriptor_.window.size / 2;
                        tokens_2d = tokens_2d.roll({-shift, -shift}, {1, 2});
                    }
                    auto windows = window_partition(tokens_2d, descriptor_.window.size);
                    // Same batch-first vs. seq-first mismatch as the ViT branch above.
                    auto windows_seq_first = windows.transpose(0, 1);
                    auto attention = attention_->forward(windows_seq_first, windows_seq_first, windows_seq_first);
                    auto windows_out = std::get<0>(attention).transpose(0, 1);
                    auto merged = window_reverse(windows_out, descriptor_.window.size, height, width);
                    if (descriptor_.window.shift) {
                        const auto shift = descriptor_.window.size / 2;
                        merged = merged.roll({shift, shift}, {1, 2});
                    }
                    const auto batch = tokens.size(0);
                    auto flattened = merged.view({batch, height * width, options_.embed_dim});
                    if (has_class_token_) {
                        attn_output = torch::cat({cls_token, flattened}, 1);
                    } else {
                        attn_output = flattened;
                    }
                }

                if (attention_dropout_) {
                    attn_output = attention_dropout_->forward(attn_output);
                }
                if (residual_dropout_) {
                    attn_output = residual_dropout_->forward(attn_output);
                }

                auto output = residual + attn_output;
                if (!options_.pre_norm) {
                    output = norm1_->forward(output);
                }

                residual = output;
                auto ff_input = options_.pre_norm ? norm2_->forward(output) : output;
                auto hidden = linear1_->forward(ff_input);
                hidden = ::Nott::Activation::Details::apply(descriptor_.feed_forward.activation.type, std::move(hidden));
                if (feed_forward_dropout_) {
                    hidden = feed_forward_dropout_->forward(hidden);
                }
                hidden = linear2_->forward(hidden);
                if (residual_dropout_) {
                    hidden = residual_dropout_->forward(hidden);
                }

                output = residual + hidden;
                if (!options_.pre_norm) {
                    output = norm2_->forward(output);
                }

                return output;
            }

        private:
            EncoderOptions options_{};
            EncoderLayerDescriptor descriptor_{};
            bool has_class_token_{false};
            torch::nn::LayerNorm norm1_{nullptr};
            torch::nn::LayerNorm norm2_{nullptr};
            torch::nn::MultiheadAttention attention_{nullptr};
            torch::nn::Linear linear1_{nullptr};
            torch::nn::Linear linear2_{nullptr};
            torch::nn::Dropout attention_dropout_{nullptr};
            torch::nn::Dropout residual_dropout_{nullptr};
            torch::nn::Dropout feed_forward_dropout_{nullptr};
            std::int64_t hidden_dim_{0};
        };

        TORCH_MODULE(VisionEncoderLayer);

        class VisionEncoderImpl : public torch::nn::Module {
        public:
            explicit VisionEncoderImpl(EncoderDescriptor descriptor)
                : options_(std::move(descriptor.options))
            {
                if (options_.embed_dim <= 0) {
                    throw std::invalid_argument("Vision transformer requires a positive embedding dimension.");
                }

                patch_embed_ = register_module("patch_embed", PatchEmbed(options_.patch_embedding));

                if (options_.positional_encoding.type != PositionalEncodingType::None ||
                    options_.positional_encoding.dropout > 0.0) {
                    positional_encoding_ = register_module(
                        "positional_encoding",
                        ClassicPositionalEncoding(options_.embed_dim, options_.positional_encoding));
                }

                if (options_.final_layer_norm) {
                    auto norm_options = torch::nn::LayerNormOptions(std::vector<int64_t>{options_.embed_dim})
                                             .eps(options_.layer_norm.eps)
                                             .elementwise_affine(options_.layer_norm.elementwise_affine);
                    final_layer_norm_ = register_module("final_layer_norm", torch::nn::LayerNorm(norm_options));
                }

                layers_.reserve(descriptor.layers.size());
                for (std::size_t index = 0; index < descriptor.layers.size(); ++index) {
                    auto layer = register_module(
                        "layer_" + std::to_string(index),
                        VisionEncoderLayer(std::move(descriptor.layers[index]), options_, patch_embed_->has_class_token()));
                    layers_.push_back(std::move(layer));
                }
            }

            torch::Tensor forward(torch::Tensor input)
            {
                auto tokens = patch_embed_->forward(std::move(input));
                const auto [height, width] = patch_embed_->grid_shape();

                auto output = tokens;
                if (positional_encoding_) {
                    output = positional_encoding_->forward(output);
                }

                for (auto& layer : layers_) {
                    output = layer->forward(std::move(output), height, width);
                }

                if (final_layer_norm_) {
                    output = final_layer_norm_->forward(output);
                }

                return output;
            }

        private:
            EncoderOptions options_{};
            PatchEmbed patch_embed_{nullptr};
            std::vector<VisionEncoderLayer> layers_{};
            ClassicPositionalEncoding positional_encoding_{nullptr};
            torch::nn::LayerNorm final_layer_norm_{nullptr};
        };

        TORCH_MODULE(VisionEncoder);
    }

    using PatchEmbed = Detail::PatchEmbed;
    using VisionEncoderLayer = Detail::VisionEncoderLayer;
    using VisionEncoder = Detail::VisionEncoder;

}

#endif // Nott_VISION_HPP