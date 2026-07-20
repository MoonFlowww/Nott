#ifndef Nott_LONGFORMER_XL_HPP
#define Nott_LONGFORMER_XL_HPP
/// "Longformer: The Long-Document Transformer", Beltagy (arXiv:2004.05150).
/// and "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context" https://arxiv.org/pdf/1901.02860
/// Sliding-window self-attention with optional segment-level recurrence for long-context sequence modeling.

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

namespace Nott::Block::Details::Transformer::LongformerXL {
    using ::Nott::Block::Details::Transformer::Classic::Detail::normalise_to_sequence;
    using ::Nott::Block::Details::Transformer::Classic::Detail::restore_from_sequence;

    struct AttentionOptions {
        std::int64_t embed_dim{768};
        std::int64_t num_heads{12};
        double dropout{0.0};
        bool bias{true};
        bool batch_first{true};
    };

    struct FeedForwardOptions {
        std::int64_t embed_dim{768};
        double mlp_ratio{4.0};
        ::Nott::Activation::Descriptor activation{::Nott::Activation::GeLU};
        bool bias{true};
        double dropout{0.0};
    };

    struct LayerNormOptions {
        double eps{1e-5};
        bool elementwise_affine{true};
    };

    struct EncoderLayerDescriptor {
        AttentionOptions attention{};
        FeedForwardOptions feed_forward{};
    };

    struct EncoderOptions {
        std::size_t layers{12};
        std::int64_t embed_dim{768};
        AttentionOptions attention{};
        FeedForwardOptions feed_forward{};
        LayerNormOptions layer_norm{};
        std::int64_t window_size{256};
        std::size_t global_tokens{0};
        bool causal{false};
        bool use_memory{false};
        std::size_t memory_size{0};
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
            throw std::invalid_argument("Longformer/Transformer-XL encoder requires a positive embedding dimension.");
        }
        if (options.layers == 0) {
            throw std::invalid_argument("Longformer/Transformer-XL encoder requires at least one layer.");
        }
        if (options.window_size <= 0) {
            throw std::invalid_argument("Longformer/Transformer-XL window size must be positive.");
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
            descriptor.layers.emplace_back(std::move(layer));
        }

        return descriptor;
    }

    namespace Detail {
        class LongformerEncoderLayerImpl : public torch::nn::Module {
        public:
            LongformerEncoderLayerImpl(EncoderLayerDescriptor descriptor, const EncoderOptions& options)
                : options_(options), descriptor_(std::move(descriptor))
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
                                  const torch::Tensor& memory,
                                  std::int64_t window_size,
                                  std::size_t global_tokens,
                                  bool causal)
            {
                if (!input.defined()) {
                    return input;
                }
                if (input.dim() != 3) {
                    throw std::invalid_argument("Longformer layer expects 3D input (B, N, C).");
                }

                torch::Tensor context;
                if (memory.defined()) {
                    context = torch::cat({memory, input}, 1);
                } else {
                    context = input;
                }
                const auto memory_length = memory.defined() ? memory.size(1) : 0;
                const auto target_length = input.size(1);
                const auto source_length = context.size(1);

                auto residual = input;
                auto query = options_.pre_norm ? norm1_->forward(input) : input;

                auto device = query.device();
                auto query_positions = torch::arange(target_length, torch::TensorOptions().dtype(torch::kLong).device(device));
                auto source_positions = torch::arange(source_length, torch::TensorOptions().dtype(torch::kLong).device(device)) - memory_length;

                auto query_matrix = query_positions.to(torch::kLong).unsqueeze(1).expand({target_length, source_length});
                auto source_matrix = source_positions.to(torch::kLong).unsqueeze(0).expand({target_length, source_length});
                auto distance = (query_matrix - source_matrix).abs();

                auto mask = torch::full({target_length, source_length},
                                        -std::numeric_limits<float>::infinity(),
                                        torch::TensorOptions().dtype(torch::kFloat32).device(device));
                auto allowed = distance <= window_size;
                if (causal) {
                    allowed = allowed & (source_matrix <= query_matrix);
                }
                mask = mask.masked_fill(allowed, 0.0f);

                if (global_tokens > 0) {
                    auto global_count = static_cast<std::int64_t>(std::min<std::size_t>(global_tokens, static_cast<std::size_t>(target_length)));
                    if (global_count > 0) {
                        mask.index_put_({torch::indexing::Slice(0, global_count), torch::indexing::Slice()}, 0.0f);
                        auto column_end = memory_length + global_count;
                        column_end = std::min(column_end, source_length);
                        mask.index_put_({torch::indexing::Slice(), torch::indexing::Slice(memory_length, column_end)}, 0.0f);
                    }
                }

                // `mask` is a [target_len, source_len] additive attention-bias mask, which
                // belongs in the attn_mask slot (6th param). It was being passed as
                // key_padding_mask (4th param, expects [batch, source_len]), tripping
                // libtorch's internal key_padding_mask_.size(...) shape assertions.
                //
                // torch::nn::MultiheadAttentionImpl (unlike Python's nn.MultiheadAttention)
                // has no batch_first option and always expects [seq, batch, embed]; every
                // tensor around this call (query, context, the mask's own construction via
                // cat(..., dim=1)) is batch-first [batch, seq, embed]. Without transposing
                // in and back out, batch and sequence silently swap roles inside attention
                // instead of crashing outright (this file's own AttentionOptions::batch_first
                // was never actually applied here).
                auto query_seq_first = query.transpose(0, 1);
                auto context_seq_first = context.transpose(0, 1);
                auto attention = attention_->forward(
                    query_seq_first, context_seq_first, context_seq_first,
                    /*key_padding_mask=*/{}, /*need_weights=*/true, /*attn_mask=*/mask);
                auto attn_output = std::get<0>(attention).transpose(0, 1);

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

        TORCH_MODULE(LongformerEncoderLayer);

        class LongformerEncoderImpl : public torch::nn::Module {
        public:
            explicit LongformerEncoderImpl(EncoderDescriptor descriptor)
                : options_(std::move(descriptor.options))
            {
                if (options_.embed_dim <= 0) {
                    throw std::invalid_argument("Longformer/Transformer-XL encoder requires a positive embedding dimension.");
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
                        LongformerEncoderLayer(std::move(descriptor.layers[index]), options_));
                    layers_.push_back(std::move(layer));
                }
            }

            torch::Tensor forward(torch::Tensor input)
            {
                auto [tokens, shape] = normalise_to_sequence(std::move(input), options_.embed_dim);

                torch::Tensor memory = options_.use_memory ? memory_ : torch::Tensor{};
                auto output = tokens;

                for (auto& layer : layers_) {
                    output = layer->forward(std::move(output), memory, options_.window_size, options_.global_tokens, options_.causal);
                    if (options_.use_memory) {
                        memory = memory.defined() ? torch::cat({memory, output}, 1) : output;
                    }
                }

                if (options_.final_layer_norm) {
                    output = final_layer_norm_->forward(output);
                }

                if (options_.use_memory && options_.memory_size > 0) {
                    auto combined = memory.defined() ? memory : output;
                    const auto memory_limit = static_cast<std::int64_t>(std::min<std::size_t>(options_.memory_size, static_cast<std::size_t>(combined.size(1))));
                    memory_ = combined.slice(1, combined.size(1) - memory_limit, combined.size(1)).detach();
                }

                return restore_from_sequence(std::move(output), shape);
            }

        private:
            EncoderOptions options_{};
            std::vector<LongformerEncoderLayer> layers_{};
            torch::nn::LayerNorm final_layer_norm_{nullptr};
            torch::Tensor memory_{};
        };

        TORCH_MODULE(LongformerEncoder);
    }

    using LongformerEncoderLayer = Detail::LongformerEncoderLayer;
    using LongformerEncoder = Detail::LongformerEncoder;

}

#endif //Nott_LONGFORMER_XL_HPP