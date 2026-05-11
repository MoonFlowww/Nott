#ifndef Nott_BERT_HPP
#define Nott_BERT_HPP
/// "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" https://arxiv.org/pdf/1810.04805

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
#include "../../../layer/details/positional_encoding.hpp"

namespace Nott::Block::Details::Transformer::Bert {
    using PositionalEncodingType = ::Nott::Layer::Details::PositionalEncodingType;
    using PositionalEncodingOptions = ::Nott::Layer::Details::PositionalEncodingOptions;

    struct AttentionOptions {
        std::int64_t embed_dim{768};
        std::int64_t num_heads{12};
        double dropout{0.1};
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
        double eps{1e-12};
        bool elementwise_affine{true};
    };

    struct EmbeddingOptions {
        std::int64_t vocab_size{30522};
        std::int64_t type_vocab_size{2};
        std::int64_t max_position_embeddings{512};
        double dropout{0.1};
        bool use_token_type{true};
        bool use_position_embeddings{true};
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
        EmbeddingOptions embedding{};
        double residual_dropout{0.1};
        double attention_dropout{0.1};
        double feed_forward_dropout{0.1};
        bool pre_norm{false};
        bool final_layer_norm{true};
    };

    struct EncoderDescriptor {
        EncoderOptions options{};
        std::vector<EncoderLayerDescriptor> layers{};
    };

    [[nodiscard]] inline auto Encoder(const EncoderOptions& options) -> EncoderDescriptor
    {
        if (options.embed_dim <= 0) {
            throw std::invalid_argument("BERT encoder requires a positive embedding dimension.");
        }
        if (options.layers == 0) {
            throw std::invalid_argument("BERT encoder requires at least one layer.");
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
        using ::Nott::Block::Details::Transformer::Classic::Detail::normalise_to_sequence;
        using ::Nott::Block::Details::Transformer::Classic::Detail::restore_from_sequence;

        class BertEmbeddingImpl : public torch::nn::Module {
        public:
            BertEmbeddingImpl(std::int64_t embed_dim, EmbeddingOptions options)
                : embed_dim_(embed_dim), options_(std::move(options))
            {
                if (embed_dim_ <= 0) {
                    throw std::invalid_argument("BERT embeddings require a positive embedding dimension.");
                }

                if (options_.vocab_size > 0) {
                    word_embeddings_ = register_module(
                        "word_embeddings",
                        torch::nn::Embedding(torch::nn::EmbeddingOptions(options_.vocab_size, embed_dim_)));
                }

                if (options_.use_token_type && options_.type_vocab_size > 0) {
                    token_type_embeddings_ = register_module(
                        "token_type_embeddings",
                        torch::nn::Embedding(torch::nn::EmbeddingOptions(options_.type_vocab_size, embed_dim_)));
                }

                if (options_.use_position_embeddings && options_.max_position_embeddings > 0) {
                    position_embeddings_ = register_module(
                        "position_embeddings",
                        torch::nn::Embedding(torch::nn::EmbeddingOptions(options_.max_position_embeddings, embed_dim_)));
                }

                auto norm_options = torch::nn::LayerNormOptions(std::vector<int64_t>{embed_dim_})
                                         .eps(1e-12)
                                         .elementwise_affine(true);
                layer_norm_ = register_module("layer_norm", torch::nn::LayerNorm(norm_options));

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

                torch::Tensor embeddings;
                if (input.scalar_type() == torch::kLong && word_embeddings_) {
                    embeddings = word_embeddings_->forward(input);
                } else {
                    embeddings = input;
                    if (embeddings.size(-1) != embed_dim_) {
                        throw std::invalid_argument(
                            "BERT embeddings expected last dimension " + std::to_string(embed_dim_) +
                            " but received " + std::to_string(embeddings.size(-1)) + ".");
                    }
                }

                const auto batch_size = embeddings.size(0);
                const auto sequence_length = embeddings.size(1);

                if (position_embeddings_) {
                    auto positions = torch::arange(sequence_length,
                                                   torch::TensorOptions().dtype(torch::kLong).device(embeddings.device()));
                    positions = positions.unsqueeze(0).expand({batch_size, sequence_length});
                    embeddings = embeddings + position_embeddings_->forward(positions);
                }

                if (token_type_embeddings_) {
                    auto token_types = torch::zeros({batch_size, sequence_length},
                                                    torch::TensorOptions().dtype(torch::kLong).device(embeddings.device()));
                    embeddings = embeddings + token_type_embeddings_->forward(token_types);
                }

                embeddings = layer_norm_->forward(embeddings);
                if (dropout_) {
                    embeddings = dropout_->forward(embeddings);
                }
                return embeddings;
            }

        private:
            std::int64_t embed_dim_{};
            EmbeddingOptions options_{};
            torch::nn::Embedding word_embeddings_{nullptr};
            torch::nn::Embedding token_type_embeddings_{nullptr};
            torch::nn::Embedding position_embeddings_{nullptr};
            torch::nn::LayerNorm layer_norm_{nullptr};
            torch::nn::Dropout dropout_{nullptr};
        };

        TORCH_MODULE(BertEmbedding);

        class BertEncoderLayerImpl : public torch::nn::Module {
        public:
            BertEncoderLayerImpl(EncoderLayerDescriptor descriptor, const EncoderOptions& options)
                : options_(options), descriptor_(std::move(descriptor))
            {
                if (options_.embed_dim <= 0) {
                    throw std::invalid_argument("BERT encoder layer requires a positive embedding dimension.");
                }

                auto norm_options = torch::nn::LayerNormOptions(std::vector<int64_t>{options_.embed_dim})
                                         .eps(options_.layer_norm.eps)
                                         .elementwise_affine(options_.layer_norm.elementwise_affine);
                norm1_ = register_module("norm1", torch::nn::LayerNorm(norm_options));
                norm2_ = register_module("norm2", torch::nn::LayerNorm(norm_options));

                auto attn_options = torch::nn::MultiheadAttentionOptions(options_.embed_dim,
                                                                         descriptor_.attention.num_heads);
                attn_options.dropout(descriptor_.attention.dropout);
                attn_options.bias(descriptor_.attention.bias);
                attention_ = register_module("self_attention", torch::nn::MultiheadAttention(attn_options));

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
                                  const torch::Tensor& attn_mask = {},
                                  const torch::Tensor& key_padding_mask = {})
            {
                auto [sequence, shape] = normalise_to_sequence(std::move(input), options_.embed_dim);

                auto residual = sequence;
                auto attn_input = options_.pre_norm ? norm1_->forward(sequence) : sequence;

                c10::optional<torch::Tensor> attention_mask{};
                if (attn_mask.defined()) {
                    attention_mask = attn_mask;
                }

                c10::optional<torch::Tensor> padding_mask{};
                if (key_padding_mask.defined()) {
                    padding_mask = key_padding_mask;
                }

                auto attention = attention_->forward(attn_input, attn_input, attn_input, padding_mask.value(), true, attention_mask.value());
                auto attn_output = std::get<0>(attention);

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

                return restore_from_sequence(std::move(output), shape);
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

        TORCH_MODULE(BertEncoderLayer);

        class BertEncoderImpl : public torch::nn::Module {
        public:
            explicit BertEncoderImpl(EncoderDescriptor descriptor)
                : options_(std::move(descriptor.options))
            {
                if (options_.embed_dim <= 0) {
                    throw std::invalid_argument("BERT encoder requires a positive embedding dimension.");
                }

                embedding_ = register_module("embedding", BertEmbedding(options_.embed_dim, options_.embedding));

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
                        BertEncoderLayer(std::move(descriptor.layers[index]), options_));
                    layers_.push_back(std::move(layer));
                }
            }

            torch::Tensor forward(torch::Tensor input,
                                  const torch::Tensor& attn_mask = {},
                                  const torch::Tensor& key_padding_mask = {})
            {
                auto embedded = embedding_->forward(std::move(input));
                auto [normalised, shape] = normalise_to_sequence(std::move(embedded), options_.embed_dim);

                auto output = std::move(normalised);
                for (auto& layer : layers_) {
                    output = layer->forward(std::move(output), attn_mask, key_padding_mask);
                }

                if (final_layer_norm_) {
                    output = final_layer_norm_->forward(output);
                }

                return restore_from_sequence(std::move(output), shape);
            }

        private:
            EncoderOptions options_{};
            BertEmbedding embedding_{nullptr};
            std::vector<BertEncoderLayer> layers_{};
            torch::nn::LayerNorm final_layer_norm_{nullptr};
        };

        TORCH_MODULE(BertEncoder);
    }

    using BertEmbedding = Detail::BertEmbedding;
    using BertEncoderLayer = Detail::BertEncoderLayer;
    using BertEncoder = Detail::BertEncoder;

}

#endif //Nott_BERT_HPP