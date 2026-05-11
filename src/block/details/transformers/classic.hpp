#ifndef Nott_CLASSIC_HPP
#define Nott_CLASSIC_HPP

/// "Attention Is All You Need", Vaswani et al., NeurIPS 2017 (arXiv:1706.03762).
/// Canonical encoder/decoder transformer with multi-head self-attention, position encodings,
/// residual connections, and feed-forward sublayers serving as the baseline architecture.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>


#include <torch/torch.h>


#include "../../../activation/activation.hpp"
#include "../../../activation/apply.hpp"
#include "../../../attention/builder.hpp"
#include "../../../attention/attention.hpp"
#include "../../../initialization/initialization.hpp"
#include "../../../layer/details/positional_encoding.hpp"
#include "../../../layer/layer.hpp"
#include "../../../layer/registry.hpp"
#include "../blocks/residual.hpp"
#include "../blocks/sequential.hpp"

namespace Nott::Block::Details::Transformer::Classic {
    using PositionalEncodingType = ::Nott::Layer::Details::PositionalEncodingType;
    using PositionalEncodingOptions = ::Nott::Layer::Details::PositionalEncodingOptions;

    struct AttentionOptions {
        std::int64_t embed_dim{512};
        std::int64_t num_heads{8};
        double dropout{0.0};
        bool bias{true};
        bool batch_first{true};
        ::Nott::Attention::Variant variant{::Nott::Attention::Variant::Full};
    };

    struct FeedForwardOptions {
        std::int64_t embed_dim{512};
        double mlp_ratio{4.0};
        ::Nott::Activation::Descriptor activation{::Nott::Activation::GeLU};
        bool bias{true};
        ::Nott::Initialization::Descriptor initialization{::Nott::Initialization::Default};
    };

    struct LayerNormOptions {
        double eps{1e-5};
        bool elementwise_affine{true};
    };



    struct EncoderLayerDescriptor {
        ::Nott::Attention::Descriptor attention;
        ::Nott::Layer::Descriptor attention_dropout;
        std::vector<::Nott::Layer::Descriptor> feed_forward;
        ::Nott::Layer::Descriptor feed_forward_dropout;
    };

    struct EncoderOptions {
        std::size_t layers{1};
        std::int64_t embed_dim{512};
        AttentionOptions attention{};
        FeedForwardOptions feed_forward{};
        LayerNormOptions layer_norm{};
        PositionalEncodingOptions positional_encoding{};
        double dropout{0.0};
    };

    struct EncoderDescriptor {
        EncoderOptions options{};
        std::vector<EncoderLayerDescriptor> layers{};
    };

    [[nodiscard]] inline auto Encoder(const EncoderOptions& options) -> EncoderDescriptor
    {
        EncoderDescriptor descriptor{};
        descriptor.options = options;

        auto attention_options = options.attention;
        attention_options.embed_dim = options.embed_dim;

        auto feed_forward = options.feed_forward;
        feed_forward.embed_dim = options.embed_dim;

        auto hidden_dim = static_cast<std::int64_t>(std::llround(feed_forward.mlp_ratio * static_cast<double>(options.embed_dim)));
        if (hidden_dim <= 0) {
            hidden_dim = options.embed_dim;
        }

        for (std::size_t index = 0; index < options.layers; ++index) {
            EncoderLayerDescriptor layer{};

            ::Nott::Attention::MultiHeadOptions attention_descriptor_options{};
            attention_descriptor_options.embed_dim = attention_options.embed_dim;
            attention_descriptor_options.num_heads = attention_options.num_heads;
            attention_descriptor_options.dropout = attention_options.dropout;
            attention_descriptor_options.bias = attention_options.bias;
            attention_descriptor_options.batch_first = attention_options.batch_first;
            attention_descriptor_options.variant = attention_options.variant;
            layer.attention = ::Nott::Attention::MultiHead(attention_descriptor_options);

            layer.attention_dropout = ::Nott::Layer::HardDropout({attention_options.dropout});

            ::Nott::Layer::FCOptions fc1_options{feed_forward.embed_dim, hidden_dim, feed_forward.bias};
            layer.feed_forward.emplace_back(::Nott::Layer::FC(fc1_options, feed_forward.activation, feed_forward.initialization));

            ::Nott::Layer::FCOptions fc2_options{hidden_dim, feed_forward.embed_dim, feed_forward.bias};
            layer.feed_forward.emplace_back(::Nott::Layer::FC(fc2_options, ::Nott::Activation::Identity, feed_forward.initialization));

            layer.feed_forward_dropout = ::Nott::Layer::HardDropout({options.dropout});

            descriptor.layers.emplace_back(std::move(layer));
        }

        return descriptor;
    }

    struct DecoderLayerDescriptor {
        ::Nott::Attention::Descriptor self_attention;
        ::Nott::Layer::Descriptor self_attention_dropout;
        ::Nott::Attention::Descriptor cross_attention;
        ::Nott::Layer::Descriptor cross_attention_dropout;
        std::vector<::Nott::Layer::Descriptor> feed_forward;
        ::Nott::Layer::Descriptor feed_forward_dropout;
    };

    struct DecoderOptions {
        std::size_t layers{1};
        std::int64_t embed_dim{512};
        AttentionOptions self_attention{};
        AttentionOptions cross_attention{};
        FeedForwardOptions feed_forward{};
        LayerNormOptions layer_norm{};
        PositionalEncodingOptions positional_encoding{};
        double dropout{0.0};
    };

    struct DecoderDescriptor {
        DecoderOptions options{};
        std::vector<DecoderLayerDescriptor> layers{};
    };

    [[nodiscard]] inline auto Decoder(const DecoderOptions& options) -> DecoderDescriptor {
        DecoderDescriptor descriptor{};
        descriptor.options = options;

        auto self_attention = options.self_attention;
        self_attention.embed_dim = options.embed_dim;

        auto cross_attention = options.cross_attention;
        cross_attention.embed_dim = options.embed_dim;

        auto feed_forward = options.feed_forward;
        feed_forward.embed_dim = options.embed_dim;

        auto hidden_dim = static_cast<std::int64_t>(std::llround(feed_forward.mlp_ratio * static_cast<double>(options.embed_dim)));
        if (hidden_dim <= 0) {
            hidden_dim = options.embed_dim;
        }

        for (std::size_t index = 0; index < options.layers; ++index) {
            DecoderLayerDescriptor layer{};

            ::Nott::Attention::MultiHeadOptions self_attention_descriptor_options{};
            self_attention_descriptor_options.embed_dim = self_attention.embed_dim;
            self_attention_descriptor_options.num_heads = self_attention.num_heads;
            self_attention_descriptor_options.dropout = self_attention.dropout;
            self_attention_descriptor_options.bias = self_attention.bias;
            self_attention_descriptor_options.batch_first = self_attention.batch_first;
            self_attention_descriptor_options.variant = self_attention.variant;
            layer.self_attention = ::Nott::Attention::MultiHead(self_attention_descriptor_options);
            layer.self_attention_dropout = ::Nott::Layer::HardDropout({self_attention.dropout});

            ::Nott::Attention::MultiHeadOptions cross_attention_descriptor_options{};
            cross_attention_descriptor_options.embed_dim = cross_attention.embed_dim;
            cross_attention_descriptor_options.num_heads = cross_attention.num_heads;
            cross_attention_descriptor_options.dropout = cross_attention.dropout;
            cross_attention_descriptor_options.bias = cross_attention.bias;
            cross_attention_descriptor_options.batch_first = cross_attention.batch_first;
            cross_attention_descriptor_options.variant = cross_attention.variant;
            layer.cross_attention = ::Nott::Attention::MultiHead(cross_attention_descriptor_options);
            layer.cross_attention_dropout = ::Nott::Layer::HardDropout({cross_attention.dropout});

            ::Nott::Layer::FCOptions fc1_options{feed_forward.embed_dim, hidden_dim, feed_forward.bias};
            layer.feed_forward.emplace_back(::Nott::Layer::FC(fc1_options, feed_forward.activation, feed_forward.initialization));

            ::Nott::Layer::FCOptions fc2_options{hidden_dim, feed_forward.embed_dim, feed_forward.bias};
            layer.feed_forward.emplace_back(::Nott::Layer::FC(fc2_options, ::Nott::Activation::Identity, feed_forward.initialization));

            layer.feed_forward_dropout = ::Nott::Layer::HardDropout({options.dropout});

            descriptor.layers.emplace_back(std::move(layer));
        }

        return descriptor;
    }

    namespace Detail {
        struct SequenceShape {
            bool added_sequence_dim{false};
            bool flattened_spatial{false};
            std::int64_t batch{0};
            std::int64_t embed{0};
            std::int64_t height{0};
            std::int64_t width{0};
        };

        inline auto normalise_to_sequence(torch::Tensor input, std::int64_t expected_embed_dim)
            -> std::pair<torch::Tensor, SequenceShape>
        {
            SequenceShape shape{};
            if (!input.defined()) {
                return {std::move(input), shape};
            }

            auto normalised = std::move(input);
            switch (normalised.dim()) {
            case 0:
            case 1:
                throw std::invalid_argument("Transformer encoder requires inputs with at least two dimensions.");
            case 2:
                shape.added_sequence_dim = true;
                shape.batch = normalised.size(0);
                shape.embed = normalised.size(1);
                normalised = normalised.unsqueeze(1);
                break;
            case 3:
                shape.batch = normalised.size(0);
                shape.embed = normalised.size(2);
                break;
            case 4: {
                shape.flattened_spatial = true;
                shape.batch = normalised.size(0);
                shape.embed = normalised.size(1);
                shape.height = normalised.size(2);
                shape.width = normalised.size(3);
                normalised = normalised.contiguous()
                                 .view({shape.batch, shape.embed, shape.height * shape.width})
                                 .transpose(1, 2);
                break;
            }
            default:
                throw std::invalid_argument("Transformer encoder received input with unsupported rank " +
                                            std::to_string(normalised.dim()) +
                                            ". Expected 2D (batch, embed), 3D (batch, sequence, embed) or 4D "
                                            "(batch, channels, height, width).");
            }

            normalised = normalised.contiguous();

            const auto actual_embed_dim = normalised.size(-1);
            if (expected_embed_dim > 0 && actual_embed_dim != expected_embed_dim) {
                throw std::invalid_argument("Transformer encoder expected embedding dimension " +
                                            std::to_string(expected_embed_dim) + " but received " +
                                            std::to_string(actual_embed_dim) + ".");
            }

            if (shape.batch == 0) {
                shape.batch = normalised.size(0);
            }
            shape.embed = actual_embed_dim;

            return {std::move(normalised), shape};
        }

        inline torch::Tensor restore_from_sequence(torch::Tensor output, const SequenceShape& shape)
        {
            if (!output.defined()) {
                return output;
            }

            auto restored = std::move(output);
            if (shape.flattened_spatial) {
                restored = restored.transpose(1, 2)
                               .contiguous()
                               .view({shape.batch, shape.embed, shape.height, shape.width});
            }
            if (shape.added_sequence_dim) {
                restored = restored.squeeze(1);
            }
            return restored;
        }


        class PositionalEncodingImpl : public torch::nn::Module {
        public:
            PositionalEncodingImpl(std::int64_t embed_dim, PositionalEncodingOptions options)
                : embed_dim_(embed_dim),
                  options_(std::move(options))
            {
                if (embed_dim_ <= 0) {
                    throw std::invalid_argument("Positional encoding requires a positive embedding dimension.");
                }

                dropout_ = register_module(
                    "dropout",
                    torch::nn::Dropout(torch::nn::DropoutOptions(options_.dropout)));

                if (options_.type == PositionalEncodingType::Sinusoidal) {
                    if (options_.max_length == 0) {
                        throw std::invalid_argument("Sinusoidal positional encoding requires a positive max length.");
                    }

                    auto encoding = torch::zeros({static_cast<long>(options_.max_length), embed_dim_},
                                                 torch::TensorOptions().dtype(torch::kFloat32));
                    auto accessor = encoding.accessor<float, 2>();

                    for (std::size_t position = 0; position < options_.max_length; ++position) {
                        for (std::int64_t dimension = 0; dimension < embed_dim_; ++dimension) {
                            const auto div_term = std::pow(10000.0,
                                                           (2.0 * std::floor(static_cast<double>(dimension) / 2.0)) /
                                                               static_cast<double>(embed_dim_));
                            const auto angle = static_cast<double>(position) / div_term;
                            accessor[position][dimension] = (dimension % 2 == 0)
                                                                 ? static_cast<float>(std::sin(angle))
                                                                 : static_cast<float>(std::cos(angle));
                        }
                    }

                    encoding_ = encoding.unsqueeze(0);
                    register_buffer("encoding", encoding_);
                } else if (options_.type == PositionalEncodingType::Learned) {
                    if (options_.max_length == 0) {
                        throw std::invalid_argument("Learned positional encoding requires a positive max length.");
                    }
                    embedding_ = register_module(
                        "embedding",
                        torch::nn::Embedding(torch::nn::EmbeddingOptions(options_.max_length, embed_dim_)));
                }
            }

            torch::Tensor forward(torch::Tensor input) {
                auto [normalised_input, shape] = normalise_to_sequence(std::move(input), embed_dim_);
                auto output = std::move(normalised_input);
                if (options_.type == PositionalEncodingType::Sinusoidal) {
                    if (!output.defined()) {
                        return output;
                    }
                    const auto sequence_length = output.size(1);
                    if (sequence_length > static_cast<long>(options_.max_length)) {
                        throw std::invalid_argument("Input sequence length exceeds configured positional encoding max length.");
                    }
                    auto encoding = encoding_.narrow(1, 0, sequence_length);
                    encoding = encoding.to(output.device(), output.scalar_type());
                    output = output + encoding;
                } else if (options_.type == PositionalEncodingType::Learned) {
                    if (!output.defined()) {
                        return output;
                    }
                    const auto sequence_length = output.size(1);
                    if (sequence_length > static_cast<long>(options_.max_length)) {
                        throw std::invalid_argument("Input sequence length exceeds configured positional encoding max length.");
                    }
                    if (!embedding_) {
                        throw std::logic_error("Learned positional encoding is not initialised.");
                    }
                    auto positions = torch::arange(sequence_length,
                                                   torch::TensorOptions().dtype(torch::kLong).device(output.device()));
                    positions = positions.unsqueeze(0).expand({output.size(0), sequence_length});
                    auto encoding = embedding_->forward(positions);
                    output = output + encoding;
                }

                if (dropout_) {
                    output = dropout_->forward(output);
                }
                return output;
            }

        private:
            std::int64_t embed_dim_{};
            PositionalEncodingOptions options_{};
            torch::nn::Dropout dropout_{nullptr};
            torch::nn::Embedding embedding_{nullptr};
            torch::Tensor encoding_{};
        };

        TORCH_MODULE(PositionalEncoding);

        class TransformerEncoderLayerImpl : public torch::nn::Module {
        public:
            TransformerEncoderLayerImpl(EncoderLayerDescriptor descriptor, const EncoderOptions& options)
                : embed_dim_(options.embed_dim),
                  layer_norm_options_(options.layer_norm)
            {
                if (embed_dim_ <= 0) {
                    throw std::invalid_argument("Transformer encoder layer requires a positive embedding dimension.");
                }

                auto norm_options = torch::nn::LayerNormOptions(std::vector<int64_t>{embed_dim_})
                                         .eps(layer_norm_options_.eps)
                                         .elementwise_affine(layer_norm_options_.elementwise_affine);

                norm1_ = register_module("norm1", torch::nn::LayerNorm(norm_options));
                norm2_ = register_module("norm2", torch::nn::LayerNorm(norm_options));

                attention_ = ::Nott::Attention::Details::register_attention(*this, "self_attention", std::move(descriptor.attention));

                std::size_t module_index = 0;
                auto register_layer = [&](::Nott::Layer::Descriptor layer_descriptor) {
                    return std::visit(
                        [&](const auto& concrete_descriptor) {
                            return ::Nott::Layer::Details::build_registered_layer(*this,
                                                                                  concrete_descriptor,
                                                                                  module_index++);
                        },
                        std::move(layer_descriptor));
                };

                attention_dropout_ = register_layer(std::move(descriptor.attention_dropout));

                feed_forward_layers_.reserve(descriptor.feed_forward.size());
                for (auto& layer : descriptor.feed_forward) {
                    feed_forward_layers_.push_back(register_layer(std::move(layer)));
                }

                feed_forward_dropout_ = register_layer(std::move(descriptor.feed_forward_dropout));
            }

            torch::Tensor forward(torch::Tensor input,
                                  const torch::Tensor& attn_mask = {},
                                  const torch::Tensor& key_padding_mask = {})
            {
                auto [output, shape] = normalise_to_sequence(std::move(input), embed_dim_);

                auto residual = output;
                auto normalised = norm1_->forward(residual);
                auto attention = ::Nott::Attention::Details::forward_attention(attention_, normalised, normalised, normalised, attn_mask, key_padding_mask);
                if (attention_dropout_.forward) {
                    attention = attention_dropout_.forward(std::move(attention));
                    attention = ::Nott::Activation::Details::apply(attention_dropout_.activation, std::move(attention));
                }
                output = residual + attention;

                residual = output;
                auto feed_forward = norm2_->forward(output);
                for (auto& layer : feed_forward_layers_) {
                    if (!layer.forward) {
                        continue;
                    }
                    feed_forward = layer.forward(std::move(feed_forward));
                    feed_forward = ::Nott::Activation::Details::apply(layer.activation, std::move(feed_forward));
                }
                if (feed_forward_dropout_.forward) {
                    feed_forward = feed_forward_dropout_.forward(std::move(feed_forward));
                    feed_forward = ::Nott::Activation::Details::apply(feed_forward_dropout_.activation,
                                                                      std::move(feed_forward));
                }

                output = residual + feed_forward;
                return restore_from_sequence(std::move(output), shape);
            }

        private:
            std::int64_t embed_dim_{};
            LayerNormOptions layer_norm_options_{};
            ::Nott::Attention::Details::AttentionModule attention_{};
            torch::nn::LayerNorm norm1_{nullptr};
            torch::nn::LayerNorm norm2_{nullptr};
            ::Nott::Layer::Details::RegisteredLayer attention_dropout_{};
            std::vector<::Nott::Layer::Details::RegisteredLayer> feed_forward_layers_{};
            ::Nott::Layer::Details::RegisteredLayer feed_forward_dropout_{};
        };

        TORCH_MODULE(TransformerEncoderLayer);

        class TransformerEncoderImpl : public torch::nn::Module {
        public:
            explicit TransformerEncoderImpl(EncoderDescriptor descriptor)
                : options_(std::move(descriptor.options))
            {
                if (options_.embed_dim <= 0) {
                    throw std::invalid_argument("Transformer encoder requires a positive embedding dimension.");
                }

                if (options_.positional_encoding.type != PositionalEncodingType::None ||
                    options_.positional_encoding.dropout > 0.0) {
                    positional_encoding_ = register_module(
                        "positional_encoding",
                        PositionalEncoding(options_.embed_dim, options_.positional_encoding));
                    }

                auto norm_options = torch::nn::LayerNormOptions(std::vector<int64_t>{options_.embed_dim})
                                         .eps(options_.layer_norm.eps)
                                         .elementwise_affine(options_.layer_norm.elementwise_affine);
                final_layer_norm_ = register_module("final_layer_norm", torch::nn::LayerNorm(norm_options));

                layers_.reserve(descriptor.layers.size());
                for (std::size_t index = 0; index < descriptor.layers.size(); ++index) {
                    auto layer = register_module("layer_" + std::to_string(index),
                                                 TransformerEncoderLayer(std::move(descriptor.layers[index]), options_));
                    layers_.push_back(std::move(layer));
                }
            }

            torch::Tensor forward(torch::Tensor input,
                                  const torch::Tensor& attn_mask = {},
                                  const torch::Tensor& key_padding_mask = {})
            {
                auto [normalised_input, shape] = normalise_to_sequence(std::move(input), options_.embed_dim);
                auto output = std::move(normalised_input);
                if (positional_encoding_) {
                    output = positional_encoding_->forward(std::move(output));
                }

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
            std::vector<TransformerEncoderLayer> layers_{};
            PositionalEncoding positional_encoding_{nullptr};
            torch::nn::LayerNorm final_layer_norm_{nullptr};
        };

        TORCH_MODULE(TransformerEncoder);

        class TransformerDecoderLayerImpl : public torch::nn::Module {
        public:
            TransformerDecoderLayerImpl(DecoderLayerDescriptor descriptor, const DecoderOptions& options)
                : embed_dim_(options.embed_dim),
                  layer_norm_options_(options.layer_norm)
            {
                if (embed_dim_ <= 0) {
                    throw std::invalid_argument("Transformer decoder layer requires a positive embedding dimension.");
                }

                auto norm_options = torch::nn::LayerNormOptions(std::vector<int64_t>{embed_dim_})
                                         .eps(layer_norm_options_.eps)
                                         .elementwise_affine(layer_norm_options_.elementwise_affine);

                norm1_ = register_module("norm1", torch::nn::LayerNorm(norm_options));
                norm2_ = register_module("norm2", torch::nn::LayerNorm(norm_options));
                norm3_ = register_module("norm3", torch::nn::LayerNorm(norm_options));

                self_attention_ = ::Nott::Attention::Details::register_attention(*this, "self_attention", std::move(descriptor.self_attention));

                cross_attention_ = ::Nott::Attention::Details::register_attention(*this, "cross_attention", std::move(descriptor.cross_attention));

                std::size_t module_index = 0;
                auto register_layer = [&](::Nott::Layer::Descriptor layer_descriptor) {
                    return std::visit(
                        [&](const auto& concrete_descriptor) {
                            return ::Nott::Layer::Details::build_registered_layer(
                                *this, concrete_descriptor, module_index++);
                        },
                        std::move(layer_descriptor));
                };

                self_attention_dropout_ = register_layer(std::move(descriptor.self_attention_dropout));
                cross_attention_dropout_ = register_layer(std::move(descriptor.cross_attention_dropout));

                feed_forward_layers_.reserve(descriptor.feed_forward.size());
                for (auto& layer : descriptor.feed_forward) {
                    feed_forward_layers_.push_back(register_layer(std::move(layer)));
                }

                feed_forward_dropout_ = register_layer(std::move(descriptor.feed_forward_dropout));
            }

            torch::Tensor forward(torch::Tensor target,
                                  const torch::Tensor& memory = {},
                                  const torch::Tensor& tgt_mask = {},
                                  const torch::Tensor& memory_mask = {},
                                  const torch::Tensor& tgt_key_padding_mask = {},
                                  const torch::Tensor& memory_key_padding_mask = {})
            {
                auto [normalised_target, target_shape] = normalise_to_sequence(std::move(target), embed_dim_);

                auto output = std::move(normalised_target);

                auto residual = output;
                auto self_norm = norm1_->forward(residual);
                auto self_attention = ::Nott::Attention::Details::forward_attention(self_attention_, self_norm, self_norm, self_norm, tgt_mask, tgt_key_padding_mask);
                if (self_attention_dropout_.forward) {
                    self_attention = self_attention_dropout_.forward(std::move(self_attention));
                    self_attention = ::Nott::Activation::Details::apply(
                        self_attention_dropout_.activation, std::move(self_attention));
                }
                output = residual + self_attention;

                if (memory.defined()) {
                    residual = output;
                    auto cross_norm = norm2_->forward(output);
                    auto cross_attention = ::Nott::Attention::Details::forward_attention(cross_attention_, cross_norm, memory, memory, memory_mask, memory_key_padding_mask);
                    if (cross_attention_dropout_.forward) {
                        cross_attention = cross_attention_dropout_.forward(std::move(cross_attention));
                        cross_attention = ::Nott::Activation::Details::apply(
                            cross_attention_dropout_.activation, std::move(cross_attention));
                    }
                    output = residual + cross_attention;
                }

                residual = output;
                auto feed_forward = norm3_->forward(output);
                for (auto& layer : feed_forward_layers_) {
                    if (!layer.forward) {
                        continue;
                    }
                    feed_forward = layer.forward(std::move(feed_forward));
                    feed_forward = ::Nott::Activation::Details::apply(
                        layer.activation, std::move(feed_forward));
                }
                if (feed_forward_dropout_.forward) {
                    feed_forward = feed_forward_dropout_.forward(std::move(feed_forward));
                    feed_forward = ::Nott::Activation::Details::apply(
                        feed_forward_dropout_.activation, std::move(feed_forward));
                }

                output = residual + feed_forward;
                return restore_from_sequence(std::move(output), target_shape);
            }

        private:
            std::int64_t embed_dim_{};
            LayerNormOptions layer_norm_options_{};
            ::Nott::Attention::Details::AttentionModule self_attention_{};
            ::Nott::Attention::Details::AttentionModule cross_attention_{};
            torch::nn::LayerNorm norm1_{nullptr};
            torch::nn::LayerNorm norm2_{nullptr};
            torch::nn::LayerNorm norm3_{nullptr};
            ::Nott::Layer::Details::RegisteredLayer self_attention_dropout_{};
            ::Nott::Layer::Details::RegisteredLayer cross_attention_dropout_{};
            std::vector<::Nott::Layer::Details::RegisteredLayer> feed_forward_layers_{};
            ::Nott::Layer::Details::RegisteredLayer feed_forward_dropout_{};
        };

        TORCH_MODULE(TransformerDecoderLayer);

        class TransformerDecoderImpl : public torch::nn::Module {
        public:
            explicit TransformerDecoderImpl(DecoderDescriptor descriptor)
                : options_(std::move(descriptor.options))
            {
                if (options_.embed_dim <= 0) {
                    throw std::invalid_argument("Transformer decoder requires a positive embedding dimension.");
                }

                if (options_.positional_encoding.type != PositionalEncodingType::None ||
                    options_.positional_encoding.dropout > 0.0) {
                    positional_encoding_ = register_module(
                        "positional_encoding",
                        PositionalEncoding(options_.embed_dim, options_.positional_encoding));
                }

                auto norm_options = torch::nn::LayerNormOptions(std::vector<int64_t>{options_.embed_dim})
                                         .eps(options_.layer_norm.eps)
                                         .elementwise_affine(options_.layer_norm.elementwise_affine);
                final_layer_norm_ = register_module("final_layer_norm", torch::nn::LayerNorm(norm_options));

                layers_.reserve(descriptor.layers.size());
                for (std::size_t index = 0; index < descriptor.layers.size(); ++index) {
                    auto layer = register_module(
                        "layer_" + std::to_string(index),
                        TransformerDecoderLayer(std::move(descriptor.layers[index]), options_));
                    layers_.push_back(std::move(layer));
                }
            }

            torch::Tensor forward(torch::Tensor target,
                                  torch::Tensor memory = {},
                                  const torch::Tensor& tgt_mask = {},
                                  const torch::Tensor& memory_mask = {},
                                  const torch::Tensor& tgt_key_padding_mask = {},
                                  const torch::Tensor& memory_key_padding_mask = {})
            {
                auto [normalised_target, target_shape] = normalise_to_sequence(std::move(target), options_.embed_dim);

                torch::Tensor normalised_memory;
                if (memory.defined()) {
                    auto memory_pair = normalise_to_sequence(std::move(memory), options_.embed_dim);
                    normalised_memory = std::move(memory_pair.first);
                }

                auto output = std::move(normalised_target);
                if (positional_encoding_) {
                    output = positional_encoding_->forward(std::move(output));
                }

                for (auto& layer : layers_) {
                    output = layer->forward(std::move(output),
                                            normalised_memory,
                                            tgt_mask,
                                            memory_mask,
                                            tgt_key_padding_mask,
                                            memory_key_padding_mask);
                }

                if (final_layer_norm_) {
                    output = final_layer_norm_->forward(output);
                }

                return restore_from_sequence(std::move(output), target_shape);
            }

        private:
            DecoderOptions options_{};
            std::vector<TransformerDecoderLayer> layers_{};
            PositionalEncoding positional_encoding_{nullptr};
            torch::nn::LayerNorm final_layer_norm_{nullptr};
        };

        TORCH_MODULE(TransformerDecoder);
    }

    using PositionalEncoding = Detail::PositionalEncoding;

    using TransformerEncoderLayer = Detail::TransformerEncoderLayer;
    using TransformerEncoder = Detail::TransformerEncoder;

    using TransformerDecoderLayer = Detail::TransformerDecoderLayer;
    using TransformerDecoder = Detail::TransformerDecoder;
}

#endif //Nott_CLASSIC_HPP