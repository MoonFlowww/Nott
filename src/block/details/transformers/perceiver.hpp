#ifndef Nott_PERCEIVER_HPP
#define Nott_PERCEIVER_HPP
/// "Perceiver: General Perception with Iterative Attention" https://arxiv.org/pdf/2103.03206
/// Hybrid architecture attending from a small latent array to high-dimensional inputs via cross and latent self-attention.

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

#include "../../../activation/activation.hpp"
#include "../../../activation/apply.hpp"

namespace Nott::Block::Details::Transformer::Perceiver {

    struct AttentionOptions {
        std::int64_t query_dim{512};
        std::int64_t key_dim{512};
        std::int64_t num_heads{8};
        double dropout{0.0};
        bool bias{true};
        bool batch_first{true};
    };

    struct FeedForwardOptions {
        std::int64_t embed_dim{512};
        double mlp_ratio{4.0};
        ::Nott::Activation::Descriptor activation{::Nott::Activation::GeLU};
        bool bias{true};
        double dropout{0.0};
    };

    struct EncoderLayerDescriptor {
        FeedForwardOptions feed_forward{};
    };

    struct EncoderOptions {
        std::size_t layers{4};
        std::size_t self_layers{2};
        std::size_t repeats{1};
        std::int64_t latent_dim{512};
        std::int64_t input_dim{512};
        std::size_t latent_slots{256};
        AttentionOptions cross_attention{};
        AttentionOptions self_attention{};
        FeedForwardOptions feed_forward{};
        double residual_dropout{0.0};
        double attention_dropout{0.0};
    };

    struct EncoderDescriptor {
        EncoderOptions options{};
        std::vector<EncoderLayerDescriptor> layers{};
    };

    [[nodiscard]] inline auto Encoder(const EncoderOptions& options) -> EncoderDescriptor
    {
        if (options.latent_dim <= 0) {
            throw std::invalid_argument("Perceiver requires a positive latent dimension.");
        }
        if (options.latent_slots == 0) {
            throw std::invalid_argument("Perceiver requires at least one latent slot.");
        }
        if (options.layers == 0) {
            throw std::invalid_argument("Perceiver requires at least one cross-attention layer.");
        }

        EncoderDescriptor descriptor{};
        descriptor.options = options;
        descriptor.layers.reserve(options.layers);
        for (std::size_t index = 0; index < options.layers; ++index) {
            EncoderLayerDescriptor layer{};
            layer.feed_forward = options.feed_forward;
            descriptor.layers.emplace_back(std::move(layer));
        }
        return descriptor;
    }

    namespace Detail {
        inline auto normalise_inputs(torch::Tensor input, std::int64_t expected_dim)
            -> std::pair<torch::Tensor, std::int64_t>
        {
            if (!input.defined()) {
                throw std::invalid_argument("Perceiver encoder requires a defined input tensor.");
            }

            if (input.dim() == 2) {
                if (expected_dim > 0 && input.size(1) != expected_dim) {
                    throw std::invalid_argument("Perceiver input feature dimension mismatch.");
                }
                return {input.unsqueeze(1), input.size(1)};
            }

            if (input.dim() == 3) {
                if (expected_dim > 0 && input.size(2) != expected_dim) {
                    throw std::invalid_argument("Perceiver input feature dimension mismatch.");
                }
                return {input, input.size(2)};
            }

            auto flattened = input.flatten(1, input.dim() - 2);
            const auto feature_dim = flattened.size(-1);
            if (expected_dim > 0 && feature_dim != expected_dim) {
                throw std::invalid_argument("Perceiver input feature dimension mismatch.");
            }
            return {flattened, feature_dim};
        }

        class FeedForwardImpl : public torch::nn::Module {
        public:
            FeedForwardImpl(FeedForwardOptions options, std::int64_t embed_dim)
                : options_(std::move(options))
            {
                const auto hidden_dim = static_cast<std::int64_t>(std::llround(
                    options_.mlp_ratio * static_cast<double>(embed_dim)));
                auto linear1 = torch::nn::Linear(torch::nn::LinearOptions(embed_dim, hidden_dim)
                                                      .bias(options_.bias));
                auto linear2 = torch::nn::Linear(torch::nn::LinearOptions(hidden_dim, embed_dim)
                                                      .bias(options_.bias));
                linear1_ = register_module("linear1", std::move(linear1));
                linear2_ = register_module("linear2", std::move(linear2));
                if (options_.dropout > 0.0) {
                    dropout_ = register_module(
                        "dropout",
                        torch::nn::Dropout(torch::nn::DropoutOptions(options_.dropout)));
                }
            }

            torch::Tensor forward(torch::Tensor input)
            {
                auto hidden = linear1_->forward(input);
                hidden = ::Nott::Activation::Details::apply(options_.activation.type, std::move(hidden));
                if (dropout_) {
                    hidden = dropout_->forward(hidden);
                }
                hidden = linear2_->forward(hidden);
                if (dropout_) {
                    hidden = dropout_->forward(hidden);
                }
                return hidden;
            }

        private:
            FeedForwardOptions options_{};
            torch::nn::Linear linear1_{nullptr};
            torch::nn::Linear linear2_{nullptr};
            torch::nn::Dropout dropout_{nullptr};
        };

        TORCH_MODULE(FeedForward);

        class PerceiverEncoderImpl : public torch::nn::Module {
        public:
            explicit PerceiverEncoderImpl(EncoderDescriptor descriptor)
                : options_(std::move(descriptor.options))
            {
                if (options_.latent_dim <= 0) {
                    throw std::invalid_argument("Perceiver encoder requires a positive latent dimension.");
                }

                if (options_.input_dim != options_.latent_dim) {
                    input_projection_ = register_module(
                        "input_projection",
                        torch::nn::Linear(torch::nn::LinearOptions(options_.input_dim, options_.latent_dim)
                                              .bias(true)));
                }

                cross_attention_ = register_module(
                    "cross_attention",
                    torch::nn::MultiheadAttention(torch::nn::MultiheadAttentionOptions(options_.latent_dim,
                                                                                      options_.cross_attention.num_heads)
                                                      .bias(options_.cross_attention.bias)
                                                      .dropout(options_.cross_attention.dropout)));

                latent_attention_ = register_module(
                    "latent_attention",
                    torch::nn::MultiheadAttention(torch::nn::MultiheadAttentionOptions(options_.latent_dim,
                                                                                       options_.self_attention.num_heads)
                                                       .bias(options_.self_attention.bias)
                                                       .dropout(options_.self_attention.dropout)));

                feed_forward_cross_ = register_module(
                    "feed_forward_cross", FeedForward(options_.feed_forward, options_.latent_dim));
                feed_forward_latent_ = register_module(
                    "feed_forward_latent", FeedForward(options_.feed_forward, options_.latent_dim));

                if (options_.residual_dropout > 0.0) {
                    residual_dropout_ = register_module(
                        "residual_dropout",
                        torch::nn::Dropout(torch::nn::DropoutOptions(options_.residual_dropout)));
                }
                if (options_.attention_dropout > 0.0) {
                    attention_dropout_ = register_module(
                        "attention_dropout",
                        torch::nn::Dropout(torch::nn::DropoutOptions(options_.attention_dropout)));
                }

                latents_ = register_parameter("latents",torch::nn::functional::normalize(torch::randn({ static_cast<int64_t>(options_.latent_slots), options_.latent_dim}))
                );

            }

            torch::Tensor forward(torch::Tensor input)
            {
                auto normalized = normalise_inputs(std::move(input), options_.input_dim);
                auto data = std::move(normalized.first);
                const auto batch = data.size(0);
                auto latents = latents_.unsqueeze(0).expand({batch, -1, -1}).clone();
                auto encoded_data = input_projection_ ? input_projection_->forward(data) : data;

                for (std::size_t repeat = 0; repeat < options_.repeats; ++repeat) {
                    auto attn = cross_attention_->forward(latents, encoded_data, encoded_data);
                    auto attn_output = std::get<0>(attn);
                    if (attention_dropout_) {
                        attn_output = attention_dropout_->forward(attn_output);
                    }
                    latents = latents + attn_output;
                    auto ff = feed_forward_cross_->forward(latents);
                    if (residual_dropout_) {
                        ff = residual_dropout_->forward(ff);
                    }
                    latents = latents + ff;

                    for (std::size_t layer = 0; layer < options_.self_layers; ++layer) {
                        auto latent_attn = latent_attention_->forward(latents, latents, latents);
                        auto latent_output = std::get<0>(latent_attn);
                        if (attention_dropout_) {
                            latent_output = attention_dropout_->forward(latent_output);
                        }
                        latents = latents + latent_output;
                        auto latent_ff = feed_forward_latent_->forward(latents);
                        if (residual_dropout_) {
                            latent_ff = residual_dropout_->forward(latent_ff);
                        }
                        latents = latents + latent_ff;
                    }
                }

                return latents;
            }

        private:
            EncoderOptions options_{};
            torch::nn::Linear input_projection_{nullptr};
            torch::nn::MultiheadAttention cross_attention_{nullptr};
            torch::nn::MultiheadAttention latent_attention_{nullptr};
            FeedForward feed_forward_cross_{nullptr};
            FeedForward feed_forward_latent_{nullptr};
            torch::nn::Dropout residual_dropout_{nullptr};
            torch::nn::Dropout attention_dropout_{nullptr};
            torch::Tensor latents_{};
        };

        TORCH_MODULE(PerceiverEncoder);
    }

    using PerceiverEncoder = Detail::PerceiverEncoder;

}

#endif // Nott_PERCEIVER_HPP