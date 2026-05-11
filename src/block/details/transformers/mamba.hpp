#ifndef Nott_MAMBA_HPP
#define Nott_MAMBA_HPP
/// https://arxiv.org/pdf/2312.00752
/// "Mamba: Linear-Time Sequence Modeling with Selective State Spaces", Gu et al., 2023 (arXiv:2312.00752).
/// Implements the selective state space sequence model with convolutional gating and RMS normalization
/// providing an alternative to attention for long-context sequence modeling.
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <algorithm>


#include <torch/torch.h>

namespace Nott::Block::Details::Transformer::Mamba {
    enum class NormalizationOrder {
        Pre,
        Post,
    };

    struct RMSNormOptions {
        double eps{1e-6};
        bool learnable{true};
    };

    struct SelectiveStateSpaceOptions {
        std::int64_t embed_dim{256};
        double state_expansion{2.0};
        std::int64_t ssm_layers{1};
        std::int64_t conv_kernel_size{4};
        double dropout{0.0};
        bool batch_first{true};
    };

    struct FeedForwardOptions {
        std::int64_t embed_dim{256};
        double expansion_ratio{2.0};
        double dropout{0.0};
        bool gated{true};
    };

    struct EncoderLayerDescriptor {
        SelectiveStateSpaceOptions selective_state{};
        FeedForwardOptions feed_forward{};
    };

    struct EncoderOptions {
        std::size_t layers{1};
        std::int64_t embed_dim{256};
        RMSNormOptions rms_norm{};
        NormalizationOrder normalization{NormalizationOrder::Pre};
        double residual_dropout{0.0};
        double feed_forward_dropout{0.0};
        bool residual_gating{true};
        bool feed_forward_gating{true};
        bool batch_first{true};
        bool final_layer_norm{true};
        SelectiveStateSpaceOptions selective_state{};
        FeedForwardOptions feed_forward{};
    };

    struct EncoderDescriptor {
        EncoderOptions options{};
        std::vector<EncoderLayerDescriptor> layers{};
    };

    [[nodiscard]] inline auto Encoder(const EncoderOptions& options) -> EncoderDescriptor
    {
        if (options.embed_dim <= 0) {
            throw std::invalid_argument("Mamba encoder requires a positive embedding dimension.");
        }
        if (options.layers == 0) {
            throw std::invalid_argument("Mamba encoder requires at least one layer.");
        }

        EncoderDescriptor descriptor{};
        descriptor.options = options;

        auto selective = options.selective_state;
        selective.embed_dim = options.embed_dim;
        auto feed_forward = options.feed_forward;
        feed_forward.embed_dim = options.embed_dim;

        descriptor.layers.reserve(options.layers);
        for (std::size_t index = 0; index < options.layers; ++index) {
            EncoderLayerDescriptor layer{};
            layer.selective_state = selective;
            layer.feed_forward = feed_forward;
            descriptor.layers.push_back(layer);
        }

        return descriptor;
    }

    namespace Detail {
        struct SelectiveStateSpaceState {
            torch::Tensor ssm{};
            torch::Tensor conv{};
        };

        struct SelectiveStateSpaceResult {
            torch::Tensor output{};
            SelectiveStateSpaceState state{};
        };

        class RMSNormImpl : public torch::nn::Module {
        public:
            RMSNormImpl(std::int64_t embed_dim, RMSNormOptions options)
                : embed_dim_(embed_dim),
                  options_(options)
            {
                if (embed_dim_ <= 0) {
                    throw std::invalid_argument("RMSNorm requires a positive embedding dimension.");
                }
                if (options_.learnable) {
                    weight_ = register_parameter("weight", torch::ones({embed_dim_}));
                }
            }

            torch::Tensor forward(torch::Tensor input)
            {
                if (!input.defined()) {
                    throw std::invalid_argument("RMSNorm requires a defined input tensor.");
                }
                const auto dims = input.dim();
                if (dims == 0) {
                    throw std::invalid_argument("RMSNorm expects a tensor with at least one dimension.");
                }

                auto variance = input.pow(2).mean(-1, true);
                auto normalized = input * torch::rsqrt(variance + options_.eps);
                if (weight_.defined()) {
                    std::vector<int64_t> shape(static_cast<std::size_t>(dims), 1);
                    shape.back() = embed_dim_;
                    normalized = normalized * weight_.view(shape);
                }
                return normalized;
            }

        private:
            std::int64_t embed_dim_{};
            RMSNormOptions options_{};
            torch::Tensor weight_{};
        };

        TORCH_MODULE(RMSNorm);

        class SelectiveStateSpaceImpl : public torch::nn::Module {
        public:
            explicit SelectiveStateSpaceImpl(SelectiveStateSpaceOptions options)
                : options_(std::move(options))
            {
                if (options_.embed_dim <= 0) {
                    throw std::invalid_argument("Selective state space requires a positive embedding dimension.");
                }
                if (options_.ssm_layers <= 0) {
                    throw std::invalid_argument("Selective state space requires at least one state layer.");
                }
                if (options_.conv_kernel_size < 1) {
                    throw std::invalid_argument("Selective state space convolution kernel must be positive.");
                }
                if (options_.dropout < 0.0 || options_.dropout >= 1.0) {
                    throw std::invalid_argument("Selective state space dropout must be in [0, 1).");
                }

                inner_dim_ = static_cast<std::int64_t>(std::llround(options_.state_expansion * static_cast<double>(options_.embed_dim)));
                if (inner_dim_ <= 0) {
                    inner_dim_ = options_.embed_dim;
                }

                in_proj_ = register_module(
                    "in_proj",
                    torch::nn::Linear(torch::nn::LinearOptions(options_.embed_dim, inner_dim_ * 2)));

                decay_linears_.reserve(static_cast<std::size_t>(options_.ssm_layers));
                input_linears_.reserve(static_cast<std::size_t>(options_.ssm_layers));

                for (std::int64_t layer = 0; layer < options_.ssm_layers; ++layer) {
                    auto decay = register_module(
                        "decay_" + std::to_string(layer),
                        torch::nn::Linear(torch::nn::LinearOptions(inner_dim_, inner_dim_)));
                    auto input = register_module(
                        "input_" + std::to_string(layer),
                        torch::nn::Linear(torch::nn::LinearOptions(inner_dim_, inner_dim_)));
                    decay_linears_.push_back(std::move(decay));
                    input_linears_.push_back(std::move(input));
                }

                out_proj_ = register_module(
                    "out_proj",
                    torch::nn::Linear(torch::nn::LinearOptions(inner_dim_, options_.embed_dim)));

                if (options_.conv_kernel_size > 1) {
                    conv_ = register_module(
                        "conv",
                        torch::nn::Conv1d(torch::nn::Conv1dOptions(inner_dim_, inner_dim_, options_.conv_kernel_size)
                                              .stride(1)
                                              .padding(0)
                                              .groups(inner_dim_)));
                }

                if (options_.dropout > 0.0) {
                    dropout_ = register_module(
                        "dropout",
                        torch::nn::Dropout(torch::nn::DropoutOptions(options_.dropout)));
                }
            }

            [[nodiscard]] std::int64_t hidden_size() const noexcept { return inner_dim_; }
            [[nodiscard]] std::int64_t num_layers() const noexcept { return options_.ssm_layers; }
            [[nodiscard]] std::int64_t conv_kernel_size() const noexcept { return options_.conv_kernel_size; }

            SelectiveStateSpaceResult forward_with_state(torch::Tensor input, const SelectiveStateSpaceState* state = nullptr)
            {
                if (!input.defined()) {
                    throw std::invalid_argument("Selective state space requires a defined input tensor.");
                }
                if (input.dim() != 3) {
                    throw std::invalid_argument("Selective state space expects inputs shaped as (batch, seq, feature) or (seq, batch, feature).");
                }

                bool transposed = false;
                auto tensor = input;
                if (!options_.batch_first) {
                    tensor = tensor.transpose(0, 1);
                    transposed = true;
                }

                const auto batch = tensor.size(0);
                const auto seq_len = tensor.size(1);

                auto projected = in_proj_->forward(tensor);
                auto chunks = projected.chunk(2, -1);
                auto content = chunks[0];
                auto gate = torch::sigmoid(chunks[1]);

                SelectiveStateSpaceState new_state{};

                if (conv_) {
                    auto conv_input = content.transpose(1, 2);
                    torch::Tensor prefix;
                    if (state && state->conv.defined() && state->conv.sizes().size() == 3 &&
                        state->conv.size(0) == batch && state->conv.size(1) == inner_dim_ &&
                        state->conv.size(2) == options_.conv_kernel_size - 1) {
                        prefix = state->conv;
                    } else {
                        prefix = torch::zeros({batch, inner_dim_, options_.conv_kernel_size - 1}, conv_input.options());
                    }
                    conv_input = torch::cat({prefix, conv_input}, 2);
                    auto conv_output = conv_->forward(conv_input);
                    content = conv_output.transpose(1, 2);
                    new_state.conv = conv_input.slice(2, conv_input.size(2) - (options_.conv_kernel_size - 1), conv_input.size(2));
                }

                torch::Tensor working_states;
                const auto state_shape = std::vector<int64_t>{options_.ssm_layers, batch, inner_dim_};
                if (state && state->ssm.defined() && state->ssm.sizes().size() == 3 &&
                    state->ssm.size(0) == options_.ssm_layers && state->ssm.size(1) == batch &&
                    state->ssm.size(2) == inner_dim_) {
                    working_states = state->ssm.clone();
                } else {
                    working_states = torch::zeros(state_shape, tensor.options());
                }

                std::vector<torch::Tensor> emissions(static_cast<std::size_t>(seq_len));

                for (std::int64_t t = 0; t < seq_len; ++t) {
                    auto u = content.select(1, t);
                    auto next_input = u;
                    for (std::int64_t layer = 0; layer < options_.ssm_layers; ++layer) {
                        auto hx = working_states[layer];
                        auto decay = torch::sigmoid(decay_linears_[static_cast<std::size_t>(layer)]->forward(next_input));
                        auto input_proj = torch::tanh(input_linears_[static_cast<std::size_t>(layer)]->forward(next_input));
                        hx = decay * hx + input_proj;
                        working_states[layer] = hx;
                        next_input = hx;
                    }
                    emissions[static_cast<std::size_t>(t)] = next_input;
                }

                auto output = torch::stack(emissions, 1);
                output = output * gate;
                output = out_proj_->forward(output);
                if (dropout_) {
                    output = dropout_->forward(output);
                }

                if (transposed) {
                    output = output.transpose(0, 1);
                }

                new_state.ssm = std::move(working_states);
                if (!new_state.conv.defined() && options_.conv_kernel_size > 1) {
                    new_state.conv = torch::zeros({batch, inner_dim_, options_.conv_kernel_size - 1}, tensor.options());
                }

                return {std::move(output), std::move(new_state)};
            }

        private:
            SelectiveStateSpaceOptions options_{};
            std::int64_t inner_dim_{};
            torch::nn::Linear in_proj_{nullptr};
            std::vector<torch::nn::Linear> decay_linears_{};
            std::vector<torch::nn::Linear> input_linears_{};
            torch::nn::Linear out_proj_{nullptr};
            torch::nn::Conv1d conv_{nullptr};
            torch::nn::Dropout dropout_{nullptr};
        };

        TORCH_MODULE(SelectiveStateSpace);

        struct EncoderLayerState {
            SelectiveStateSpaceState selective_state{};
        };

        struct EncoderLayerResult {
            torch::Tensor output{};
            EncoderLayerState state{};
        };

        class EncoderLayerImpl : public torch::nn::Module {
        public:
            EncoderLayerImpl(EncoderLayerDescriptor descriptor, const EncoderOptions& options)
                : embed_dim_(options.embed_dim),
                  normalization_(options.normalization),
                  residual_gating_(options.residual_gating),
                  feed_forward_gating_(options.feed_forward_gating)
            {
                if (embed_dim_ <= 0) {
                    throw std::invalid_argument("Mamba encoder layer requires a positive embedding dimension.");
                }

                auto selective_options = descriptor.selective_state;
                selective_options.embed_dim = embed_dim_;
                selective_options.batch_first = true;
                selective_state_ = register_module(
                    "selective_state",
                    SelectiveStateSpace(std::move(selective_options)));

                norm1_ = register_module("norm1", RMSNorm(embed_dim_, options.rms_norm));
                norm2_ = register_module("norm2", RMSNorm(embed_dim_, options.rms_norm));

                const auto inner_dim = std::max<std::int64_t>(
                    1,
                    static_cast<std::int64_t>(std::llround(descriptor.feed_forward.expansion_ratio * static_cast<double>(embed_dim_))));

                feed_forward_up_ = register_module(
                    "ff_up",
                    torch::nn::Linear(torch::nn::LinearOptions(embed_dim_, inner_dim)));

                if (feed_forward_gating_) {
                    feed_forward_gate_proj_ = register_module(
                        "ff_gate",
                        torch::nn::Linear(torch::nn::LinearOptions(embed_dim_, inner_dim)));
                }

                feed_forward_down_ = register_module(
                    "ff_down",
                    torch::nn::Linear(torch::nn::LinearOptions(inner_dim, embed_dim_)));

                if (options.residual_dropout > 0.0) {
                    residual_dropout_ = register_module(
                        "residual_dropout",
                        torch::nn::Dropout(torch::nn::DropoutOptions(options.residual_dropout)));
                }

                if (descriptor.feed_forward.dropout > 0.0 || options.feed_forward_dropout > 0.0) {
                    auto probability = std::max(descriptor.feed_forward.dropout, options.feed_forward_dropout);
                    feed_forward_dropout_ = register_module(
                        "feed_forward_dropout",
                        torch::nn::Dropout(torch::nn::DropoutOptions(probability)));
                }

                if (residual_gating_) {
                    residual_gate_ = register_parameter("residual_gate", torch::ones({embed_dim_}));
                }
                if (feed_forward_gating_) {
                    feed_forward_gate_param_ = register_parameter("feed_forward_gate", torch::ones({embed_dim_}));
                }
            }

            [[nodiscard]] SelectiveStateSpace selective_state_module() const { return selective_state_; }

            EncoderLayerResult forward_with_state(torch::Tensor input, const EncoderLayerState* state = nullptr)
            {
                auto working = input;
                const SelectiveStateSpaceState* selective_state_ptr = state ? &state->selective_state : nullptr;

                auto selective = [&]() {
                    if (normalization_ == NormalizationOrder::Pre) {
                        auto normed = norm1_->forward(working);
                        return selective_state_->forward_with_state(normed, selective_state_ptr);
                    }
                    return selective_state_->forward_with_state(working, selective_state_ptr);
                }();

                auto branch = selective.output;
                if (normalization_ == NormalizationOrder::Post) {
                    branch = norm1_->forward(branch);
                }
                branch = apply_residual_branch(std::move(branch));

                if (normalization_ == NormalizationOrder::Pre) {
                    working = working + branch;
                } else {
                    working = norm1_->forward(working + branch);
                }

                if (normalization_ == NormalizationOrder::Pre) {
                    auto ff_input = norm2_->forward(working);
                    auto ff_branch = compute_feed_forward(std::move(ff_input));
                    working = working + ff_branch;
                } else {
                    auto ff_branch = compute_feed_forward(working);
                    working = norm2_->forward(working + ff_branch);
                }

                EncoderLayerResult result{};
                result.output = std::move(working);
                result.state.selective_state = std::move(selective.state);
                return result;
            }

        private:
            torch::Tensor apply_residual_branch(torch::Tensor branch)
            {
                if (residual_gate_.defined()) {
                    branch = branch * residual_gate_.view({1, 1, -1});
                }
                if (residual_dropout_) {
                    branch = residual_dropout_->forward(branch);
                }
                return branch;
            }

            torch::Tensor compute_feed_forward(torch::Tensor input)
            {
                auto up = feed_forward_up_->forward(input);
                if (feed_forward_gating_ && feed_forward_gate_proj_) {
                    auto gate = torch::silu(feed_forward_gate_proj_->forward(input));
                    up = up * gate;
                } else {
                    up = torch::silu(up);
                }
                auto down = feed_forward_down_->forward(up);
                if (feed_forward_gate_param_.defined()) {
                    down = down * feed_forward_gate_param_.view({1, 1, -1});
                }
                if (feed_forward_dropout_) {
                    down = feed_forward_dropout_->forward(down);
                }
                return down;
            }

            std::int64_t embed_dim_{};
            NormalizationOrder normalization_{};
            bool residual_gating_{};
            bool feed_forward_gating_{};
            SelectiveStateSpace selective_state_{nullptr};
            RMSNorm norm1_{nullptr};
            RMSNorm norm2_{nullptr};
            torch::nn::Linear feed_forward_up_{nullptr};
            torch::nn::Linear feed_forward_gate_proj_{nullptr};
            torch::nn::Linear feed_forward_down_{nullptr};
            torch::nn::Dropout residual_dropout_{nullptr};
            torch::nn::Dropout feed_forward_dropout_{nullptr};
            torch::Tensor residual_gate_{};
            torch::Tensor feed_forward_gate_param_{};
        };

        TORCH_MODULE(EncoderLayer);

        struct EncoderState {
            std::vector<EncoderLayerState> layers{};
        };

        struct EncoderForwardResult {
            torch::Tensor output{};
            EncoderState state{};
        };

        class EncoderImpl : public torch::nn::Module {
        public:
            explicit EncoderImpl(EncoderDescriptor descriptor)
                : options_(std::move(descriptor.options))
            {
                if (descriptor.layers.empty()) {
                    throw std::invalid_argument("Mamba encoder requires at least one layer descriptor.");
                }

                layers_.reserve(descriptor.layers.size());
                for (std::size_t index = 0; index < descriptor.layers.size(); ++index) {
                    auto layer = register_module(
                        "layer_" + std::to_string(index),
                        EncoderLayer(std::move(descriptor.layers[index]), options_));
                    layers_.push_back(std::move(layer));
                }

                if (options_.final_layer_norm) {
                    final_norm_ = register_module("final_norm", RMSNorm(options_.embed_dim, options_.rms_norm));
                }
            }

            EncoderForwardResult forward_with_state(torch::Tensor input, const EncoderState* state = nullptr)
            {
                auto [tensor, transposed] = normalize_layout(std::move(input), options_.batch_first);
                EncoderState new_state;
                new_state.layers.resize(layers_.size());

                auto working = tensor;
                for (std::size_t index = 0; index < layers_.size(); ++index) {
                    const EncoderLayerState* layer_state = nullptr;
                    if (state && index < state->layers.size()) {
                        layer_state = &state->layers[index];
                    }
                    auto result = layers_[index]->forward_with_state(working, layer_state);
                    working = std::move(result.output);
                    new_state.layers[index] = std::move(result.state);
                }

                if (final_norm_) {
                    working = final_norm_->forward(working);
                }

                if (transposed) {
                    working = working.transpose(0, 1);
                }

                return {std::move(working), std::move(new_state)};
            }

            torch::Tensor forward(torch::Tensor input)
            {
                return forward_with_state(std::move(input)).output;
            }

            [[nodiscard]] EncoderState initial_state(std::int64_t batch, torch::Device device) const
            {
                if (batch <= 0) {
                    throw std::invalid_argument("initial_state requires a positive batch size.");
                }
                EncoderState state;
                state.layers.resize(layers_.size());
                auto options = torch::TensorOptions().device(device);
                for (std::size_t index = 0; index < layers_.size(); ++index) {
                    auto selective = layers_[index]->selective_state_module();
                    state.layers[index].selective_state.ssm = torch::zeros(
                        {selective->num_layers(), batch, selective->hidden_size()}, options);
                    if (selective->conv_kernel_size() > 1) {
                        state.layers[index].selective_state.conv = torch::zeros(
                            {batch, selective->hidden_size(), selective->conv_kernel_size() - 1}, options);
                    }
                }
                return state;
            }

        private:
            static std::pair<torch::Tensor, bool> normalize_layout(torch::Tensor tensor, bool batch_first)
            {
                if (tensor.dim() != 3) {
                    throw std::invalid_argument("Mamba encoder expects inputs shaped as (batch, seq, feature) or (seq, batch, feature).");
                }
                if (batch_first) {
                    return {std::move(tensor), false};
                }
                return {tensor.transpose(0, 1), true};
            }

            EncoderOptions options_{};
            std::vector<EncoderLayer> layers_{};
            RMSNorm final_norm_{nullptr};
        };

        TORCH_MODULE(Encoder);
    }

    using SelectiveStateSpaceState = Detail::SelectiveStateSpaceState;
    using EncoderLayerState = Detail::EncoderLayerState;
    using EncoderState = Detail::EncoderState;
    using EncoderForwardResult = Detail::EncoderForwardResult;

    using SelectiveStateSpace = Detail::SelectiveStateSpace;
    using EncoderLayer = Detail::EncoderLayer;
    using EncoderModule = Detail::Encoder;
}


#endif //Nott_MAMBA_HPP