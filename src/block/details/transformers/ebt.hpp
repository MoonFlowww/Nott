#ifndef Nott_EBT_HPP
#define Nott_EBT_HPP
/// "Energy-Based Transformers", Haziza et al., 2025 (arXiv:2507.02092).
/// Models discrete or continuous modalities by refining latent tokens through learned energy gradients instead of explicit softmax attention.
/// https://arxiv.org/pdf/2507.02092
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <deque>
#include <tuple>
#include <utility>
#include <vector>
#include <cmath>

#include <torch/torch.h>

#ifdef TORCH_CUDA_AVAILABLE
#include <ATen/cuda/CUDAEvent.h>
#include <ATen/cuda/CUDAStream.h>
#endif

namespace Nott::Block::Details::Transformer::EBT {
    enum class ModalityType {
        Discrete,
        Continuous,
    };

    struct ModalityOptions {
        ModalityType type{ModalityType::Discrete};
        std::int64_t vocab_size{0};
        std::int64_t input_dim{0};
        std::int64_t embed_dim{128};
    };

    struct EnergyScorerOptions {
        std::size_t depth{2};
        std::int64_t hidden_size{256};
        std::int64_t modality_heads{1};
    };

    struct OptimizerOptions {
        double learning_rate{1e-1};
        double momentum{0.0};
        double gradient_clip_norm{0.0};
    };

    struct RefinementOptions {
        std::size_t max_steps{8};
        double tolerance{1e-5};
        bool stop_on_plateau{true};
    };

    struct EncoderOptions {
        ModalityOptions modality{};
        EnergyScorerOptions energy{};
        OptimizerOptions optimizer{};
        RefinementOptions refinement{};
    };

    struct DecoderOptions {
        ModalityOptions target{};
        std::optional<ModalityOptions> context{};
        EnergyScorerOptions energy{};
        OptimizerOptions optimizer{};
        RefinementOptions refinement{};
    };

    struct EncoderDescriptor {
        EncoderOptions options{};
    };

    struct DecoderDescriptor {
        DecoderOptions options{};
    };

    namespace Detail {
        class TokenEmbeddingImpl : public torch::nn::Module {
        public:
            explicit TokenEmbeddingImpl(ModalityOptions options)
                : options_(std::move(options))
            {
                if (options_.embed_dim <= 0) {
                    throw std::invalid_argument("Token embedding requires a positive embedding dimension.");
                }

                if (options_.type == ModalityType::Discrete) {
                    if (options_.vocab_size <= 0) {
                        throw std::invalid_argument("Discrete modality requires a positive vocabulary size.");
                    }
                    embedding_ = register_module(
                        "embedding",
                        torch::nn::Embedding(torch::nn::EmbeddingOptions(options_.vocab_size, options_.embed_dim)));
                } else {
                    if (options_.input_dim <= 0) {
                        throw std::invalid_argument("Continuous modality requires a positive input dimension.");
                    }
                    if (options_.input_dim != options_.embed_dim) {
                        projector_ = register_module(
                            "projector",
                            torch::nn::Linear(torch::nn::LinearOptions(options_.input_dim, options_.embed_dim)));
                    }
                }
            }

            [[nodiscard]] auto embedding_dim() const noexcept -> std::int64_t
            {
                return options_.embed_dim;
            }

            torch::Tensor forward(torch::Tensor input)
            {
                if (!input.defined()) {
                    return input;
                }

                if (options_.type == ModalityType::Discrete) {
                    if (!embedding_) {
                        throw std::logic_error("Discrete embedding requested but the embedding module is not initialised.");
                    }
                    if (input.scalar_type() == torch::kLong || input.scalar_type() == torch::kInt64) {
                        return embedding_->forward(std::move(input));
                    }

                    if (input.dim() == 0) {
                        input = input.unsqueeze(0);
                    }
                    if (input.size(-1) != options_.vocab_size) {
                        throw std::invalid_argument("Token logits must match the configured vocabulary size.");
                    }

                    auto logits = std::move(input);
                    auto probabilities = torch::softmax(logits, -1);
                    auto hard_indices = std::get<0>(torch::max(probabilities, -1));
                    auto hard = torch::one_hot(hard_indices, options_.vocab_size)
                                    .to(probabilities.dtype())
                                    .to(probabilities.device());
                    auto straight_through = hard + probabilities - probabilities.detach();

                    auto view = straight_through.contiguous().view({-1, options_.vocab_size});
                    auto embeddings = torch::matmul(view, embedding_->weight);
                    auto shape = straight_through.sizes().vec();
                    shape.back() = options_.embed_dim;
                    return embeddings.view(shape);
                }

                if (projector_) {
                    return projector_->forward(std::move(input));
                }
                return input;
            }

        private:
            ModalityOptions options_{};
            torch::nn::Embedding embedding_{nullptr};
            torch::nn::Linear projector_{nullptr};
        };

        TORCH_MODULE(TokenEmbedding);

        class EnergyScorerImpl : public torch::nn::Module {
        public:
            EnergyScorerImpl(std::int64_t feature_dim, EnergyScorerOptions options)
                : feature_dim_(feature_dim), options_(std::move(options))
            {
                if (feature_dim_ <= 0) {
                    throw std::invalid_argument("Energy scorer requires a positive feature dimension.");
                }
                if (options_.hidden_size <= 0) {
                    throw std::invalid_argument("Energy scorer requires a positive hidden size.");
                }
                if (options_.modality_heads <= 0) {
                    throw std::invalid_argument("Energy scorer requires at least one modality head.");
                }

                const auto hidden_layers = std::max<std::size_t>(1, options_.depth);
                for (std::int64_t head = 0; head < options_.modality_heads; ++head) {
                    auto sequence = torch::nn::Sequential();
                    auto in_dim = feature_dim_;
                    for (std::size_t layer = 0; layer < hidden_layers; ++layer) {
                        sequence->push_back(torch::nn::Linear(torch::nn::LinearOptions(in_dim, options_.hidden_size)));
                        sequence->push_back(torch::nn::SiLU());
                        in_dim = options_.hidden_size;
                    }
                    sequence->push_back(torch::nn::Linear(torch::nn::LinearOptions(in_dim, 1)));
                    auto module_name = "head_" + std::to_string(head);
                    heads_.push_back(register_module(module_name, sequence));
                }
            }

            [[nodiscard]] auto feature_dim() const noexcept -> std::int64_t
            {
                return feature_dim_;
            }

            torch::Tensor forward(torch::Tensor features)
            {
                if (!features.defined()) {
                    return features;
                }
                auto normalised = ensure_three_dim(std::move(features));
                const auto batch = normalised.size(0);
                const auto sequence = normalised.size(1);
                auto flattened = normalised.contiguous().view({batch * sequence, feature_dim_});

                auto energy = torch::zeros({batch}, flattened.options());
                for (auto& head : heads_) {
                    auto head_output = head->forward(flattened);
                    head_output = head_output.view({batch, sequence, 1});
                    energy = energy + head_output.sum(1).squeeze(-1);
                }

                energy = energy / static_cast<double>(heads_.size());
                return energy;
            }

        private:
            static torch::Tensor ensure_three_dim(torch::Tensor tensor)
            {
                if (!tensor.defined()) {
                    return tensor;
                }
                if (tensor.dim() == 1) {
                    tensor = tensor.unsqueeze(0).unsqueeze(0);
                } else if (tensor.dim() == 2) {
                    tensor = tensor.unsqueeze(1);
                } else if (tensor.dim() < 1 || tensor.dim() > 3) {
                    throw std::invalid_argument("Energy scorer expects features with rank 1, 2 or 3.");
                }
                if (tensor.size(-1) <= 0) {
                    throw std::invalid_argument("Energy scorer received an empty feature dimension.");
                }
                return tensor;
            }

            std::int64_t feature_dim_{};
            EnergyScorerOptions options_{};
            std::vector<torch::nn::Sequential> heads_{};
        };

        TORCH_MODULE(EnergyScorer);

        class RefinementLoopImpl : public torch::nn::Module {
        public:
            RefinementLoopImpl(ModalityOptions prediction_options,
                               std::optional<ModalityOptions> context_options,
                               EnergyScorerOptions scorer_options,
                               OptimizerOptions optimizer_options,
                               RefinementOptions refinement_options)
                : prediction_options_(std::move(prediction_options)),
                  context_options_(std::move(context_options)),
                  optimizer_options_(std::move(optimizer_options)),
                  refinement_options_(std::move(refinement_options))
            {
                prediction_embedding_ = register_module("prediction_embedding", TokenEmbedding(prediction_options_));

                std::int64_t feature_dim = prediction_embedding_->embedding_dim();
                if (context_options_.has_value()) {
                    context_embedding_ = register_module("context_embedding", TokenEmbedding(*context_options_));
                    feature_dim += context_embedding_->embedding_dim();
                }

                energy_ = register_module("energy", EnergyScorer(feature_dim, std::move(scorer_options)));
            }

            torch::Tensor forward(torch::Tensor predictions, const torch::Tensor& context = {})
            {
                if (context_options_.has_value() && !context.defined()) {
                    throw std::invalid_argument("Context tensor is required for the configured EBT decoder.");
                }
                auto [refined, trace] = run_refinement(std::move(predictions), context);
                last_energy_trace_ = std::move(trace);
                return refined;
            }

            [[nodiscard]] torch::Tensor last_energy_trace() const
            {
                return last_energy_trace_;
            }

        private:
            static torch::Tensor normalise_tensor(torch::Tensor tensor)
            {
                if (!tensor.defined()) {
                    return tensor;
                }
                if (tensor.dim() == 0) {
                    tensor = tensor.unsqueeze(0);
                }
                return tensor;
            }

            torch::Tensor combine_with_context(torch::Tensor prediction_features, const torch::Tensor& context)
            {
                if (!context_embedding_) {
                    return prediction_features;
                }
                if (!context.defined()) {
                    throw std::invalid_argument("Context embedding requested but no context tensor was supplied.");
                }

                auto context_features = context_embedding_->forward(context);
                auto prediction_shape = prediction_features.sizes();
                if (prediction_shape.size() < 2) {
                    prediction_features = prediction_features.view({prediction_shape[0], 1, -1});
                    prediction_shape = prediction_features.sizes();
                }

                std::vector<int64_t> expanded_shape(prediction_features.dim(), 1);
                expanded_shape[0] = prediction_shape[0];
                if (prediction_features.dim() >= 2) {
                    expanded_shape[1] = prediction_shape[1];
                }

                if (context_features.dim() == prediction_features.dim() &&
                    context_features.size(1) == prediction_features.size(1)) {
                    /// already aligned
                } else if (context_features.dim() == prediction_features.dim()) {
                    context_features = context_features.mean(1, true);
                } else if (context_features.dim() + 1 == prediction_features.dim()) {
                    context_features = context_features.unsqueeze(1);
                } else {
                    context_features = context_features.view({context_features.size(0), 1, -1});
                }

                context_features = context_features.expand({
                    expanded_shape[0],
                    expanded_shape[1],
                    context_features.size(-1)});
                std::vector<torch::Tensor> concat_inputs{};
                concat_inputs.reserve(2);
                concat_inputs.push_back(std::move(prediction_features));
                concat_inputs.push_back(std::move(context_features));
                return torch::cat(concat_inputs, -1);
            }

            std::pair<torch::Tensor, torch::Tensor> run_refinement(torch::Tensor predictions, const torch::Tensor& context)
            {
                auto state = normalise_tensor(std::move(predictions));
                state = state.clone();
                state.requires_grad_(true);

                torch::Tensor velocity = torch::zeros_like(state);
                std::vector<torch::Tensor> energy_history{};
                energy_history.reserve(refinement_options_.max_steps + 1);

                torch::Tensor previous_energy{};

                auto evaluate_plateau = [&](double improvement_value) {
                    if (!refinement_options_.stop_on_plateau) {
                        return false;
                    }
                    if (std::abs(improvement_value) < refinement_options_.tolerance) {
                        return true;
                    }
                    if (improvement_value < -refinement_options_.tolerance) {
                        return true;
                    }
                    return false;
                };

#ifdef TORCH_CUDA_AVAILABLE
                struct PendingImprovement {
                    at::cuda::CUDAEvent event{};
                    torch::Tensor value;
                };
                std::deque<PendingImprovement> pending_improvements{};
#endif

                auto poll_pending_improvements = [&](bool synchronise_all) {
#ifdef TORCH_CUDA_AVAILABLE
                    if (!state.is_cuda()) {
                        return false;
                    }
                    if (at::cuda::isCurrentStreamCapturing()) {
                        return false;
                    }

                    while (!pending_improvements.empty()) {
                        auto& front = pending_improvements.front();
                        if (!front.event.query()) {
                            if (!synchronise_all) {
                                return false;
                            }
                            break;
                        }
                        auto improvement_value = front.value.item<double>();
                        pending_improvements.pop_front();
                        if (evaluate_plateau(improvement_value)) {
                            return true;
                        }
                    }
#else
                    (void)synchronise_all;
#endif
                    return false;
                };

                bool should_break = false;

                for (std::size_t step = 0; step < refinement_options_.max_steps && !should_break; ++step) {
#ifdef TORCH_CUDA_AVAILABLE
                    if (poll_pending_improvements(false)) {
                        break;
                    }
#endif
                    auto prediction_features = prediction_embedding_->forward(state);
                    auto combined_features = combine_with_context(std::move(prediction_features), context);
                    auto energy_batch = energy_->forward(combined_features);
                    auto energy_scalar = energy_batch.mean();
                    energy_history.push_back(energy_scalar.detach());

                    if (previous_energy.defined()) {
                        auto improvement = (previous_energy - energy_batch).mean();
#ifdef TORCH_CUDA_AVAILABLE
                        if (state.is_cuda()) {
                            auto staged = improvement.detach().to(torch::kCPU, torch::kFloat64, /*non_blocking=*/true);
                            at::cuda::CUDAEvent event;
                            event.record(at::cuda::getCurrentCUDAStream());
                            pending_improvements.push_back(PendingImprovement{std::move(event), std::move(staged)});
                        } else
#endif
                        {
                            should_break = evaluate_plateau(improvement.item<double>());
                            if (should_break) {
                                break;
                            }
                        }
                    }

                    previous_energy = energy_batch.detach();

                    torch::autograd::variable_list outputs{energy_scalar};
                    torch::autograd::variable_list inputs{state};
                    torch::autograd::variable_list grad_outputs{};
                    auto gradients = torch::autograd::grad(outputs, inputs, grad_outputs, /*retain_graph=*/true, /*create_graph=*/true)[0];
                    if (optimizer_options_.gradient_clip_norm > 0.0) {
                        auto grad_norm = gradients.norm().clamp_min(1e-12);
                        auto max_norm = optimizer_options_.gradient_clip_norm;
                        auto clip = (max_norm / grad_norm).clamp_max(1.0);
                        gradients = gradients * clip;
                    }

                    velocity = optimizer_options_.momentum * velocity + gradients;
                    state = state - optimizer_options_.learning_rate * velocity;
                }

#ifdef TORCH_CUDA_AVAILABLE
                if (!should_break) {
                    should_break = poll_pending_improvements(true);
                }
#else
                (void)should_break;
#endif


                if (energy_history.empty()) {
                    energy_history.push_back(torch::zeros({}, state.options()));
                }

                auto stacked_history = torch::stack(energy_history);
                return {state, stacked_history};
            }

            ModalityOptions prediction_options_{};
            std::optional<ModalityOptions> context_options_{};
            OptimizerOptions optimizer_options_{};
            RefinementOptions refinement_options_{};
            TokenEmbedding prediction_embedding_{nullptr};
            TokenEmbedding context_embedding_{nullptr};
            EnergyScorer energy_{nullptr};
            torch::Tensor last_energy_trace_{};
        };

        TORCH_MODULE(RefinementLoop);

        class EncoderImpl : public torch::nn::Module {
        public:
            explicit EncoderImpl(EncoderDescriptor descriptor)
                : options_(std::move(descriptor.options))
            {
                refiner_ = register_module(
                    "refiner",
                    RefinementLoop(options_.modality, std::nullopt, options_.energy, options_.optimizer, options_.refinement));
            }

            torch::Tensor forward(torch::Tensor predictions)
            {
                return refiner_->forward(std::move(predictions));
            }

            [[nodiscard]] torch::Tensor last_energy_trace() const
            {
                return refiner_->last_energy_trace();
            }

        private:
            EncoderOptions options_{};
            RefinementLoop refiner_{nullptr};
        };

        TORCH_MODULE(Encoder);

        class DecoderImpl : public torch::nn::Module {
        public:
            explicit DecoderImpl(DecoderDescriptor descriptor)
                : options_(std::move(descriptor.options))
            {
                refiner_ = register_module(
                    "refiner",
                    RefinementLoop(options_.target, options_.context, options_.energy, options_.optimizer, options_.refinement));
            }

            torch::Tensor forward(torch::Tensor predictions, const torch::Tensor& context)
            {
                return refiner_->forward(std::move(predictions), context);
            }

            [[nodiscard]] torch::Tensor last_energy_trace() const
            {
                return refiner_->last_energy_trace();
            }

        private:
            DecoderOptions options_{};
            RefinementLoop refiner_{nullptr};
        };

        TORCH_MODULE(Decoder);
    }

    [[nodiscard]] inline auto Encoder(const EncoderOptions& options) -> EncoderDescriptor
    {
        EncoderDescriptor descriptor{};
        descriptor.options = options;
        return descriptor;
    }


    [[nodiscard]] inline auto Decoder(const DecoderOptions& options) -> DecoderDescriptor
    {
        DecoderDescriptor descriptor{};
        descriptor.options = options;
        return descriptor;
    }

    using TokenEmbedding = Detail::TokenEmbedding;
    using EnergyScorer = Detail::EnergyScorer;
    using RefinementLoop = Detail::RefinementLoop;
    using EncoderModule = Detail::Encoder;
    using DecoderModule = Detail::Decoder;
}

#endif //Nott_EBT_HPP