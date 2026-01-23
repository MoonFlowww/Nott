#ifndef Nott_CORE_HPP
#define Nott_CORE_HPP
/*
 * Core orchestrator of the framework.
 * ---------------------------------------------------------------------------
 * Planned responsibilities:
 *  - Define the master constexpr configuration (model topology, data pipeline,
 *    optimization, metrics, regularization) that remains immutable at runtime.
 *  - Collect requests from main.cpp and route them to the appropriate
 *    factories located under the module directories (activation, block, layer,
 *    optimizer, etc.).
 *  - Materialise the selected components as inline function objects or raw
 *    pointers that can be handed over to network.hpp without leaking
 *    higher-level orchestration details.
 *  - Push only the latency-inflating feature toggles (regularisation, data
 *    augmentation, k-folding, etc.) behind `if constexpr` / template
 *    specialisation so the generated runtime code path is branchless once the
 *    compile-time configuration is instantiated; constant-cost utilities can
 *    stay in regular functions for readability.
 *  - Expose helper APIs to retrieve training, evaluation, calibration and
 *    monitoring routines pre-bound to the compile-time configuration.
 */
#include <iostream>
#include <algorithm>
#include <array>
#include <functional>
#include <chrono>
#include <cassert>
#include <cstddef>
#include <deque>
#include <initializer_list>
#include <cmath>
#include <memory>
#include <optional>
#include <random>
#include <sstream>
#include <new>
#include <stdexcept>
#include <type_traits>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>
#include <deque>
#include <filesystem>
#include <unordered_map>
#include <unordered_set>
#include <limits>
#include <cctype>
#include <iterator>


#include <torch/torch.h>
#include <torch/optim/adamw.h>
#include <torch/optim/sgd.h>
#ifdef TORCH_CUDA_AVAILABLE
#include <torch/cuda.h>
#include <torch/cuda/amp.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>
#include <ATen/cuda/CUDAStream.h>
#include <nvToolsExt.h>
#endif
#include <ATen/DeviceGuard.h>
#include <ATen/autocast_mode.h>


#include "common/streaming.hpp"

#include "common/graph.hpp"
#include "common/save_load.hpp"
#include "utils/terminal.hpp"
#include "activation/activation.hpp"
#include "activation/apply.hpp"
#include "initialization/initialization.hpp"
#include "attention/details/head.hpp"
#include "block/block.hpp"
#include "initialization/apply.hpp"
#include "layer/layer.hpp"
#include "block/details/blocks/residual.hpp"
#include "block/details/blocks/sequential.hpp"
#include "evaluation/evaluation.hpp"
#include "loss/loss.hpp"
#include "loss/details/mse.hpp"
#include "optimizer/optimizer.hpp"
#include "lrscheduler/lrscheduler.hpp"
#include "block/block.hpp"
#include "layer/details/positional_encoding.hpp"
#include "regularization/regularization.hpp"
#include "regularization/apply.hpp"
#include "calibration/calibration.hpp"
#include "training/kfold.hpp"

namespace Nott {
    template<class... Ts>
    struct Overloaded : Ts... {
        using Ts::operator()...;
    };

    template<class... Ts>
    Overloaded(Ts...) -> Overloaded<Ts...>;

    namespace Core {
        template<std::size_t BufferVRAMBatches>
        struct DevicePolicy {
            [[nodiscard]] static torch::Device select() {
                if constexpr (BufferVRAMBatches > 0) {
                    if (!torch::cuda::is_available()) {
                        throw std::runtime_error("CUDA device requested for VRAM buffering but is unavailable.");
                    }
                    return torch::Device(torch::kCUDA);
                } else {
                    return torch::Device(torch::kCPU);
                }
            }
        };

        template<std::size_t Epochs,
            std::size_t BatchSize,
            bool Shuffle,
            std::size_t BufferVRAMBatches,
            class DevicePolicyT = DevicePolicy<BufferVRAMBatches> >
        struct TrainingConfig {
            static_assert(Epochs > 0, "TrainingConfig requires at least one epoch.");
            static_assert(BatchSize > 0, "TrainingConfig requires a positive batch size.");

            static constexpr std::size_t epochs = Epochs;
            static constexpr std::size_t batch_size = BatchSize;
            static constexpr bool shuffle = Shuffle;
            static constexpr std::size_t buffer_vram = BufferVRAMBatches;

            using DevicePolicy = DevicePolicyT;
        };

        using SupervisedSample = std::pair<torch::Tensor, torch::Tensor>;
        using SupervisedDataset = std::vector<SupervisedSample>;

        using DefaultTrainingConfig = TrainingConfig<10, 32, true, 0>;
        inline constexpr auto kDefaultTrainingConfig = DefaultTrainingConfig{};
    }


    struct TrainOptions {
        std::size_t epoch{Core::kDefaultTrainingConfig.epochs};
        std::size_t batch_size{Core::kDefaultTrainingConfig.batch_size};
        bool fold{false};
        bool shuffle{Core::kDefaultTrainingConfig.shuffle};
        bool monitor{true};
        bool restore_best_state{false};
        std::optional<std::vector<torch::Tensor> > test{};
        std::ostream *stream{&std::cout};
        std::size_t buffer_vram{Core::kDefaultTrainingConfig.buffer_vram};
        GraphMode graph_mode{GraphMode::Disabled};
        // Enable CUDA graph capture/replay; pad or drop remainder batches first.
        bool enable_amp{false}; // Enable TensorCores
        torch::MemoryFormat memory_format{torch::MemoryFormat::Contiguous};
    };

    struct NvtxRange {
        explicit NvtxRange(const char *name) {
#ifdef TORCH_CUDA_AVAILABLE
            nvtxRangePushA(name);
#else
            (void) name;
#endif
        }

        ~NvtxRange() {
#ifdef TORCH_CUDA_AVAILABLE
            nvtxRangePop();
#endif
        }
    };


    class Model : public torch::nn::Module {
        using RegularizationState = Regularization::StateVariant;
        using RegularizationStateStorage = std::shared_ptr<std::vector<RegularizationState> >;
        using RegularizationAccumulator = Regularization::Accumulator;
        using CalibrationMethod = Calibration::MethodPtr;


        struct OptimizerBinding {
            std::unique_ptr<torch::optim::Optimizer> instance{};
            std::function<void(torch::optim::Optimizer &)> warmup{};
            bool capture_safe{true};
            bool warmed_up{false};

            OptimizerBinding() = default;

            OptimizerBinding(OptimizerBinding &&) noexcept = default;

            OptimizerBinding &operator=(OptimizerBinding &&) noexcept = default;

            OptimizerBinding(const OptimizerBinding &) = delete;

            OptimizerBinding &operator=(const OptimizerBinding &) = delete;
        };

        struct RegularizationBinding {
            Regularization::Descriptor descriptor{};
            RegularizationStateStorage states{};
            RegularizationAccumulator accumulator{};
        };

        enum class GraphExecutionPhase {
            Training,
            Inference
        };

        struct GraphCaptureState {
#ifdef TORCH_CUDA_AVAILABLE
            std::unique_ptr<torch::cuda::CUDAGraph> graph{};
            std::optional<torch::cuda::CUDAStream> capture_stream{};
            torch::Dict<std::string, torch::Tensor> amp_scaler_state{};
            bool amp_scaler_state_valid{false};
#endif
            bool captured{false};
            bool dirty{true};
            torch::Tensor loss_buffer{};
            torch::Tensor target_buffer{};
        };


        struct GraphTensorSignature {
            torch::Device device{torch::kCPU};
            torch::ScalarType dtype{torch::kFloat32};
            std::vector<int64_t> shape{};
        };

        struct GraphRegularizationBindingInfo {
            bool initialised{false};
            bool participates{false};
            GraphTensorSignature signature{};
        };

        struct GraphCalibrationInfo {
            bool initialised{false};
            GraphTensorSignature signature{};
        };

    public:
        struct TrainingTelemetry {
            struct DeferredScalar {
                torch::Tensor host_tensor{};
#ifdef TORCH_CUDA_AVAILABLE
                mutable std::shared_ptr<at::cuda::CUDAEvent> ready_event{};
                int device_index{-1};
#endif
                mutable std::optional<double> cached_value{};

                DeferredScalar() = default;

                static DeferredScalar from_tensor(torch::Tensor tensor, const torch::Device &device) {
                    DeferredScalar scalar{};

                    if (!tensor.defined()) {
                        return scalar;
                    }

                    tensor = tensor.detach();
                    if (tensor.scalar_type() != torch::kFloat64) {
                        tensor = tensor.to(torch::kFloat64);
                    }

#ifdef TORCH_CUDA_AVAILABLE
                    if (device.is_cuda()) {
                        const auto device_index = device.index();
                        auto stream = at::cuda::getCurrentCUDAStream(device_index);
                        auto host_copy = tensor.to(torch::kCPU, torch::kFloat64, /*non_blocking=*/true);
                        auto event = std::make_shared<at::cuda::CUDAEvent>();
                        event->record(stream);
                        scalar.host_tensor = std::move(host_copy);
                        scalar.ready_event = std::move(event);
                        scalar.device_index = device_index;
                        return scalar;
                    }
#endif

                    scalar.host_tensor = tensor.to(torch::kCPU, torch::kFloat64);
                    return scalar;
                }

                [[nodiscard]] bool defined() const noexcept { return host_tensor.defined(); }

                [[nodiscard]] bool is_ready() const {
                    if (!host_tensor.defined()) {
                        return false;
                    }

#ifdef TORCH_CUDA_AVAILABLE
                    if (ready_event) {
                        if (!ready_event->query()) {
                            return false;
                        }
                        ready_event.reset();
                    }
#endif

                    if (!cached_value) {
                        cached_value = host_tensor.item<double>();
                    }

                    return true;
                }


                double materialize() const {
                    if (!host_tensor.defined()) {
                        return 0.0;
                    }

#ifdef TORCH_CUDA_AVAILABLE
                    if (ready_event) {
                        if (!ready_event->query()) {
                            ready_event->synchronize();
                        }
                        ready_event.reset();
                    }
#endif

                    if (!cached_value) {
                        cached_value = host_tensor.item<double>();
                    }

                    return *cached_value;
                }
            };

            struct EpochSnapshot {
                std::size_t epoch_index{};
                DeferredScalar train_loss{};
                std::optional<DeferredScalar> test_loss{};
                std::optional<double> delta{};
                std::vector<double> learning_rates{};
                std::chrono::system_clock::time_point timestamp{};
                double duration_seconds{};
                DeferredScalar step_latency{};


                [[nodiscard]] double train_loss_value() const { return train_loss.materialize(); }
                [[nodiscard]] double step_latency_value() const { return step_latency.materialize(); }

                [[nodiscard]] std::optional<double> test_loss_value() const {
                    if (!test_loss) {
                        return std::nullopt;
                    }
                    return test_loss->materialize();
                }
            };

            struct DatasetLossSnapshot {
                DeferredScalar loss{};
                std::size_t sample_count{};
                std::vector<double> learning_rates{};
                std::chrono::system_clock::time_point timestamp{};
                [[nodiscard]] double loss_value() const { return loss.materialize(); }
            };

            [[nodiscard]] const std::vector<EpochSnapshot> &epochs() const noexcept { return epochs_; }

            [[nodiscard]] const std::vector<DatasetLossSnapshot> &dataset_losses() const noexcept {
                return dataset_losses_;
            }

            void clear() noexcept {
                epochs_.clear();
                dataset_losses_.clear();
            }

        private:
            friend class Model;

            void append_epoch(EpochSnapshot snapshot) {
                epochs_.push_back(std::move(snapshot));
            }

            void append_dataset_loss(DatasetLossSnapshot snapshot) {
                dataset_losses_.push_back(std::move(snapshot));
            }

            std::vector<EpochSnapshot> epochs_{};
            std::vector<DatasetLossSnapshot> dataset_losses_{};
        };

        explicit Model(std::string_view name = {}) : name_(name) {
        }

        [[nodiscard]] const TrainingTelemetry &training_telemetry() const noexcept { return telemetry_; }
        void clear_training_telemetry() noexcept { telemetry_.clear(); }

        void train(bool on = true) override {
            torch::nn::Module::train(on);
            if (on) {
                invalidate_graph_capture(GraphExecutionPhase::Training);
            } else {
                invalidate_graph_capture(GraphExecutionPhase::Inference);
            }
        }

        void eval() {
            torch::nn::Module::train(false);
            invalidate_graph_capture(GraphExecutionPhase::Inference);
        }

        [[nodiscard]] const std::string &name() const noexcept { return name_; }


        template<class PrepareBatch, class ConsumeBatch>
        bool stream_forward(torch::Tensor dataset_inputs,
                            torch::Tensor dataset_targets,
                            const StreamingOptions &options,
                            PrepareBatch &&prepare_batch,
                            ConsumeBatch &&consume_batch) {
            if (!dataset_inputs.defined()) {
                return false;
            }

            if (dataset_inputs.dim() == 0) {
                auto prepared_batch = prepare_batch(std::move(dataset_inputs), std::move(dataset_targets));
                if (!prepared_batch.has_value()) {
                    return false;
                }

                auto batch = std::move(*prepared_batch);
                if (!batch.inputs.defined()) {
                    return false;
                }

                ForwardOptions forward_options{};
                if (options.forward_chunk_size.has_value()) {
                    forward_options.max_chunk_size = options.forward_chunk_size;
                }

                auto outputs = forward(batch.inputs, std::move(forward_options));
                consume_batch(std::move(outputs), std::move(batch));
                return true;
            }

            const auto total_samples = dataset_inputs.size(0);
            if (total_samples <= 0) {
                return false;
            }

            std::size_t effective_batch_size = options.batch_size;
            if (effective_batch_size == 0) {
                effective_batch_size = static_cast<std::size_t>(total_samples);
            }

            if (effective_batch_size == 0) {
                throw std::invalid_argument("Streaming batch size must be greater than zero.");
            }

            const auto step = static_cast<std::int64_t>(effective_batch_size);
            const bool targets_match_leading = dataset_targets.defined()
                                               && dataset_targets.dim() > 0
                                               && dataset_targets.size(0) == total_samples;

            bool processed_any = false;

            for (std::int64_t offset = 0; offset < total_samples; offset += step) {
                const auto remaining = total_samples - offset;
                const auto current_batch = std::min<std::int64_t>(step, remaining);
                if (current_batch <= 0) {
                    break;
                }

                auto input_slice = dataset_inputs.narrow(0, offset, current_batch);

                torch::Tensor target_slice;
                if (dataset_targets.defined()) {
                    if (targets_match_leading) {
                        target_slice = dataset_targets.narrow(0, offset, current_batch);
                    } else {
                        target_slice = dataset_targets;
                    }
                }

                auto prepared_batch = prepare_batch(std::move(input_slice), std::move(target_slice));
                if (!prepared_batch.has_value()) {
                    continue;
                }

                auto batch = std::move(*prepared_batch);
                if (!batch.inputs.defined()) {
                    continue;
                }

                ForwardOptions forward_options{};
                if (options.forward_chunk_size.has_value()) {
                    forward_options.max_chunk_size = options.forward_chunk_size;
                }

                auto outputs = forward(batch.inputs, std::move(forward_options));
                consume_batch(std::move(outputs), std::move(batch));
                processed_any = true;
            }

            return processed_any;
        }


        using ModuleDescriptor = Common::SaveLoad::ModuleDescriptor;
        using NamedModuleDescriptor = Common::SaveLoad::NamedModuleDescriptor;

        void add(ModuleDescriptor descriptor, std::string name = {}) {
            if (regularization_configured_)
                throw std::logic_error("Cannot add modules after regularization has been configured.");

            clear_compiled_graph();

            const std::string module_name = std::move(name);
            if (!module_name.empty() && module_name_index_.find(module_name) != module_name_index_.end()) {
                throw std::invalid_argument("Module name '" + module_name + "' is already registered.");
            }

            ModuleDescriptor preserved_descriptor = descriptor;
            auto store_layer = [&](Layer::Details::RegisteredLayer registered_layer) {
                registered_layer.name = module_name;
                layers_.push_back(std::move(registered_layer));
                const auto layer_index = layers_.size() - 1;
                if (!module_name.empty()) {
                    auto &binding = module_name_index_[module_name];
                    if (!binding.has_entry()) {
                        binding.entry = layer_index;
                    }
                    binding.exit = layer_index;
                    binding.layers.push_back(layer_index);
                }
                register_layer_runtime(layers_.back());
            };
            auto register_layer = [&](auto &&concrete_descriptor) {
                using DescriptorType = std::decay_t<decltype(concrete_descriptor)>;
                auto registered = Layer::Details::build_registered_layer(
                    *this,
                    static_cast<const DescriptorType &>(concrete_descriptor),
                    next_module_index());
                store_layer(std::move(registered));
            };

            auto layer_dispatcher = Overloaded{
                [&](auto &&concrete_descriptor) {
                    register_layer(std::forward<decltype(concrete_descriptor)>(concrete_descriptor));
                }
            };

            auto sequential_block_handler = [&](Block::SequentialDescriptor sequential) {
                const bool block_declares_local_optimizer = sequential.local.optimizer.has_value();
                const bool any_layer_declares_local_optimizer = std::any_of(
                    sequential.layers.begin(),
                    sequential.layers.end(),
                    [](const Layer::Descriptor &layer_descriptor) {
                        return std::visit(
                            [](const auto &concrete_layer) {
                                return concrete_layer.local.optimizer.has_value();
                            }, layer_descriptor);
                    });

                if (block_declares_local_optimizer && !any_layer_declares_local_optimizer) {
                    const auto index = next_module_index();
                    auto module = register_module(
                        "sequential_block_" + std::to_string(index),
                        Block::Details::SequentialBlockModule(std::move(sequential.layers)));

                    Layer::Details::RegisteredLayer registered_layer{};
                    registered_layer.activation = Activation::Type::Identity;
                    registered_layer.module = Layer::Details::to_shared_module_ptr(module);
                    registered_layer.local = std::move(sequential.local);
                    registered_layer.bind_module_forward(module.get());


                    store_layer(std::move(registered_layer));
                } else {
                    if (block_declares_local_optimizer) {
                        for (auto &layer: sequential.layers) {
                            std::visit(
                                [&](auto &concrete_layer) {
                                    if (!concrete_layer.local.optimizer.has_value()) {
                                        concrete_layer.local = sequential.local;
                                    }
                                },
                                layer);
                        }
                    }
                    for (auto &layer: sequential.layers) {
                        std::visit(layer_dispatcher, std::move(layer));
                    }
                }
            };

            auto residual_block_handler = [&](Block::ResidualDescriptor residual) {
                auto residual_local = residual.local;
                const auto index = next_module_index();
                auto module = register_module(
                    "residual_block_" + std::to_string(index),
                    Block::Details::ResidualBlock(std::move(residual)));

                module->set_preferred_tensor_memory_format(preferred_tensor_memory_format());

                Layer::Details::RegisteredLayer registered_layer{};
                registered_layer.activation = Activation::Type::Identity;
                registered_layer.module = Layer::Details::to_shared_module_ptr(module);
                registered_layer.local = std::move(residual_local);
                registered_layer.bind_module_forward(module.get());

                store_layer(std::move(registered_layer));
            };

            auto transformer_block_handler = Overloaded{
                [&](Block::Transformer::Classic::EncoderDescriptor encoder_descriptor) {
                    const auto index = next_module_index();
                    auto module = register_module(
                        "transformer_encoder_" + std::to_string(index),
                        Block::Transformer::Classic::TransformerEncoder(std::move(encoder_descriptor)));

                    Layer::Details::RegisteredLayer registered_layer{};
                    registered_layer.activation = Activation::Type::Identity;
                    registered_layer.module = Layer::Details::to_shared_module_ptr(module);
                    registered_layer.bind_module_forward(module.get());

                    store_layer(std::move(registered_layer));
                },
                [&](Block::Transformer::Classic::DecoderDescriptor decoder_descriptor) {
                    const auto index = next_module_index();
                    auto module = register_module(
                        "transformer_decoder_" + std::to_string(index),
                        Block::Transformer::Classic::TransformerDecoder(std::move(decoder_descriptor)));

                    Layer::Details::RegisteredLayer registered_layer{};
                    registered_layer.activation = Activation::Type::Identity;
                    registered_layer.module = Layer::Details::to_shared_module_ptr(module);
                    struct TransformerDecoderForward {
                        decltype(module.get()) module_ptr;

                        torch::Tensor operator()(torch::Tensor input) const {
                            return module_ptr->forward(std::move(input), torch::Tensor{});
                        }
                    };
                    registered_layer.bind_inline_forward(TransformerDecoderForward{module.get()});

                    store_layer(std::move(registered_layer));
                },
                [&](Block::Transformer::EBT::EncoderDescriptor encoder_descriptor) {
                    const auto index = next_module_index();
                    auto module = register_module(
                        "ebt_encoder_" + std::to_string(index),
                        Block::Transformer::EBT::EncoderModule(std::move(encoder_descriptor)));

                    Layer::Details::RegisteredLayer registered_layer{};
                    registered_layer.activation = Activation::Type::Identity;
                    registered_layer.module = Layer::Details::to_shared_module_ptr(module);
                    registered_layer.bind_module_forward(module.get());

                    store_layer(std::move(registered_layer));
                },
                [&](Block::Transformer::EBT::DecoderDescriptor decoder_descriptor) {
                    const auto index = next_module_index();
                    auto module = register_module(
                        "ebt_decoder_" + std::to_string(index),
                        Block::Transformer::EBT::DecoderModule(std::move(decoder_descriptor)));

                    Layer::Details::RegisteredLayer registered_layer{};
                    registered_layer.activation = Activation::Type::Identity;
                    registered_layer.module = Layer::Details::to_shared_module_ptr(module);
                    struct EBTDecoderForward {
                        decltype(module.get()) module_ptr;

                        torch::Tensor operator()(torch::Tensor input) const {
                            return module_ptr->forward(std::move(input), torch::Tensor{});
                        }
                    };
                    registered_layer.bind_inline_forward(EBTDecoderForward{module.get()});

                    store_layer(std::move(registered_layer));
                },
                [&](Block::Transformer::PlusPlus::EncoderDescriptor encoder_descriptor) {
                    const auto index = next_module_index();
                    auto module = register_module(
                        "transformer_pp_encoder_" + std::to_string(index),
                        Block::Transformer::PlusPlus::TransformerPlusPlusEncoder(std::move(encoder_descriptor)));

                    Layer::Details::RegisteredLayer registered_layer{};
                    registered_layer.activation = Activation::Type::Identity;
                    registered_layer.module = Layer::Details::to_shared_module_ptr(module);
                    registered_layer.bind_module_forward(module.get());

                    layers_.push_back(std::move(registered_layer));
                    register_layer_runtime(layers_.back());
                },
                [&](Block::Transformer::PlusPlus::DecoderDescriptor decoder_descriptor) {
                    const auto index = next_module_index();
                    auto module = register_module(
                        "transformer_pp_decoder_" + std::to_string(index),
                        Block::Transformer::PlusPlus::TransformerPlusPlusDecoder(std::move(decoder_descriptor)));

                    Layer::Details::RegisteredLayer registered_layer{};
                    registered_layer.activation = Activation::Type::Identity;
                    registered_layer.module = Layer::Details::to_shared_module_ptr(module);
                    struct TransformerDecoderForward {
                        decltype(module.get()) module_ptr;

                        torch::Tensor operator()(torch::Tensor input) const {
                            auto result = module_ptr->forward(std::move(input), torch::Tensor{});
                            return std::move(result.main);
                        }
                    };
                    registered_layer.bind_inline_forward(TransformerDecoderForward{module.get()});

                    store_layer(std::move(registered_layer));
                },
                [&](Block::Transformer::Mamba::EncoderDescriptor encoder_descriptor) {
                    const auto index = next_module_index();
                    auto module = register_module(
                        "mamba_encoder_" + std::to_string(index),
                        Block::Transformer::Mamba::EncoderModule(std::move(encoder_descriptor)));

                    Layer::Details::RegisteredLayer registered_layer{};
                    registered_layer.activation = Activation::Type::Identity;
                    registered_layer.module = Layer::Details::to_shared_module_ptr(module);
                    registered_layer.bind_module_forward(module.get());


                    store_layer(std::move(registered_layer));
                },
                [&](Block::Transformer::Vision::EncoderDescriptor encoder_descriptor) {
                    const auto index = next_module_index();
                    auto module = register_module(
                        "vision_transformer_" + std::to_string(index),
                        Block::Transformer::Vision::VisionEncoder(std::move(encoder_descriptor)));

                    Layer::Details::RegisteredLayer registered_layer{};
                    registered_layer.activation = Activation::Type::Identity;
                    registered_layer.module = Layer::Details::to_shared_module_ptr(module);
                    registered_layer.bind_module_forward(module.get());

                    store_layer(std::move(registered_layer));
                },
                [&](Block::Transformer::Perceiver::EncoderDescriptor encoder_descriptor) {
                    const auto index = next_module_index();
                    auto module = register_module(
                        "perceiver_encoder_" + std::to_string(index),
                        Block::Transformer::Perceiver::PerceiverEncoder(std::move(encoder_descriptor)));

                    Layer::Details::RegisteredLayer registered_layer{};
                    registered_layer.activation = Activation::Type::Identity;
                    registered_layer.module = Layer::Details::to_shared_module_ptr(module);
                    registered_layer.bind_module_forward(module.get());

                    store_layer(std::move(registered_layer));
                },
                [&](Block::Transformer::LongformerXL::EncoderDescriptor encoder_descriptor) {
                    const auto index = next_module_index();
                    auto module = register_module(
                        "longformer_xl_encoder_" + std::to_string(index),
                        Block::Transformer::LongformerXL::LongformerEncoder(std::move(encoder_descriptor)));

                    Layer::Details::RegisteredLayer registered_layer{};
                    registered_layer.activation = Activation::Type::Identity;
                    registered_layer.module = Layer::Details::to_shared_module_ptr(module);
                    registered_layer.bind_module_forward(module.get());
                    store_layer(std::move(registered_layer));
                },
                [&](Block::Transformer::Bert::EncoderDescriptor encoder_descriptor) {
                    const auto index = next_module_index();
                    auto module = register_module(
                        "bert_encoder_" + std::to_string(index),
                        Block::Transformer::Bert::BertEncoder(std::move(encoder_descriptor)));

                    Layer::Details::RegisteredLayer registered_layer{};
                    registered_layer.activation = Activation::Type::Identity;
                    registered_layer.module = Layer::Details::to_shared_module_ptr(module);
                    registered_layer.bind_module_forward(module.get());
                    store_layer(std::move(registered_layer));
                }
            };

            auto block_dispatcher = Overloaded{
                [&](Block::SequentialDescriptor sequential) {
                    sequential_block_handler(std::move(sequential));
                },
                [&](Block::ResidualDescriptor residual) {
                    residual_block_handler(std::move(residual));
                },
                [&](Block::Transformer::Classic::EncoderDescriptor encoder_descriptor) {
                    transformer_block_handler(std::move(encoder_descriptor));
                },
                [&](Block::Transformer::Classic::DecoderDescriptor decoder_descriptor) {
                    transformer_block_handler(decoder_descriptor);
                },
                [&](Block::Transformer::EBT::EncoderDescriptor encoder_descriptor) {
                    transformer_block_handler(std::move(encoder_descriptor));
                },
                [&](Block::Transformer::EBT::DecoderDescriptor decoder_descriptor) {
                    transformer_block_handler(decoder_descriptor);
                },
                [&](Block::Transformer::PlusPlus::EncoderDescriptor encoder_descriptor) {
                    transformer_block_handler(std::move(encoder_descriptor));
                },
                [&](Block::Transformer::PlusPlus::DecoderDescriptor decoder_descriptor) {
                    transformer_block_handler(std::move(decoder_descriptor));
                },
                [&](Block::Transformer::Mamba::EncoderDescriptor encoder_descriptor) {
                    transformer_block_handler(std::move(encoder_descriptor));
                },
                [&](Block::Transformer::Vision::EncoderDescriptor encoder_descriptor) {
                    transformer_block_handler(std::move(encoder_descriptor));
                },
                [&](Block::Transformer::Perceiver::EncoderDescriptor encoder_descriptor) {
                    transformer_block_handler(std::move(encoder_descriptor));
                },
                [&](Block::Transformer::LongformerXL::EncoderDescriptor encoder_descriptor) {
                    transformer_block_handler(std::move(encoder_descriptor));
                },
                [&](Block::Transformer::Bert::EncoderDescriptor encoder_descriptor) {
                    transformer_block_handler(std::move(encoder_descriptor));
                }
            };

            auto module_dispatcher = Overloaded{
                [&](Layer::Descriptor layer_descriptor) {
                    std::visit(layer_dispatcher, std::move(layer_descriptor));
                },
                [&](Block::Descriptor block_descriptor) {
                    std::visit(block_dispatcher, std::move(block_descriptor));
                }
            };

            std::visit(module_dispatcher, std::move(descriptor));
            module_descriptors_.emplace_back(std::move(preserved_descriptor), module_name);
        }

        struct LinkParams {
            std::unordered_map<std::string, std::size_t> inputs{}; // alias -> input index
            std::unordered_map<std::string, std::size_t> outputs{}; // alias -> output index
            bool enable_graph_capture{false};
        };

        void links(std::vector<LinkSpec> specifications, bool enable_graph_capture) {
            LinkParams params{};
            params.enable_graph_capture = enable_graph_capture;
            links(std::move(specifications), std::move(params));
        }

        // Updated multi-IO + params form
        void links(std::vector<LinkSpec> specifications, LinkParams params) {
            graph_capture_opt_in_ = params.enable_graph_capture && !specifications.empty();
            invalidate_graph_captures();
            if (specifications.empty()) {
                clear_compiled_graph();
                return;
            }

            auto max_index_from = [](const std::unordered_map<std::string, std::size_t> &m)-> std::size_t {
                std::size_t mx = 0;
                for (auto &kv: m) mx = std::max(mx, kv.second);
                return m.empty() ? 0 : mx;
            };
            const std::size_t num_inputs = std::max<std::size_t>(1, max_index_from(params.inputs) + 1);
            const std::size_t num_outputs = std::max<std::size_t>(1, max_index_from(params.outputs) + 1);

            std::vector<CompiledNode> nodes;
            std::vector<CompiledStep> steps;
            std::vector<JoinBuffer> joins;
            std::vector<LinkSpec> resolved_links;

            nodes.reserve(layers_.size() + specifications.size() * 2 + 2 + num_inputs + num_outputs);
            resolved_links.reserve(specifications.size());

            // Build N input nodes (@input, @input[1], ...)
            std::vector<std::size_t> input_node_indices(num_inputs, std::numeric_limits<std::size_t>::max());
            for (std::size_t i = 0; i < num_inputs; ++i) {
                CompiledNode n{};
                n.kind = CompiledNode::Kind::Input;
                n.label = "@input";
                if (num_inputs > 1) { n.label += "[" + std::to_string(i) + "]"; }
                nodes.push_back(std::move(n));
                input_node_indices[i] = nodes.size() - 1;
            }

            // Modules
            std::vector<std::size_t> module_node_indices(layers_.size(), std::numeric_limits<std::size_t>::max());
            for (std::size_t index = 0; index < layers_.size(); ++index) {
                CompiledNode node{};
                node.kind = CompiledNode::Kind::Module;
                node.index = index;
                const auto &layer = layers_[index];
                if (!layer.name.empty()) {
                    node.label = layer.name;
                    auto binding = module_name_index_.find(layer.name);
                    if (binding != module_name_index_.end() && binding->second.layers.size() > 1) {
                        const auto &sequence = binding->second.layers;
                        auto position = std::find(sequence.begin(), sequence.end(), index);
                        if (position != sequence.end()) {
                            node.label.append("[").append(std::to_string(std::distance(sequence.begin(), position))).
                                    append("]");
                        }
                    }
                } else {
                    node.label = "#" + std::to_string(index);
                }
                nodes.push_back(std::move(node));
                module_node_indices[index] = nodes.size() - 1;
            }

            // Outputs created lazily; track per-index
            std::vector<std::size_t> output_node_indices(num_outputs, std::numeric_limits<std::size_t>::max());
            auto ensure_output_node_k = [&](std::size_t k) -> std::size_t {
                if (k >= num_outputs) {
                    throw std::invalid_argument("Requested output index " + std::to_string(k) +
                                                " but num_outputs=" + std::to_string(num_outputs) + ".");
                }
                auto &slot = output_node_indices[k];
                if (slot != std::numeric_limits<std::size_t>::max()) return slot;
                CompiledNode n{};
                n.kind = CompiledNode::Kind::Output;
                n.label = "@output";
                if (num_outputs > 1) { n.label += "[" + std::to_string(k) + "]"; }
                nodes.push_back(std::move(n));
                slot = nodes.size() - 1;
                return slot;
            };

            std::unordered_map<std::string, std::size_t> join_lookup{};
            std::unordered_map<std::size_t, std::size_t> join_buffer_lookup{};

            auto parse_concat_dimension = [](const Port &port) -> std::optional<int64_t> {
                if (port.join_dimension) return port.join_dimension;
                if (port.attribute.empty()) return std::nullopt;
                std::string token;
                token.reserve(port.attribute.size());
                for (char ch: port.attribute) if (!std::isspace(static_cast<unsigned char>(ch))) token.push_back(ch);
                auto strip_prefix = [](std::string &v, std::string_view p) {
                    if (v.rfind(p, 0) == 0) v.erase(0, p.size());
                };
                strip_prefix(token, "dim=");
                strip_prefix(token, "axis=");
                if (token.empty()) throw std::invalid_argument(
                    "Join port '" + port.describe() + "' specifies an empty concat dimension.");
                try { return std::stoll(token); } catch (...) {
                    throw std::invalid_argument(
                        "Join port '" + port.describe() + "' specifies an invalid concat dimension '" + token + "'.");
                }
            };

            auto ensure_join_node = [&](const Port &port) -> std::size_t {
                const auto key = port.storage_key();
                if (auto it = join_lookup.find(key); it != join_lookup.end()) {
                    const auto node_index = it->second;
                    const auto buffer_index = join_buffer_lookup.at(node_index);
                    auto &buffer = joins[buffer_index];
                    if (buffer.policy != port.merge_policy)
                        throw std::invalid_argument(
                            "Join node '" + port.describe() + "' requested with conflicting merge policies.");
                    if (buffer.policy == MergePolicy::Stack) {
                        auto requested = parse_concat_dimension(port);
                        if (requested) {
                            if (buffer.concat_dimension && *requested != *buffer.concat_dimension)
                                throw std::invalid_argument(
                                    "Join node '" + port.describe() +
                                    "' requested with conflicting concat dimensions.");
                            if (!buffer.concat_dimension) buffer.concat_dimension = requested;
                        }
                    }
                    return node_index;
                }
                CompiledNode node{};
                node.kind = CompiledNode::Kind::Join;
                node.merge = port.merge_policy;
                node.index = joins.size();
                node.label = port.describe();
                nodes.push_back(node);
                const std::size_t node_index = nodes.size() - 1;

                JoinBuffer buffer{};
                buffer.node_index = node_index;
                buffer.policy = port.merge_policy;
                if (buffer.policy == MergePolicy::Stack) buffer.concat_dimension = parse_concat_dimension(port);
                join_buffer_lookup[node_index] = joins.size();
                joins.push_back(std::move(buffer));
                join_lookup.emplace(key, node_index);
                return node_index;
            };

            auto parse_numeric_identifier = [](std::string_view token) -> std::optional<std::size_t> {
                if (token.empty()) return std::nullopt;
                if (token.front() == '#') token.remove_prefix(1);
                if (token.empty()) return std::nullopt;
                std::size_t value = 0;
                for (char c: token) {
                    if (!std::isdigit(static_cast<unsigned char>(c))) return std::nullopt;
                    value = value * 10 + static_cast<std::size_t>(c - '0');
                }
                return value;
            };

            enum class PortRole { Source, Target };

            auto resolve_module = [&](const Port &port, PortRole role) -> std::size_t {
                if (port.identifier.empty()) {
                    throw std::invalid_argument("Module port '" + port.describe() + "' is missing an identifier.");
                }
                std::optional<std::size_t> module_index{};
                if (auto by_name = module_name_index_.find(port.identifier); by_name != module_name_index_.end()) {
                    const auto &binding = by_name->second;
                    if (!binding.has_entry())
                        throw std::invalid_argument(
                            "Module port '" + port.describe() + "' references an unregistered module name.");
                    if (role == PortRole::Source) {
                        if (binding.exit == std::numeric_limits<std::size_t>::max())
                            throw std::invalid_argument(
                                "Module port '" + port.describe() + "' could not resolve an output endpoint.");
                        module_index = binding.exit;
                    } else {
                        if (binding.entry == std::numeric_limits<std::size_t>::max())
                            throw std::invalid_argument(
                                "Module port '" + port.describe() + "' could not resolve an input endpoint.");
                        module_index = binding.entry;
                    }
                } else {
                    module_index = parse_numeric_identifier(port.identifier);
                }
                if (!module_index.has_value() || *module_index >= module_node_indices.size())
                    throw std::invalid_argument("Unknown module referenced by port '" + port.describe() + "'.");
                const auto node_index = module_node_indices[*module_index];
                if (node_index == std::numeric_limits<std::size_t>::max())
                    throw std::invalid_argument("Module port '" + port.describe() + "' could not be resolved.");
                return node_index;
            };

            auto trim_view = [](std::string_view token) {
                while (!token.empty() && std::isspace(static_cast<unsigned char>(token.front()))) token.
                        remove_prefix(1);
                while (!token.empty() && std::isspace(static_cast<unsigned char>(token.back()))) token.remove_suffix(1);
                return token;
            };

            auto iequals = [](std::string_view a, std::string_view b) {
                if (a.size() != b.size()) return false;
                for (size_t i = 0; i < a.size(); ++i) if (
                    std::tolower((unsigned char) a[i]) != std::tolower((unsigned char) b[i])) return false;
                return true;
            };

            // Multi-IO
            auto resolve_port = [&](Port &port, PortRole role) -> std::size_t {
                switch (port.kind) {
                    case Port::Kind::Input: {
                        // default single input semantics
                        if (port.identifier.empty() || iequals(port.identifier, "@input")) {
                            port.assign_node(input_node_indices.front());
                            return input_node_indices.front();
                        }
                        // numeric?
                        if (auto n = parse_numeric_identifier(port.identifier)) {
                            if (*n >= input_node_indices.size())
                                throw std::invalid_argument("Input index " + std::to_string(*n) + " out of range.");
                            port.assign_node(input_node_indices[*n]);
                            return input_node_indices[*n];
                        }
                        // alias?
                        if (auto it = params.inputs.find(port.identifier); it != params.inputs.end()) {
                            const auto idx = it->second;
                            if (idx >= input_node_indices.size())
                                throw std::invalid_argument(
                                    "Input alias '" + port.identifier + "' mapped to out-of-range index.");
                            port.assign_node(input_node_indices[idx]);
                            return input_node_indices[idx];
                        }
                        throw std::invalid_argument(
                            "Unknown input '" + port.identifier + "'; use @input, #k or alias.");
                    }
                    case Port::Kind::Output: {
                        // default single output
                        if (port.identifier.empty() || iequals(port.identifier, "@output")) {
                            const auto idx = ensure_output_node_k(0);
                            port.assign_node(idx);
                            return idx;
                        }
                        // numeric?
                        if (auto n = parse_numeric_identifier(port.identifier)) {
                            const auto idx = ensure_output_node_k(*n);
                            port.assign_node(idx);
                            return idx;
                        }
                        // alias?
                        if (auto it = params.outputs.find(port.identifier); it != params.outputs.end()) {
                            const auto idx = ensure_output_node_k(it->second);
                            port.assign_node(idx);
                            return idx;
                        }
                        throw std::invalid_argument(
                            "Unknown output '" + port.identifier + "'; use @output, #k or alias.");
                    }
                    case Port::Kind::Module: {
                        const auto idx = resolve_module(port, role);
                        port.assign_node(idx);
                        return idx;
                    }
                    case Port::Kind::Join: {
                        const auto idx = ensure_join_node(port);
                        port.assign_node(idx);
                        if (auto it = join_buffer_lookup.find(idx); it != join_buffer_lookup.end())
                            port.assign_join(it->second);
                        return idx;
                    }
                }
                throw std::invalid_argument("Unsupported port kind encountered while resolving links.");
            };

            std::unordered_map<std::size_t, std::size_t> consumer_inbound{};

            // track explicit join edges to avoid duplicate inferred edges
            std::unordered_set<std::string> auto_link_keys{};
            auto record_join_edge = [&](const LinkSpec &spec) {
                if (!spec.target.is_join()) return;
                const auto key = spec.source.storage_key() + "->" + spec.target.storage_key();
                auto_link_keys.insert(key);
            };
            for (const auto &s: specifications) record_join_edge(s);

            std::vector<LinkSpec> inferred_links;
            inferred_links.reserve(specifications.size());
            auto schedule_join_members = [&](const Port &port) {
                if (!port.is_join() || port.join_members.empty()) return;
                for (const auto &member: port.join_members) {
                    auto module_port = Port::Module(member);
                    auto join_port = port;
                    join_port.node_index.reset();
                    join_port.join_index.reset();
                    const auto key = module_port.storage_key() + "->" + join_port.storage_key();
                    if (auto_link_keys.insert(key).second) {
                        inferred_links.emplace_back(std::move(module_port), std::move(join_port));
                    }
                }
            };
            for (const auto &s: specifications) {
                schedule_join_members(s.source);
                schedule_join_members(s.target);
            }
            specifications.insert(specifications.end(),
                                  std::make_move_iterator(inferred_links.begin()),
                                  std::make_move_iterator(inferred_links.end()));

            // Apply links
            for (auto &specification: specifications) {
                auto link = specification;
                const auto source_index = resolve_port(link.source, PortRole::Source);
                const auto target_index = resolve_port(link.target, PortRole::Target);

                const auto source_kind = nodes[source_index].kind;
                const auto target_kind = nodes[target_index].kind;

                if (source_kind == CompiledNode::Kind::Output)
                    throw std::invalid_argument(
                        "Output port '" + link.source.describe() + "' cannot be used as a source.");
                if (target_kind == CompiledNode::Kind::Input)
                    throw std::invalid_argument(
                        "Input port '" + link.target.describe() + "' cannot be used as a target.");

                if (target_kind != CompiledNode::Kind::Join) {
                    auto &inbound = consumer_inbound[target_index];
                    if (inbound > 0)
                        throw std::invalid_argument(
                            "Consumer port '" + link.target.describe() + "' already has a producer.");
                    ++inbound;
                }

                nodes[source_index].outputs.push_back(target_index);
                nodes[target_index].inputs.push_back(source_index);

                if (target_kind == CompiledNode::Kind::Join) {
                    const auto buffer_index = join_buffer_lookup.at(target_index);
                    auto &buffer = joins[buffer_index];
                    if (std::find(buffer.producers.begin(), buffer.producers.end(), source_index) != buffer.producers.
                        end())
                        throw std::invalid_argument(
                            "Join node '" + link.target.describe() + "' already receives input from '"
                            + link.source.describe() + "'.");
                    buffer.producers.push_back(source_index);
                }
                resolved_links.push_back(std::move(link));
            }


            for (const auto &[name, binding]: module_name_index_) {
                if (binding.layers.size() < 2) continue;
                for (std::size_t offset = 1; offset < binding.layers.size(); ++offset) {
                    const auto upstream_layer = binding.layers[offset - 1];
                    const auto downstream_layer = binding.layers[offset];
                    if (upstream_layer >= module_node_indices.size() || downstream_layer >= module_node_indices.size())
                        throw std::invalid_argument(
                            "Module name '" + name + "' is out of sync with registered layers.");
                    const auto upstream_node = module_node_indices[upstream_layer];
                    const auto downstream_node = module_node_indices[downstream_layer];
                    if (upstream_node >= nodes.size() || downstream_node >= nodes.size())
                        throw std::invalid_argument("Module name '" + name + "' resolved to an invalid node index.");

                    auto &upstream_outputs = nodes[upstream_node].outputs;
                    if (std::find(upstream_outputs.begin(), upstream_outputs.end(), downstream_node) != upstream_outputs
                        .end()) continue;

                    auto &downstream_inputs = nodes[downstream_node].inputs;
                    if (!downstream_inputs.empty())
                        throw std::invalid_argument("Module node '" + nodes[downstream_node].label
                                                    + "' already has a producer; unable to auto-link sequential block '"
                                                    + name + "'.");
                    auto &inbound = consumer_inbound[downstream_node];
                    if (inbound > 0)
                        throw std::invalid_argument("Module node '" + nodes[downstream_node].label
                                                    + "' already has a producer; unable to auto-link sequential block '"
                                                    + name + "'.");
                    ++inbound;
                    upstream_outputs.push_back(downstream_node);
                    downstream_inputs.push_back(upstream_node);
                }
            }

            for (const auto &join: joins) {
                const auto &node = nodes[join.node_index];
                if (join.producers.empty()) throw std::invalid_argument(
                    "Join node '" + node.label + "' has no producers.");
                if (node.outputs.empty()) throw std::invalid_argument(
                    "Join node '" + node.label + "' has no consumers.");
            }
            for (const auto &node: nodes) {
                switch (node.kind) {
                    case CompiledNode::Kind::Input: break; // multiple inputs allowed
                    case CompiledNode::Kind::Module: if (node.inputs.empty())
                            throw std::invalid_argument(
                                "Module node '" + node.label + "' has no inbound links in the routing graph.");
                        break;
                    case CompiledNode::Kind::Join: break;
                    case CompiledNode::Kind::Output: if (node.inputs.empty())
                            throw std::invalid_argument("Output node has no inbound links in the routing graph.");
                        break;
                }
            }

            // Toposort
            std::vector<std::size_t> indegree(nodes.size(), 0);
            for (std::size_t ni = 0; ni < nodes.size(); ++ni)
                for (auto t: nodes[ni].outputs) {
                    if (t >= indegree.size()) throw std::invalid_argument("Invalid node index in link.");
                    ++indegree[t];
                }
            std::deque<std::size_t> queue;
            for (std::size_t ni = 0; ni < nodes.size(); ++ni) if (indegree[ni] == 0) queue.push_back(ni);

            steps.reserve(nodes.size());
            std::size_t visited = 0;
            while (!queue.empty()) {
                const auto node_index = queue.front();
                queue.pop_front();
                ++visited;
                if (nodes[node_index].kind != CompiledNode::Kind::Input) {
                    // changed
                    CompiledStep step{};
                    step.node_index = node_index;
                    step.dependencies = nodes[node_index].inputs;
                    steps.push_back(std::move(step));
                }
                for (auto t: nodes[node_index].outputs) {
                    auto &d = indegree[t];
                    if (d == 0) continue;
                    if (--d == 0) queue.push_back(t);
                }
            }
            if (visited != nodes.size())
                throw std::invalid_argument("Link specification contains cycles; unable to compile routing graph.");

            // Emit execution steps
            std::vector<ExecutionStep> execution_steps;
            execution_steps.reserve(steps.size());
            for (const auto &step: steps) {
                const auto node_index = step.node_index;
                const auto &node = nodes[node_index];

                ExecutionStep execution{};
                execution.activation_index = node_index;

                switch (node.kind) {
                    case CompiledNode::Kind::Input: continue;
                    case CompiledNode::Kind::Module: {
                        if (node.index >= layers_.size())
                            throw std::invalid_argument(
                                "Module node '" + node.label + "' references an invalid layer index.");
                        if (step.dependencies.size() != 1)
                            throw std::invalid_argument(
                                "Module node '" + node.label + "' must have exactly one dependency.");
                        execution.kind = ExecutionStep::Kind::Module;
                        execution.module.layer = &layers_[node.index];
                        execution.module.input_index = step.dependencies.front();
                        break;
                    }
                    case CompiledNode::Kind::Join: {
                        if (node.index >= joins.size())
                            throw std::invalid_argument(
                                "Join node '" + node.label + "' references an invalid join buffer index.");
                        const auto &buffer = joins[node.index];
                        if (buffer.producers.empty())
                            throw std::invalid_argument(
                                "Join node '" + node.label + "' has no producers after compilation.");
                        execution.kind = ExecutionStep::Kind::Join;
                        execution.join.policy = buffer.policy;
                        execution.join.producers = buffer.producers;
                        execution.join.workspace_index = node.index;
                        execution.join.concat_dimension = buffer.concat_dimension;
                        break;
                    }
                    case CompiledNode::Kind::Output: {
                        if (step.dependencies.size() != 1)
                            throw std::invalid_argument("Output node must have exactly one dependency.");
                        execution.kind = ExecutionStep::Kind::Output;
                        execution.output.input_index = step.dependencies.front();
                        break;
                    }
                }
                execution_steps.push_back(std::move(execution));
            }

            // Decide final terminal: if multiple distinct Output nodes referenced, auto-stack to keep single terminal
            std::vector<std::size_t> referenced_outputs;
            for (std::size_t k = 0; k < output_node_indices.size(); ++k)
                if (output_node_indices[k] != std::numeric_limits<std::size_t>::max()) referenced_outputs.push_back(
                    output_node_indices[k]);

            std::optional<std::size_t> output_node_index{};
            if (referenced_outputs.size() == 1) {
                output_node_index = referenced_outputs.front();
            } else if (referenced_outputs.size() > 1) {
                // hidden stack join on dim=1
                JoinBuffer buf{};
                buf.policy = MergePolicy::Stack;
                buf.concat_dimension = static_cast<int64_t>(1);
                joins.push_back(std::move(buf));

                CompiledNode jn{};
                jn.kind = CompiledNode::Kind::Join;
                jn.label = "@auto/output_stack";
                nodes.push_back(std::move(jn));
                const auto join_idx = nodes.size() - 1;
                const auto buf_idx = joins.size() - 1;
                join_lookup.emplace(nodes[join_idx].label, join_idx);
                join_buffer_lookup.emplace(join_idx, buf_idx);

                for (auto out_node: referenced_outputs) {
                    nodes[out_node].outputs.push_back(join_idx);
                    nodes[join_idx].inputs.push_back(out_node);
                }

                CompiledNode out{};
                out.kind = CompiledNode::Kind::Output;
                out.label = "@output";
                nodes.push_back(std::move(out));
                const auto out_idx = nodes.size() - 1;
                nodes[join_idx].outputs.push_back(out_idx);
                nodes[out_idx].inputs.push_back(join_idx);
                output_node_index = out_idx;

                {
                    // Join exec
                    ExecutionStep exJ{};
                    exJ.activation_index = join_idx;
                    exJ.kind = ExecutionStep::Kind::Join;
                    exJ.join.policy = MergePolicy::Stack;
                    exJ.join.producers = referenced_outputs;
                    exJ.join.workspace_index = buf_idx;
                    exJ.join.concat_dimension = static_cast<int64_t>(1);
                    execution_steps.push_back(std::move(exJ));
                    // Output exec
                    ExecutionStep exO{};
                    exO.activation_index = out_idx;
                    exO.kind = ExecutionStep::Kind::Output;
                    exO.output.input_index = join_idx;
                    execution_steps.push_back(std::move(exO));
                }
            } else {
                const std::size_t fallback = steps.empty() ? input_node_indices.front() : steps.back().node_index;
                output_node_index = fallback;
            }

            compiled_nodes_ = std::move(nodes);
            compiled_steps_ = std::move(steps);
            execution_steps_ = std::move(execution_steps);
            join_buffers_ = std::move(joins);
            compiled_links_ = std::move(resolved_links);
            compiled_output_node_index_ = output_node_index;
            routing_active_ = true;
            invalidate_execution_workspace();
        }


        [[nodiscard]] bool has_compiled_routing() const noexcept { return routing_active_; }
        [[nodiscard]] const std::vector<CompiledNode> &compiled_nodes() const noexcept { return compiled_nodes_; }
        [[nodiscard]] const std::vector<CompiledStep> &compiled_steps() const noexcept { return compiled_steps_; }
        [[nodiscard]] const std::vector<JoinBuffer> &join_buffers() const noexcept { return join_buffers_; }
        [[nodiscard]] const std::vector<LinkSpec> &compiled_links() const noexcept { return compiled_links_; }


        void set_optimizer(Optimizer::Descriptor descriptor,
                           std::optional<LrScheduler::Descriptor> scheduler = std::nullopt) {
            if (layers_.empty()) {
                throw std::logic_error("Cannot create optimizer before any layer has been registered.");
            }
            refresh_layer_parameter_cache();

            auto build_optimizer_for = [](const Optimizer::Descriptor &config,
                                          std::vector<torch::Tensor> parameters,
                                          std::vector<std::vector<torch::Tensor> > warmup_buckets) {
                return std::visit(
                    [&](const auto &concrete_descriptor) -> OptimizerBinding {
                        using DescriptorType = std::decay_t<decltype(concrete_descriptor)>;
                        OptimizerBinding binding{};
                        if constexpr (std::is_same_v<DescriptorType, Optimizer::Details::SGDDescriptor>) {
                            auto options = Optimizer::Details::to_torch_options(concrete_descriptor.options);
                            binding.instance = std::make_unique<Optimizer::Details::SGD>(
                                std::move(parameters), options, warmup_buckets);
                            binding.capture_safe = true;
                            binding.warmup = [](torch::optim::Optimizer &optimizer) {
                                if (auto *sgd = dynamic_cast<Optimizer::Details::SGD *>(&optimizer)) {
                                    sgd->ensure_state_initialized();
                                }
                            };
                        } else if constexpr (std::is_same_v<DescriptorType, Optimizer::Details::AdamDescriptor>) {
                            auto options = Optimizer::Details::to_torch_options(concrete_descriptor.options);
                            binding.instance = std::make_unique<Optimizer::Details::Adam>(
                                std::move(parameters), options, warmup_buckets);
                            binding.capture_safe = true;
                            binding.warmup = [](torch::optim::Optimizer &optimizer) {
                                if (auto *a = dynamic_cast<Optimizer::Details::Adam *>(&optimizer)) {
                                    a->ensure_state_initialized();
                                }
                            };
                        } else if constexpr (std::is_same_v<DescriptorType, Optimizer::Details::AdamWDescriptor>) {
                            auto options = Optimizer::Details::to_torch_options(concrete_descriptor.options);
                            binding.instance = std::make_unique<Optimizer::Details::AdamW>(
                                std::move(parameters), options, warmup_buckets);
                            binding.capture_safe = true;
                            binding.warmup = [](torch::optim::Optimizer &optimizer) {
                                if (auto *aw = dynamic_cast<Optimizer::Details::AdamW *>(&optimizer)) {
                                    aw->ensure_state_initialized();
                                }
                            };
                        } else if constexpr (std::is_same_v<DescriptorType, Optimizer::Details::SophiaGDescriptor>) {
                            binding.instance = std::make_unique<Optimizer::Details::SophiaG>(
                                std::move(parameters), concrete_descriptor.options);
                            binding.capture_safe = true;
                            binding.warmup = [](torch::optim::Optimizer &optimizer) {
                                if (auto *sophia = dynamic_cast<Optimizer::Details::SophiaG *>(&optimizer)) {
                                    sophia->ensure_state_initialized();
                                }
                            };
                        } else if constexpr (std::is_same_v<DescriptorType, Optimizer::Details::SophiaHDescriptor>) {
                            binding.instance = std::make_unique<Optimizer::Details::SophiaH>(
                                std::move(parameters), concrete_descriptor.options);
                            binding.capture_safe = true;
                            binding.warmup = [](torch::optim::Optimizer &optimizer) {
                                if (auto *sophia = dynamic_cast<Optimizer::Details::SophiaH *>(&optimizer)) {
                                    sophia->ensure_state_initialized();
                                }
                            };
                        } else if constexpr (std::is_same_v<DescriptorType, Optimizer::Details::MuonDescriptor>) {
                            binding.instance = std::make_unique<Optimizer::Details::Muon>(
                                std::move(parameters), concrete_descriptor.options);
                            binding.capture_safe = true;
                            binding.warmup = [](torch::optim::Optimizer &optimizer) {
                                if (auto *muon = dynamic_cast<Optimizer::Details::Muon *>(&optimizer)) {
                                    muon->ensure_state_initialized();
                                }
                            };
                        } else if constexpr (std::is_same_v<DescriptorType, Optimizer::Details::AdaMuonDescriptor>) {
                            binding.instance = std::make_unique<Optimizer::Details::AdaMuon>(
                                std::move(parameters), concrete_descriptor.options);
                            binding.capture_safe = true;
                            binding.warmup = [](torch::optim::Optimizer &optimizer) {
                                if (auto *AdaMuon = dynamic_cast<Optimizer::Details::AdaMuon *>(&optimizer)) {
                                    AdaMuon->ensure_state_initialized();
                                }
                            };
                        } else if constexpr (std::is_same_v<DescriptorType,
                            Optimizer::Details::MuonManifoldDescriptor>) {
                            binding.instance = std::make_unique<Optimizer::Details::MuonManifold>(
                                std::move(parameters), concrete_descriptor.options);
                            binding.capture_safe = true;
                            binding.warmup = [](torch::optim::Optimizer &optimizer) {
                                if (auto *MuonManifold = dynamic_cast<Optimizer::Details::MuonManifold *>(&optimizer)) {
                                    MuonManifold->ensure_state_initialized();
                                }
                            };
                        } else if constexpr (std::is_same_v<DescriptorType, Optimizer::Details::AdafactorDescriptor>) {
                            binding.instance = std::make_unique<Optimizer::Details::Adafactor>(
                                std::move(parameters), concrete_descriptor.options);
                            binding.capture_safe = true;
                            binding.warmup = [](torch::optim::Optimizer &optimizer) {
                                if (auto *adafactor = dynamic_cast<Optimizer::Details::Adafactor *>(&optimizer)) {
                                    adafactor->ensure_state_initialized();
                                }
                            };
                        } else if constexpr (std::is_same_v<DescriptorType, Optimizer::Details::AdagradDescriptor>) {
                            auto options = Optimizer::Details::to_torch_options(concrete_descriptor.options);
                            binding.instance = std::make_unique<Optimizer::Details::Adagrad>(
                                std::move(parameters), options, warmup_buckets);
                            binding.capture_safe = true;
                            binding.warmup = [](torch::optim::Optimizer &optimizer) {
                                if (auto *ada = dynamic_cast<Optimizer::Details::Adagrad *>(&optimizer)) {
                                    ada->ensure_state_initialized();
                                }
                            };
                        } else if constexpr (std::is_same_v<DescriptorType, Optimizer::Details::LAMBDescriptor>) {
                            binding.instance = std::make_unique<Optimizer::Details::LAMB>(
                                std::move(parameters), concrete_descriptor.options);
                            binding.capture_safe = true;
                            binding.warmup = [](torch::optim::Optimizer &optimizer) {
                                if (auto *lamb = dynamic_cast<Optimizer::Details::LAMB *>(&optimizer)) {
                                    lamb->ensure_state_initialized();
                                }
                            };
                        } else if constexpr (std::is_same_v<DescriptorType, Optimizer::Details::LionDescriptor>) {
                            binding.instance = std::make_unique<Optimizer::Details::Lion>(
                                std::move(parameters), concrete_descriptor.options);
                            binding.capture_safe = true;
                            binding.warmup = [](torch::optim::Optimizer &optimizer) {
                                if (auto *lion = dynamic_cast<Optimizer::Details::Lion *>(&optimizer)) {
                                    lion->ensure_state_initialized();
                                }
                            };
                        } else if constexpr (std::is_same_v<DescriptorType, Optimizer::Details::RMSpropDescriptor>) {
                            auto options = Optimizer::Details::to_torch_options(concrete_descriptor.options);
                            binding.instance = std::make_unique<Optimizer::Details::RMSProp>(
                                std::move(parameters), options, warmup_buckets);
                            binding.capture_safe = true;
                            binding.warmup = [](torch::optim::Optimizer &optimizer) {
                                if (auto *rmsp = dynamic_cast<Optimizer::Details::RMSProp *>(&optimizer)) {
                                    rmsp->ensure_state_initialized();
                                }
                            };
                        } else {
                            static_assert(sizeof(DescriptorType) == 0,
                                          "Unsupported optimizer descriptor provided to Model::set_optimizer.");
                        }
                        return binding;
                    }, config);
            };


            optimizer_.reset();
            local_optimizers_.clear();

            std::vector<torch::Tensor> global_parameters{};
            std::vector<std::vector<torch::Tensor> > global_warmup_buckets{};
            global_warmup_buckets.reserve(layers_.size());

            for (std::size_t index = 0; index < layers_.size(); ++index) {
                const auto &layer = layers_[index];
                if (!layer.module) {
                    if (layer.local.optimizer.has_value()) {
                        throw std::logic_error("Local optimizer requested for a layer without a registered module.");
                    }
                    continue;
                }

                const auto &parameters = layer_parameters_[index];
                if (parameters.empty()) {
                    if (layer.local.optimizer.has_value()) {
                        throw std::logic_error("Local optimizer requested for a layer without trainable parameters.");
                    }
                    continue;
                }

                if (layer.local.optimizer.has_value()) {
                    std::vector<torch::Tensor> optimizer_parameters(parameters.begin(), parameters.end());
                    std::vector<std::vector<torch::Tensor> > warmup_buckets;
                    warmup_buckets.emplace_back(parameters.begin(), parameters.end());
                    local_optimizers_.push_back(build_optimizer_for(
                        *layer.local.optimizer,
                        std::move(optimizer_parameters),
                        std::move(warmup_buckets)));
                } else {
                    global_parameters.insert(global_parameters.end(), parameters.begin(), parameters.end());
                    global_warmup_buckets.emplace_back(parameters.begin(), parameters.end());
                }
            }

            if (!global_parameters.empty()) {
                optimizer_ = build_optimizer_for(
                    descriptor,
                    std::move(global_parameters),
                    std::move(global_warmup_buckets));
            }

            scheduler_.reset();
            if (scheduler.has_value()) {
                if (!optimizer_)
                    throw std::logic_error("Cannot attach a scheduler without a global optimizer.");
                scheduler_ = std::visit(
                    [&](const auto &concrete_descriptor) -> std::unique_ptr<LrScheduler::Details::Scheduler> {
                        return LrScheduler::Details::build_scheduler(*this, *optimizer_->instance, concrete_descriptor);
                    }, std::move(*scheduler));
            }
            configure_step_impl();
        }

        template<class Descriptor>
        void set_loss(Descriptor descriptor) {
            using Decayed = std::decay_t<Descriptor>;
            constexpr bool kSupported = std::disjunction_v<
                std::is_same<Decayed, Loss::Details::MSEDescriptor>,
                std::is_same<Decayed, Loss::Details::CrossEntropyDescriptor>,
                std::is_same<Decayed, Loss::Details::BCEWithLogitsDescriptor>,
                std::is_same<Decayed, Loss::Details::CosineEmbeddingDescriptor>,
                std::is_same<Decayed, Loss::Details::KLDivDescriptor>,
                std::is_same<Decayed, Loss::Details::MAEDescriptor>,
                std::is_same<Decayed, Loss::Details::MarginRankingDescriptor>,
                std::is_same<Decayed, Loss::Details::NegativeLogLikelihoodDescriptor>,
                std::is_same<Decayed, Loss::Details::SmoothL1Descriptor>,
                std::is_same<Decayed, Loss::Details::DiceDescriptor>,
                std::is_same<Decayed, Loss::Details::TverskyDescriptor>,
                std::is_same<Decayed, Loss::Details::LovaszSoftmaxDescriptor> >;
            static_assert(kSupported, "Unsupported loss descriptor type provided to Model::set_loss.");

            loss_descriptor_ = LossDescriptor{std::in_place_type<Decayed>, std::move(descriptor)};
        }

        void set_regularization(std::vector<Regularization::Descriptor> descriptors) {
            if (regularization_configured_)
                throw std::logic_error("Regularization descriptors have already been configured.");
            regularization_configured_ = true;
            global_regularization_parameters_ = collect_global_trainable_parameters();
            global_regularization_bindings_.clear();
            global_regularization_bindings_.reserve(descriptors.size());

            for (auto &descriptor: descriptors) {
                global_regularization_bindings_.push_back(
                    make_regularization_binding(std::move(descriptor), global_regularization_parameters_));
            }
            mark_graph_regularization_metadata_dirty();
        }

        void clear_regularization() noexcept {
            for (auto &bindings: layer_regularization_bindings_) {
                bindings.clear();
            }
            global_regularization_bindings_.clear();
            global_regularization_parameters_.clear();
            regularization_configured_ = false;
            mark_graph_regularization_metadata_dirty();
        }

        [[nodiscard]] bool has_regularization() const noexcept {
            if (!global_regularization_bindings_.empty())
                return true;
            return std::any_of(
                layer_regularization_bindings_.begin(),
                layer_regularization_bindings_.end(),
                [](const auto &bindings) { return !bindings.empty(); });
        }

        [[nodiscard]] torch::Tensor compute_regularization_penalty(GraphMode graph_mode = GraphMode::Disabled) const {
            const auto fallback_options = torch::TensorOptions().dtype(torch::kFloat32).device(device_);
            const auto fallback = std::optional<torch::TensorOptions>{fallback_options};


            torch::Tensor total;
            bool initialised = false;

            auto accumulate_penalty = [&](const RegularizationBinding &binding,
                                          const std::vector<torch::Tensor> &parameters,
                                          GraphRegularizationBindingInfo *metadata) {
                if (!binding.accumulator) {
                    if (graph_mode != GraphMode::Disabled && metadata && !metadata->initialised) {
                        metadata->initialised = true;
                        metadata->participates = false;
                    }
                    return;
                }

                auto penalty = binding.accumulator(parameters, fallback);
                if (!penalty.defined()) {
                    if (graph_mode != GraphMode::Disabled && metadata) {
                        if (!metadata->initialised) {
                            metadata->initialised = true;
                            metadata->participates = false;
                        } else if (metadata->participates) {
                            throw std::runtime_error(
                                "Regularisation binding toggled its participation during CUDA graph execution. "
                                "Disable graph mode or ensure the descriptor emits a tensor every step.");
                        }
                    }
                    return;
                }

                if (graph_mode != GraphMode::Disabled && metadata) {
                    const auto signature = describe_tensor_signature(penalty);
                    if (!metadata->initialised) {
                        metadata->initialised = true;
                        metadata->participates = true;
                        metadata->signature = signature;
                    } else {
                        if (!metadata->participates) {
                            throw std::runtime_error(
                                "Regularisation binding activated after CUDA graph capture. "
                                "Disable graph mode or adjust the descriptor to keep participation consistent.");
                        }
                        if (!signatures_equal(metadata->signature, signature)) {
                            throw std::runtime_error(
                                "Regularisation binding produced a tensor with a dynamic signature during CUDA graph execution. "
                                "Expected "
                                + format_signature(metadata->signature) + " but received "
                                + format_signature(signature) + '.');
                        }
                    }
                }


                if (!initialised) {
                    total = penalty;
                    initialised = true;
                    return;
                }
                if (graph_mode == GraphMode::Disabled) {
                    if (penalty.device() != total.device()) {
                        penalty = penalty.to(total.device());
                    }
                    if (penalty.scalar_type() != total.scalar_type()) {
                        penalty = penalty.to(total.scalar_type());
                    }
                } else {
                    if (penalty.device() != total.device()) {
                        throw std::runtime_error(
                            "Regularisation binding returned a tensor on a different device during CUDA graph execution.");
                    }
                    if (penalty.scalar_type() != total.scalar_type()) {
                        throw std::runtime_error(
                            "Regularisation binding returned a tensor with a different dtype during CUDA graph execution.");
                    }
                }
                total.add_(penalty);
            };

            for (std::size_t index = 0; index < global_regularization_bindings_.size(); ++index) {
                auto *metadata = graph_mode == GraphMode::Disabled
                                     ? nullptr
                                     : &graph_global_regularization_metadata_[index];
                accumulate_penalty(global_regularization_bindings_[index], global_regularization_parameters_, metadata);
            }

            for (std::size_t index = 0; index < layer_regularization_bindings_.size(); ++index) {
                const auto &bindings = layer_regularization_bindings_[index];
                if (bindings.empty()) {
                    continue;
                }

                const auto &parameters = layer_parameters_[index];
                for (std::size_t binding_index = 0; binding_index < bindings.size(); ++binding_index) {
                    auto *metadata = graph_mode == GraphMode::Disabled
                                         ? nullptr
                                         : &graph_layer_regularization_metadata_[index][binding_index];
                    accumulate_penalty(bindings[binding_index], parameters, metadata);
                }
            }

            if (!initialised) {
                return torch::zeros({}, fallback_options);
            }

            return total;
        }

        Model &use_cuda(bool use_cuda = true) {
            const auto cuda_available = torch::cuda::is_available();

            if (use_cuda && !cuda_available) {
                std::cerr << "CUDA requested but unavailable (from: torch::cuda::is_available()). Falling back to CPU."
                        << std::endl;
            }

            if (use_cuda && cuda_available) {
                device_ = torch::Device(torch::kCUDA, /*index=*/0);
            } else {
                device_ = torch::Device(torch::kCPU, /*index=*/0);
            }

            this->to(device_);
            return *this;
        }

        [[nodiscard]] const torch::Device &device() const noexcept { return device_; }


        [[nodiscard]] torch::Tensor forward(torch::Tensor input) {
            return forward_internal(std::move(input), {}, nullptr, nullptr);
        }

        [[nodiscard]] torch::Tensor forward(torch::Tensor input, ForwardOptions options) {
            return forward_internal(std::move(input), std::move(options), nullptr, nullptr);
        }

        struct ForwardActivationCaptureResult {
            torch::Tensor logits{};
            torch::Tensor activation{};
        };

        [[nodiscard]] ForwardActivationCaptureResult forward_with_activation_capture(
            torch::Tensor input,
            torch::nn::Module *target_module,
            ForwardOptions options = {}) {
            if (target_module == nullptr)
                throw std::invalid_argument("forward_with_activation_capture requires a valid module pointer.");


            auto *layer = resolve_registered_layer(target_module);
            if (layer == nullptr)
                throw std::runtime_error("Requested module is not part of the model graph.");


            torch::Tensor captured_activation;
            auto logits = forward_internal(std::move(input), std::move(options), layer, &captured_activation);
            if (!captured_activation.defined())
                throw std::runtime_error("Failed to capture activation for the requested module.");

            ForwardActivationCaptureResult result{};
            result.logits = std::move(logits);
            result.activation = std::move(captured_activation);
            return result;
        }

        [[nodiscard]] torch::Tensor forward_internal(torch::Tensor input, ForwardOptions options,
                                                     Layer::Details::RegisteredLayer *capture_layer,
                                                     torch::Tensor *captured_activation) {
            const auto phase = is_training() ? GraphExecutionPhase::Training : GraphExecutionPhase::Inference;
            const auto requested_graph_mode = options.graph_mode;
            const bool graph_mode_active = graph_execution_enabled(requested_graph_mode, phase);
            auto resolved_graph_mode = graph_mode_active ? requested_graph_mode : GraphMode::Disabled;
            if (capture_layer != nullptr && resolved_graph_mode != GraphMode::Disabled) {
                throw std::runtime_error("Activation capture is not supported when graph execution is enabled.");
            }

#ifdef TORCH_CUDA_AVAILABLE
            if (phase == GraphExecutionPhase::Inference && resolved_graph_mode == GraphMode::Capture) {
                auto &state = graph_capture_state(phase);
                if (state.captured && !state.dirty) {
                    resolved_graph_mode = GraphMode::Replay;
                }
            }
#endif
            options.graph_mode = resolved_graph_mode;

            if (capture_layer != nullptr && options.buffering_enabled()) {
                throw std::runtime_error("Activation capture is not supported when forward chunking is enabled.");
            }


            auto execute = [&](torch::Tensor tensor, GraphMode mode) {
                return execute_plan(std::move(tensor), mode, capture_layer, captured_activation);
            };


            if (phase == GraphExecutionPhase::Inference) {
                if (resolved_graph_mode == GraphMode::Replay) {
#ifdef TORCH_CUDA_AVAILABLE
                    auto &state = graph_capture_state(phase);
                    if (!state.captured || state.dirty || !state.graph) {
                        throw std::runtime_error(
                            "CUDA graph replay requested for inference before a capture was recorded.");
                    }
                    ensure_graph_input_shape(GraphMode::Replay, input);
                    ensure_execution_workspace();
                    input = stage_tensor_for_execution(std::move(input));
                    copy_into_graph_input_buffer(std::move(input), workspace_tensor_policy(GraphMode::Replay));
                    if (!state.capture_stream.has_value()) {
                        throw std::runtime_error(
                            "CUDA graph replay requested for inference without an associated capture stream.");
                    }
                    torch::cuda::CUDAStreamGuard guard(*state.capture_stream);
                    state.graph->replay();
                    return graph_output_tensor();
#else
                    throw std::runtime_error("CUDA graph replay requested but CUDA support is unavailable.");
#endif
                }

                if (resolved_graph_mode == GraphMode::Capture) {
#ifdef TORCH_CUDA_AVAILABLE
                    auto &state = graph_capture_state(phase);
                    if (state.dirty) {
                        reset_graph_shape_cache(GraphMode::Capture);
                    }
                    ensure_graph_input_shape(GraphMode::Capture, input);
                    if (!state.graph) {
                        state.graph = std::make_unique<torch::cuda::CUDAGraph>();
                    } else {
                        state.graph->reset();
                    }
                    if (!state.capture_stream.has_value()) {
                        state.capture_stream = torch::cuda::getStreamFromPool();
                    }
                    state.captured = false;
                    torch::cuda::CUDAStreamGuard guard(*state.capture_stream);
                    bool capture_started = false;
                    try {
                        state.graph->capture_begin(*state.capture_stream);
                        capture_started = true;
                        auto result = execute(std::move(input), GraphMode::Capture);
                        state.graph->capture_end();
                        capture_started = false;
                        state.captured = true;
                        state.dirty = false;
                        state.loss_buffer = torch::Tensor{};
                        return result;
                    } catch (...) {
                        if (capture_started) {
                            try {
                                state.graph->capture_end();
                            } catch (...) {
                            }
                        }
                        state.graph->reset();
                        state.capture_stream.reset();
                        state.captured = false;
                        state.dirty = true;
                        throw;
                    }
#else
                    throw std::runtime_error("CUDA graph capture requested but CUDA support is unavailable.");
#endif
                }
            }


            if (resolved_graph_mode != GraphMode::Disabled) {
                ensure_graph_input_shape(resolved_graph_mode, input);
            }


            const bool can_buffer = options.buffering_enabled() && input.defined() && input.dim() > 0;
            if (!can_buffer) {
                return execute(std::move(input), resolved_graph_mode);
            }

            const auto chunk_limit = static_cast<int64_t>(*options.max_chunk_size);
            if (chunk_limit <= 0) {
                return execute(std::move(input), resolved_graph_mode);
            }
            const auto leading = input.size(0);
            if (leading == 0 || leading <= chunk_limit) {
                return execute(std::move(input), resolved_graph_mode);
            }
            std::vector<torch::Tensor> outputs;
            outputs.reserve(static_cast<std::size_t>((leading + chunk_limit - 1) / chunk_limit));

            for (int64_t offset = 0; offset < leading; offset += chunk_limit) {
                const auto current = std::min<int64_t>(chunk_limit, leading - offset);
                auto chunk = input.narrow(0, offset, current);
                outputs.push_back(execute(std::move(chunk), resolved_graph_mode));
            }

            return torch::cat(outputs, 0);
        }

        torch::Tensor execute_plan(torch::Tensor tensor, GraphMode graph_mode,
                                   Layer::Details::RegisteredLayer *capture_layer = nullptr,
                                   torch::Tensor *captured_activation = nullptr) {
            tensor = stage_tensor_for_execution(std::move(tensor));
            if (graph_mode == GraphMode::Replay) {
                throw std::logic_error("Model::execute_plan cannot be invoked in replay mode.");
            }


            auto apply_calibrations = [&](torch::Tensor value) {
                ensure_graph_calibration_metadata_capacity(graph_mode);

                for (std::size_t index = 0; index < calibration_methods_.size(); ++index) {
                    const auto &calibration = calibration_methods_[index];
                    value = calibration->transform(std::move(value));

                    if (graph_mode != GraphMode::Disabled) {
                        if (!value.defined()) {
                            throw std::runtime_error(
                                "Calibration module produced an undefined tensor during CUDA graph execution.");
                        }

                        const auto signature = describe_tensor_signature(value);
                        auto &metadata = graph_calibration_metadata_[index];

                        if (!metadata.initialised) {
                            metadata.initialised = true;
                            metadata.signature = signature;
                        } else if (!signatures_equal(metadata.signature, signature)) {
                            throw std::runtime_error(
                                "Calibration module output shape changed between CUDA graph executions. "
                                "Disable graph mode or adjust the calibration configuration.");
                        }
                    }
                }
                return value;
            };

            if (!has_compiled_routing() || execution_steps_.empty()) {
                for (auto &layer: layers_) {
                    auto module_output = layer.forward(std::move(tensor));
                    if (capture_layer != nullptr && captured_activation != nullptr &&
                        capture_layer == &layer) {
                        *captured_activation = module_output;
                    }
                    tensor = Activation::Details::apply(layer.activation, std::move(module_output));
                }
                return apply_calibrations(std::move(tensor));
            }

            ensure_execution_workspace();

            constexpr std::size_t kInputNodeIndex = 0;
            copy_into_graph_input_buffer(std::move(tensor), workspace_tensor_policy(graph_mode));

            auto &workspace = graph_workspace_;


            const auto output_index = resolve_output_node_index();
#ifndef NDEBUG
            assert(output_index < workspace.node_buffers.size());
#endif
            workspace.bind_output(output_index);

            for (const auto &step: execution_steps_) {
                switch (step.kind) {
                    case ExecutionStep::Kind::Module: {
                        const auto input_index = step.module.input_index;
#ifndef NDEBUG
                        assert(step.module.layer != nullptr);
                        assert(input_index < workspace.node_buffers.size());
                        assert(step.activation_index < workspace.node_buffers.size());
#endif
                        auto input_tensor = workspace.node_buffers[input_index];
#ifndef NDEBUG
                        assert(input_tensor.defined());
#endif
                        auto *layer = step.module.layer;
                        auto output_tensor = layer->forward(input_tensor);
                        if (capture_layer != nullptr && captured_activation != nullptr && capture_layer == layer) {
                            *captured_activation = output_tensor;
                        }
                        output_tensor = Activation::Details::apply(layer->activation, std::move(output_tensor));
                        auto &destination = workspace.node_buffers[step.activation_index];
                        copy_tensor_into(destination, output_tensor, workspace_tensor_policy(graph_mode));
                        break;
                    }
                    case ExecutionStep::Kind::Join: {
#ifndef NDEBUG
                        assert(step.join.workspace_index < workspace.join_scratch.size());
#endif
                        auto &scratch = workspace.join_scratch[step.join.workspace_index];
                        scratch.clear();
                        scratch.reserve(step.join.producers.size());
                        for (auto producer: step.join.producers) {
#ifndef NDEBUG
                            assert(producer < workspace.node_buffers.size());
#endif
                            auto value = workspace.node_buffers[producer];
#ifndef NDEBUG
                            assert(value.defined());
#endif
                            scratch.push_back(value);
                        }

                        torch::Tensor joined;
                        switch (step.join.policy) {
                            case MergePolicy::Strict: {
#ifndef NDEBUG
                                assert(scratch.size() == 1);
#endif
                                joined = scratch.front();
                                break;
                            }
                            case MergePolicy::Broadcast: {
#ifndef NDEBUG
                                assert(!scratch.empty());
#endif
                                joined = scratch.front();
                                for (std::size_t index = 1; index < scratch.size(); ++index) {
                                    joined = joined + scratch[index];
                                }
                                break;
                            }
                            case MergePolicy::Stack: {
#ifndef NDEBUG
                                assert(!scratch.empty());
#endif
                                const auto dimension = step.join.concat_dimension.value_or(1);
                                joined = torch::cat(scratch, dimension);
                                break;
                            }
                        }

#ifndef NDEBUG
                        assert(step.activation_index < workspace.node_buffers.size());
#endif
                        auto &destination = workspace.node_buffers[step.activation_index];
                        copy_tensor_into(destination, joined, workspace_tensor_policy(graph_mode));

                        scratch.clear();
                        break;
                    }
                    case ExecutionStep::Kind::Output: {
                        const auto upstream_index = step.output.input_index;
#ifndef NDEBUG
                        assert(upstream_index < workspace.node_buffers.size());
#endif
                        auto upstream_tensor = workspace.node_buffers[upstream_index];
#ifndef NDEBUG
                        assert(upstream_tensor.defined());
#endif
                        copy_tensor_into(workspace.output, upstream_tensor, workspace_tensor_policy(graph_mode));
                        workspace.bind_output(step.activation_index);
                        break;
                    }
                }


#ifndef NDEBUG
                if (step.kind == ExecutionStep::Kind::Module) {
                    const auto node_index = step.activation_index;
                    if (node_index < compiled_nodes_.size()) {
                        const auto &node = compiled_nodes_[node_index];
                        if (node.kind == CompiledNode::Kind::Module) {
                            assert(node.index < cached_layer_pointers_.size());
                            assert(cached_layer_pointers_[node.index] == step.module.layer);
                        }
                    }
                }
#endif
            }
#ifndef NDEBUG
            assert(output_index < workspace.node_buffers.size());
#endif

            auto output_tensor = workspace.node_buffers[output_index];
            if (!output_tensor.defined()) {
                throw std::runtime_error("Model::forward produced an undefined tensor at the output node.");
            }

            copy_tensor_into(workspace.output, output_tensor, workspace_tensor_policy(graph_mode));
            workspace.bind_output(output_index);

            auto result = graph_output_tensor();

            return apply_calibrations(std::move(result));
        }


        void set_model_name(std::string name) { model_name_ = std::move(name); }

        [[nodiscard]] std::string model_name() const {
            if (model_name_.empty()) {
                return this->name();
            }
            return model_name_;
        }

        void save(const std::filesystem::path &directory) const {
            namespace fs = std::filesystem;
            if (directory.empty()) {
                throw std::invalid_argument("Model::save requires a non-empty directory path.");
            }

            auto target_dir = directory / model_name();

            if (fs::exists(target_dir)) {
                std::cout << "\aDirectory '" << target_dir.string() << "' already exists. Overwrite? [y/N]: ";
                std::string response;
                std::getline(std::cin, response);

                if (response.empty() || (response[0] != 'y' && response[0] != 'Y')) {
                    int counter = 1;
                    while (true) {
                        auto candidate = directory / (model_name() + "_" + std::to_string(counter));
                        if (!fs::exists(candidate)) {
                            target_dir = candidate;
                            break;
                        }
                        ++counter;
                    }
                    std::cout << "Saving model as: " << target_dir.string() << std::endl;
                } else {
                    std::cout << "Overwriting existing model in: " << target_dir.string() << std::endl;
                }
            }

            fs::create_directories(target_dir);

            const auto architecture_path = target_dir / "architecture.json";
            const auto parameters_path = target_dir / "parameters.binary";

            Common::SaveLoad::PropertyTree architecture;
            architecture.put("name", model_name());
            architecture.add_child("modules", Common::SaveLoad::serialize_module_list(module_descriptors_));

            try {
                Common::SaveLoad::write_json_file(architecture_path, architecture);
            } catch (const std::exception &error) {
                throw std::runtime_error(
                    std::string("Failed to write architecture description to '")
                    + architecture_path.string() + "': " + error.what());
            }

            torch::serialize::OutputArchive archive;
            torch::nn::Module::save(archive);
            archive.save_to(parameters_path.string());
        }


        void load(const std::filesystem::path &directory) {
            namespace fs = std::filesystem;
            if (directory.empty()) {
                throw std::invalid_argument("Model::load requires a non-empty directory path.");
            }

            const auto architecture_path = directory / "architecture.json";
            const auto parameters_path = directory / "parameters.binary";

            if (!fs::exists(architecture_path)) {
                throw std::runtime_error(std::string("Architecture file not found at '")
                                         + architecture_path.string() + "'.");
            }
            if (!fs::exists(parameters_path)) {
                throw std::runtime_error(std::string("Parameter archive not found at '")
                                         + parameters_path.string() + "'.");
            }

            Common::SaveLoad::PropertyTree architecture;
            try {
                architecture = Common::SaveLoad::read_json_file(architecture_path);
            } catch (const std::exception &error) {
                throw std::runtime_error(std::string("Failed to read architecture description from '")
                                         + architecture_path.string() + "': " + error.what());
            }

            auto modules_node = architecture.get_child_optional("modules");
            if (!modules_node) {
                throw std::runtime_error(std::string("Architecture description '") + architecture_path.string()
                                         + "' is missing the 'modules' entry.");
            }

            auto descriptors = Common::SaveLoad::deserialize_module_list(*modules_node, "module");

            reset_runtime_state();

            if (auto name_value = architecture.get_optional<std::string>("name")) {
                model_name_ = std::move(*name_value);
            } else {
                model_name_.clear();
            }

            for (auto &descriptor: descriptors) {
                add(std::move(descriptor.descriptor), std::move(descriptor.name));
            }

            torch::serialize::InputArchive validation_archive;
            try {
                validation_archive.load_from(parameters_path.string());
            } catch (const c10::Error &error) {
                throw std::runtime_error(std::string("Failed to open parameter archive '")
                                         + parameters_path.string() + "': " + error.what());
            }

            auto parameters = this->named_parameters(/*recurse=*/true);
            for (const auto &item: parameters) {
                torch::Tensor stored;
                try {
                    validation_archive.read(item.key(), stored);
                } catch (const c10::Error &error) {
                    throw std::runtime_error("Checkpoint is missing parameter '" + item.key() + "': " + error.what());
                }
                if (!stored.defined()) {
                    throw std::runtime_error("Checkpoint parameter '" + item.key() + "' is undefined.");
                }
                if (stored.sizes() != item.value().sizes()) {
                    throw std::runtime_error("Parameter '" + item.key() + "' shape mismatch: expected "
                                             + format_tensor_shape(item.value()) + " but found "
                                             + format_tensor_shape(stored) + ".");
                }
            }

            auto buffers = this->named_buffers(/*recurse=*/true);
            for (const auto &item: buffers) {
                if (!item.value().defined()) {
                    continue;
                }
                torch::Tensor stored;
                try {
                    validation_archive.read(item.key(), stored);
                } catch (const c10::Error &error) {
                    throw std::runtime_error("Checkpoint is missing buffer '" + item.key() + "': " + error.what());
                }
                if (!stored.defined()) {
                    throw std::runtime_error("Checkpoint buffer '" + item.key() + "' is undefined.");
                }
                if (stored.sizes() != item.value().sizes()) {
                    throw std::runtime_error("Buffer '" + item.key() + "' shape mismatch: expected "
                                             + format_tensor_shape(item.value()) + " but found "
                                             + format_tensor_shape(stored) + ".");
                }
            }

            torch::serialize::InputArchive archive;
            try {
                archive.load_from(parameters_path.string());
                torch::nn::Module::load(archive);
            } catch (const c10::Error &error) {
                throw std::runtime_error(std::string("Failed to load parameters from '")
                                         + parameters_path.string() + "': " + error.what());
            }

            configure_step_impl();
        }

        void calibrate(const torch::Tensor &inputs, const torch::Tensor &targets,
                       const Calibration::Descriptor &descriptor, bool plot = true,
                       std::optional<std::pair<torch::Tensor, torch::Tensor> > validation = std::nullopt,
                       Calibration::Options options = {}) {
            torch::NoGradGuard guard;
            eval();
            if (!inputs.defined() || !targets.defined()) {
                throw std::invalid_argument("Calibration requires defined input and target tensors.");
            }

            auto build_streaming_outputs = [this, &options](torch::Tensor dataset_inputs,
                                                            torch::Tensor dataset_targets) -> std::optional<std::pair<
                torch::Tensor, torch::Tensor> > {
                if (!dataset_inputs.defined() || dataset_inputs.dim() == 0 || dataset_inputs.size(0) == 0) {
                    return std::nullopt;
                }


                StreamingOptions streaming_options{};
                streaming_options.forward_chunk_size = options.forward_chunk_size;
                if (options.forward_buffer_batches > 0) {
                    const auto chunk_size_value = static_cast<std::size_t>(options.forward_chunk_size.value_or(
                        Core::kDefaultTrainingConfig.batch_size));
                    if (chunk_size_value == 0) {
                        throw std::invalid_argument(
                            "Calibration forward chunk size must be positive when buffering is enabled.");
                    }
                    streaming_options.batch_size = chunk_size_value;
                    streaming_options.buffer_batches = options.forward_buffer_batches;
                }

                std::vector<torch::Tensor> logits_chunks;
                std::vector<torch::Tensor> target_chunks;
                logits_chunks.reserve(8); // TODO: Modify
                target_chunks.reserve(8);

                auto prepare = [&](torch::Tensor batch_inputs,
                                   torch::Tensor batch_targets) -> std::optional<StreamingBatch> {
                    if (!batch_inputs.defined() || !batch_targets.defined()) {
                        return std::nullopt;
                    }

                    StreamingBatch batch{};
                    batch.inputs = std::move(batch_inputs);
                    batch.targets = std::move(batch_targets);

                    if (batch.targets.defined()) {
                        batch.reference_targets = DeferredHostTensor::from_tensor(batch.targets, false);
                    }

                    return batch;
                };

                auto consume = [&](torch::Tensor outputs, StreamingBatch batch) {
                    auto logits_batch = std::move(outputs);
                    if (logits_batch.defined() && !logits_batch.device().is_cpu()) {
                        logits_batch = logits_batch.to(torch::kCPU);
                    }
                    if (logits_batch.defined()) {
                        logits_chunks.push_back(logits_batch.detach());
                    }

                    torch::Tensor targets_cpu = batch.reference_targets.defined()
                                                    ? batch.reference_targets.materialize()
                                                    : batch.targets;

                    if (targets_cpu.defined() && !targets_cpu.device().is_cpu()) {
                        targets_cpu = targets_cpu.to(torch::kCPU);
                    }
                    if (targets_cpu.defined()) {
                        target_chunks.push_back(targets_cpu.detach());
                    }
                };

                const bool processed = stream_forward(std::move(dataset_inputs), std::move(dataset_targets),
                                                      streaming_options, prepare, consume);

                if (processed && !logits_chunks.empty() && !target_chunks.empty()) {
                    return std::pair<torch::Tensor, torch::Tensor>{
                        torch::cat(logits_chunks, 0),
                        torch::cat(target_chunks, 0)
                    };
                }

                return std::nullopt;
            };

            auto calibration_pair = build_streaming_outputs(inputs, targets);
            torch::Tensor logits;
            torch::Tensor calibration_targets;

            if (calibration_pair.has_value()) {
                logits = std::move(calibration_pair->first);
                calibration_targets = std::move(calibration_pair->second);
            } else {
                ForwardOptions forward_options{};
                if (options.forward_chunk_size.has_value()) {
                    forward_options.max_chunk_size = options.forward_chunk_size;
                }
                auto fallback_logits = forward(inputs, std::move(forward_options));
                if (!fallback_logits.device().is_cpu()) {
                    fallback_logits = fallback_logits.to(torch::kCPU);
                }
                logits = fallback_logits.detach();
                calibration_targets = targets;
                if (!calibration_targets.device().is_cpu()) {
                    calibration_targets = calibration_targets.to(torch::kCPU);
                }
            }

            std::optional<std::pair<torch::Tensor, torch::Tensor> > processed_validation = std::nullopt;
            if (validation.has_value()) {
                const auto &validation_inputs = validation->first;
                const auto &validation_targets = validation->second;
                if (validation_inputs.defined() && validation_targets.defined()) {
                    if (auto validation_pair = build_streaming_outputs(validation_inputs, validation_targets)) {
                        processed_validation = std::move(*validation_pair);
                    } else {
                        ForwardOptions forward_options{};
                        if (options.forward_chunk_size.has_value()) {
                            forward_options.max_chunk_size = options.forward_chunk_size;
                        }
                        auto validation_logits = forward(validation_inputs, std::move(forward_options));
                        if (!validation_logits.device().is_cpu()) {
                            validation_logits = validation_logits.to(torch::kCPU);
                        }
                        auto validation_targets_cpu = validation_targets;
                        if (!validation_targets_cpu.device().is_cpu()) {
                            validation_targets_cpu = validation_targets_cpu.to(torch::kCPU);
                        }
                        processed_validation = std::make_pair(validation_logits.detach(), validation_targets_cpu);
                    }
                }
            }

            auto method = Calibration::Calibrate(*this, device_, descriptor,
                                                 [&logits, &calibration_targets](torch::nn::Module &) {
                                                     return std::pair<torch::Tensor, torch::Tensor>{
                                                         logits, calibration_targets
                                                     };
                                                 }, std::move(processed_validation), std::move(options), plot);
            calibration_methods_.push_back(std::move(method));
            mark_graph_calibration_metadata_dirty();
        }

        [[nodiscard]] torch::Tensor compute_loss(const torch::Tensor &prediction,
                                                 const torch::Tensor &target,
                                                 const std::optional<torch::Tensor> &weight = std::nullopt) const {
            if (!loss_descriptor_.has_value()) {
                throw std::logic_error("Loss function has not been configured.");
            }
            return std::visit(
                [&](const auto &descriptor) {
                    return Loss::Details::compute(descriptor, prediction, target, weight);
                }, *loss_descriptor_);
        }

        auto evaluate(torch::Tensor evaluation_inputs, torch::Tensor evaluation_targets,
                      Evaluation::ClassificationDescriptor descriptor,
                      std::vector<Metric::Classification::Descriptor> metrics,
                      Evaluation::Options options = {}) -> Evaluation::ClassificationReport {
            return Evaluation::Evaluate(
                *this,
                std::move(evaluation_inputs),
                std::move(evaluation_targets),
                descriptor,
                std::move(metrics),
                options);
        }

        //TODO: do it cleaner
        auto evaluate(torch::Tensor evaluation_inputs, torch::Tensor evaluation_targets,
                      Evaluation::MultiClassificationDescriptor descriptor,
                      std::vector<Metric::Classification::Descriptor> metrics,
                      Evaluation::Options options = {}) -> Evaluation::ClassificationReport {
            return Evaluation::Evaluate(
                *this,
                std::move(evaluation_inputs),
                std::move(evaluation_targets),
                descriptor,
                std::move(metrics),
                options);
        }

        auto evaluate(torch::Tensor evaluation_inputs, torch::Tensor evaluation_targets,
                      Evaluation::SegmentationDescriptor descriptor,
                      std::vector<Metric::Classification::Descriptor> metrics,
                      Evaluation::Options options = {}) -> Evaluation::ClassificationReport {
            return Evaluation::Evaluate(
                *this,
                std::move(evaluation_inputs),
                std::move(evaluation_targets),
                descriptor,
                std::move(metrics),
                options);
        }

        template<class Descriptor, class... Args>
        decltype(auto) plot(Descriptor descriptor, Args &&... args);

        void zero_grad(bool set_to_none = false) {
            bool handled{false};

            if (optimizer_) {
                optimizer_->instance->zero_grad(set_to_none);
                handled = true;
            }
            if (!local_optimizers_.empty()) {
                handled = true;
                for (auto &optimizer: local_optimizers_) {
                    optimizer.instance->zero_grad(set_to_none);
                }
            }

            if (!handled) {
                torch::nn::Module::zero_grad(set_to_none);
            }
        }

        void step() { (this->*step_impl_)(); }

        [[nodiscard]] bool has_optimizer() const noexcept {
            return static_cast<bool>(optimizer_) || !local_optimizers_.empty();
        }

        [[nodiscard]] std::size_t local_optimizer_count() const noexcept {
            return local_optimizers_.size();
        }

        [[nodiscard]] bool is_amp_training_active() const noexcept { return amp_training_active_; }

        [[nodiscard]] bool has_loss() const noexcept { return loss_descriptor_.has_value(); }

        [[nodiscard]] torch::optim::Optimizer &optimizer() {
            if (!optimizer_) {
                throw std::logic_error("Optimizer has not been configured.");
            }
            return *optimizer_->instance;
        }

        template<class Config, class Dataset = Core::SupervisedDataset>
        void train(Dataset dataset) {
            static_assert(Config::batch_size > 0, "Batch size must be greater than zero.");

            if (dataset.empty()) {
                return;
            }

            TrainOptions options{};
            options.epoch = Config::epochs;
            options.batch_size = Config::batch_size;
            options.shuffle = Config::shuffle;
            options.buffer_vram = Config::buffer_vram;
            options.monitor = false;

            auto packed = TrainingDetails::pack_dataset(std::move(dataset));
            train(std::move(packed.inputs), std::move(packed.targets), options);
        }

        void train(torch::Tensor train_inputs, torch::Tensor train_targets, TrainOptions options = {}) {
            if (!has_optimizer())
                throw std::logic_error("Cannot train without an optimizer.");
            if (!has_loss())
                throw std::logic_error("Cannot train without a loss function.");
            if (!train_inputs.defined() || !train_targets.defined())
                throw std::invalid_argument("Training tensors must be defined.");
            if (train_inputs.dim() == 0 || train_targets.dim() == 0)
                throw std::invalid_argument("Training tensors must not be scalars.");
            if (train_inputs.size(0) != train_targets.size(0))
                throw std::invalid_argument("Mismatched number of training samples between inputs and targets.");
            if (options.batch_size == 0)
                throw std::invalid_argument("Batch size must be greater than zero.");

            std::int64_t fold_count = 1;

            if (options.fold) {
                if (train_inputs.dim() < 2 || train_targets.dim() < 2)
                    throw std::invalid_argument(
                        "Folded datasets must expose at least two dimensions (folds and samples).");
                if (train_targets.size(0) != train_inputs.size(0))
                    throw std::invalid_argument("Folded inputs and targets must share the same number of folds.");
                if (train_targets.size(1) != train_inputs.size(1))
                    throw std::invalid_argument(
                        "Folded inputs and targets must share the same number of samples per fold.");

                fold_count = train_inputs.size(0);
                if (fold_count < 2)
                    throw std::invalid_argument(
                        "K-fold training requires at least two folds when fold mode is enabled.");
            }

            clear_training_telemetry();
            if (options.epoch == 0) {
                return;
            }

            if (options.buffer_vram > 0 && !device_.is_cuda()) {
                throw std::runtime_error("VRAM buffering requires the model to be on a CUDA device.");
            }

            const auto total_samples = options.fold ? train_inputs.size(1) : train_inputs.size(0);
            if (total_samples == 0) {
                return;
            }

            torch::nn::Module::train();
            this->to(device_);

            TrainOptions effective_options = options;
            if (effective_options.stream == nullptr) {
                effective_options.monitor = false;
            }
#ifdef TORCH_CUDA_AVAILABLE
            amp_training_active_ = effective_options.enable_amp && device_.is_cuda();
            if (amp_training_active_) {
                ensure_amp_scaler();
            } else {
                amp_scaler_.reset();
            }
#else
            (void) effective_options.enable_amp;
            amp_training_active_ = false;
#endif
            zero_grad();

            const bool requested_channels_last =
                    effective_options.memory_format == torch::MemoryFormat::ChannelsLast;
            bool channels_last_applicable =
                    requested_channels_last && device_.is_cuda() && has_convolutional_layers_;
#ifdef TORCH_CUDA_AVAILABLE
            channels_last_applicable = channels_last_applicable && torch::cuda::is_available();
#endif
            if (channels_last_applicable) {
                channels_last_applicable = train_inputs.dim() >= 4;
            }

            effective_options.memory_format = channels_last_applicable
                                                  ? torch::MemoryFormat::ChannelsLast
                                                  : torch::MemoryFormat::Contiguous;

            set_tensor_memory_format(effective_options.memory_format);

            auto build_training_dataset = [&](torch::Tensor inputs, torch::Tensor targets) {
                auto dataset = TrainingDetails::prepare_tensor_dataset(std::move(inputs), std::move(targets),
                                                                       effective_options.memory_format);
                dataset = TrainingDetails::ensure_contiguous(std::move(dataset), effective_options.memory_format);
                dataset = TrainingDetails::ensure_cpu(std::move(dataset), effective_options.memory_format);
                return dataset;
            };

            std::optional<typename TrainingDetails::TensorDataset> test_dataset{};

            auto build_evaluation_dataset = [&](const std::vector<torch::Tensor> &dataset,
                                                std::string_view name) -> typename TrainingDetails::TensorDataset {
                if (dataset.size() != 2) {
                    throw std::invalid_argument(std::string(name) +
                                                " must contain exactly 2 tensors: [inputs, targets].");
                }
                const auto &inputs = dataset[0];
                const auto &targets = dataset[1];

                if (!inputs.defined() || !targets.defined()) {
                    throw std::invalid_argument(std::string(name) + " tensors must be defined when provided.");
                }
                if (inputs.size(0) != targets.size(0)) {
                    throw std::invalid_argument("Mismatched number of " + std::string(name) +
                                                " samples between inputs and targets.");
                }
                return TrainingDetails::prepare_tensor_dataset(inputs, targets, effective_options.memory_format);
            };

            if (options.test) {
                test_dataset = build_evaluation_dataset(*options.test, "test");
            }


            auto training_step_binding = TrainingDetails::TrainingStepBinding{};

            const bool use_buffer = effective_options.buffer_vram > 0;


            if (test_dataset) {
                *test_dataset = TrainingDetails::ensure_contiguous(std::move(*test_dataset),
                                                                   effective_options.memory_format);
                *test_dataset = TrainingDetails::ensure_cpu(std::move(*test_dataset), effective_options.memory_format);
            }

            auto run_training_dataset = [&](typename TrainingDetails::TensorDataset dataset,
                                            const std::optional<typename TrainingDetails::TensorDataset> &
                                            evaluation_dataset) {
                auto training_dataset = std::move(dataset);
                if (effective_options.shuffle) {
                    if (use_buffer) {
                        TrainingDetails::run_epochs<true, true>(*this, training_dataset, evaluation_dataset,
                                                                effective_options, training_step_binding);
                    } else {
                        TrainingDetails::run_epochs<false, true>(*this, training_dataset, evaluation_dataset,
                                                                 effective_options, training_step_binding);
                    }
                } else {
                    if (use_buffer) {
                        TrainingDetails::run_epochs<true, false>(*this, training_dataset, evaluation_dataset,
                                                                 effective_options, training_step_binding);
                    } else {
                        TrainingDetails::run_epochs<false, false>(*this, training_dataset, evaluation_dataset,
                                                                  effective_options, training_step_binding);
                    }
                }
            };


            if (options.fold) {
                if (fold_count == 0) {
                    return;
                }
                const auto fold_sample_count = train_inputs.size(1);
                if (fold_sample_count == 0) {
                    return;
                }

                auto flatten_fold_batches = [](torch::Tensor tensor) {
                    if (!tensor.defined()) {
                        return tensor;
                    }
                    if (tensor.dim() < 2) {
                        throw std::invalid_argument(
                            "Folded tensors must expose at least two dimensions when flattening.");
                    }
                    auto sizes = tensor.sizes().vec();
                    const auto combined = sizes[0] * sizes[1];
                    std::vector<int64_t> new_shape;
                    new_shape.reserve(sizes.size() - 1);
                    new_shape.push_back(combined);
                    new_shape.insert(new_shape.end(), sizes.begin() + 2, sizes.end());
                    return tensor.reshape(new_shape);
                };

                auto build_fold_tensor = [&](const torch::Tensor &tensor, std::int64_t held_out_fold) {
                    std::vector<int64_t> training_folds;
                    training_folds.reserve(static_cast<std::size_t>(fold_count - 1));
                    for (std::int64_t fold_index = 0; fold_index < fold_count; ++fold_index) {
                        if (fold_index != held_out_fold) {
                            training_folds.push_back(fold_index);
                        }
                    }

                    auto index_options = torch::TensorOptions().dtype(torch::kLong);
                    auto fold_indices = torch::tensor(training_folds, index_options);
                    if (tensor.device().is_cuda()) {
                        fold_indices = fold_indices.to(tensor.device(), torch::kLong);
                    }

                    auto selected = tensor.index_select(0, fold_indices);
                    return flatten_fold_batches(std::move(selected));
                };

                for (std::int64_t fold_index = 0; fold_index < fold_count; ++fold_index) {
                    reset_training_state();

                    auto validation_inputs = train_inputs.select(0, fold_index);
                    auto validation_targets = train_targets.select(0, fold_index);
                    auto validation_dataset = std::optional<typename TrainingDetails::TensorDataset>{};
                    validation_dataset = build_training_dataset(std::move(validation_inputs),
                                                                std::move(validation_targets));

                    auto training_inputs = build_fold_tensor(train_inputs, fold_index);
                    auto training_targets = build_fold_tensor(train_targets, fold_index);
                    auto training_dataset = build_training_dataset(std::move(training_inputs),
                                                                   std::move(training_targets));

                    run_training_dataset(std::move(training_dataset), validation_dataset);
                }
            } else {
                auto training_dataset = build_training_dataset(std::move(train_inputs), std::move(train_targets));
                run_training_dataset(std::move(training_dataset), test_dataset);
            }
        }

        void reset_training_state() {
            clear_training_telemetry();
            invalidate_graph_capture(GraphExecutionPhase::Training);
            graph_workspace_.invalidate();
#ifdef TORCH_CUDA_AVAILABLE
            if (amp_training_active_) {
                amp_scaler_.reset();
                ensure_amp_scaler();
            }
#endif
        }

        [[nodiscard]] torch::MemoryFormat preferred_tensor_memory_format() const noexcept {
            return tensor_memory_format_;
        }


        void set_staging_observer(std::function<void(const torch::Tensor &, bool)> observer) {
            staging_observer_ = std::move(observer);
        }

    private:
        void ensure_optimizer_graph_capability(GraphMode mode) const {
            if (mode == GraphMode::Disabled) {
                return;
            }

            auto build_error_message = [](const OptimizerBinding &binding) {
                std::string optimizer_name{"optimizer"};
                if (binding.instance) {
                    if (dynamic_cast<torch::optim::AdamW *>(binding.instance.get())) {
                        optimizer_name = "AdamW";
                    }
                }
                return std::string("Optimizer '") + optimizer_name +
                       "' does not support CUDA graph execution; CUDA graphs remain unsupported until a capture-safe " +
                       optimizer_name + " variant is implemented.";
            };

            if (optimizer_) {
                if (!optimizer_->capture_safe)
                    throw std::runtime_error(build_error_message(*optimizer_));
            }
            for (const auto &binding: local_optimizers_) {
                if (!binding.capture_safe)
                    throw std::runtime_error(build_error_message(binding));
            }
        }

        void prepare_optimizers_for_graph(GraphMode mode) {
            if (mode == GraphMode::Disabled) {
                return;
            }
            auto build_error_message = [](const OptimizerBinding &binding) {
                std::string optimizer_name{"optimizer"};
                if (binding.instance) {
                    if (dynamic_cast<torch::optim::AdamW *>(binding.instance.get())) {
                        optimizer_name = "AdamW";
                    }
                }
                return std::string("Optimizer '") + optimizer_name
                       + "' does not support CUDA graph execution; CUDA graphs remain unsupported until a capture-safe "
                       + optimizer_name + " variant is implemented.";
            };

            auto prepare_binding = [&](OptimizerBinding &binding) {
                if (!binding.capture_safe) {
                    throw std::runtime_error(build_error_message(binding));
                }
                if (!binding.warmed_up && binding.warmup) {
                    binding.warmup(*binding.instance);
                    binding.warmed_up = true;
                }
            };

            if (optimizer_) {
                prepare_binding(*optimizer_);
            }
            for (auto &binding: local_optimizers_) {
                prepare_binding(binding);
            }
        }

        void reset_graph_shape_cache(GraphMode mode) const {
            if (mode == GraphMode::Disabled) {
                return;
            }

            graph_input_shape_cache_.reset();
            graph_target_shape_cache_.reset();
        }

        void ensure_graph_input_shape(GraphMode mode, const torch::Tensor &tensor) const {
            if (mode == GraphMode::Disabled) {
                return;
            }

            enforce_graph_shape(mode, tensor, graph_input_shape_cache_, "input tensor");
        }

        void ensure_graph_batch_shapes(GraphMode mode,
                                       const torch::Tensor &inputs,
                                       const torch::Tensor &targets) const {
            if (mode == GraphMode::Disabled) {
                return;
            }

            ensure_graph_input_shape(mode, inputs);
            enforce_graph_shape(mode, targets, graph_target_shape_cache_, "target tensor");
        }

        void ensure_graph_replay_ready(GraphMode mode) const {
            if (mode != GraphMode::Replay) {
                return;
            }

            if (!graph_input_shape_cache_) {
                throw std::runtime_error(
                    "CUDA graph replay requested but no cached input tensor shape is available. "
                    "Run capture before attempting replay.");
            }

            if (!graph_target_shape_cache_) {
                throw std::runtime_error(
                    "CUDA graph replay requested but no cached target tensor shape is available. "
                    "Run capture before attempting replay.");
            }
        }

        void enforce_graph_shape(GraphMode mode,
                                 const torch::Tensor &tensor,
                                 std::optional<std::vector<int64_t> > &storage,
                                 std::string_view tensor_label) const {
            if (mode == GraphMode::Disabled) {
                return;
            }

            if (!tensor.defined()) {
                throw std::invalid_argument(
                    std::string("CUDA graph ") + std::string(tensor_label) + " must be defined.");
            }

            const auto shape = tensor_shape_vector(tensor);

            if (!storage.has_value()) {
                if (mode == GraphMode::Replay) {
                    throw std::runtime_error(
                        std::string("CUDA graph replay requested but no cached ")
                        + std::string(tensor_label)
                        + " shape is available. Capture a graph before replaying.");
                }

                storage = shape;
                return;
            }

            if (*storage != shape) {
                const auto expected = format_shape_vector(*storage);
                const auto actual = format_shape_vector(shape);
                throw std::runtime_error(
                    std::string("CUDA graph ") + std::string(tensor_label) + " shape mismatch. Expected "
                    + expected + " but received " + actual + ".");
            }
        }

        static std::vector<int64_t> tensor_shape_vector(const torch::Tensor &tensor) {
            const auto sizes = tensor.sizes();
            return std::vector<int64_t>(sizes.begin(), sizes.end());
        }

        static std::string format_shape_vector(const std::vector<int64_t> &shape) {
            std::ostringstream stream;
            stream << '[';
            for (std::size_t index = 0; index < shape.size(); ++index) {
                if (index > 0) {
                    stream << ", ";
                }
                stream << shape[index];
            }
            stream << ']';
            return stream.str();
        }

        static std::string scalar_type_to_string(torch::ScalarType type) {
            switch (type) {
                case torch::kByte: return "uint8";
                case torch::kChar: return "int8";
                case torch::kShort: return "int16";
                case torch::kInt: return "int32";
                case torch::kLong: return "int64";
                case torch::kHalf: return "float16";
                case torch::kFloat: return "float32";
                case torch::kDouble: return "float64";
                case torch::kBool: return "bool";
                case torch::kBFloat16: return "bfloat16";
                case torch::kComplexHalf: return "complex16";
                case torch::kComplexFloat: return "complex64";
                case torch::kComplexDouble: return "complex128";
                case torch::kQUInt8: return "quint8";
                case torch::kQInt8: return "qint8";
                case torch::kQInt32: return "qint32";
                default: return std::to_string(static_cast<int>(type));
            }
        }

        static GraphTensorSignature describe_tensor_signature(const torch::Tensor &tensor) {
            GraphTensorSignature signature{};
            signature.device = tensor.device();
            signature.dtype = tensor.scalar_type();
            signature.shape = tensor_shape_vector(tensor);
            return signature;
        }

        static bool signatures_equal(const GraphTensorSignature &lhs, const GraphTensorSignature &rhs) {
            return lhs.device == rhs.device && lhs.dtype == rhs.dtype && lhs.shape == rhs.shape;
        }

        static std::string format_signature(const GraphTensorSignature &signature) {
            std::ostringstream stream;
            stream << "shape=" << format_shape_vector(signature.shape)
                    << ", dtype=" << scalar_type_to_string(signature.dtype)
                    << ", device=" << signature.device.str();
            return stream.str();
        }

        struct BatchSignature {
            GraphTensorSignature inputs{};
            GraphTensorSignature targets{};

            [[nodiscard]] static BatchSignature
            from_tensors(const torch::Tensor &inputs, const torch::Tensor &targets) {
                BatchSignature signature{};
                signature.inputs = describe_tensor_signature(inputs);
                signature.targets = describe_tensor_signature(targets);
                return signature;
            }

            [[nodiscard]] bool matches(const torch::Tensor &inputs, const torch::Tensor &targets) const {
                return tensor_matches_signature(inputs, this->inputs)
                       && tensor_matches_signature(targets, this->targets);
            }

            bool operator==(const BatchSignature &other) const noexcept {
                return signatures_equal(inputs, other.inputs) && signatures_equal(targets, other.targets);
            }

            struct Hash {
                std::size_t operator()(const BatchSignature &signature) const noexcept {
                    std::size_t seed = 0;
                    BatchSignature::hash_signature(seed, signature.inputs);
                    BatchSignature::hash_signature(seed, signature.targets);
                    return seed;
                }
            };

        private:
            friend struct Hash;

            static void hash_combine(std::size_t &seed, std::size_t value) noexcept {
                seed ^= value + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            }

            static void hash_signature(std::size_t &seed, const GraphTensorSignature &signature) noexcept {
                hash_combine(seed, static_cast<std::size_t>(signature.dtype));
                hash_combine(seed, static_cast<std::size_t>(signature.device.type()));
                hash_combine(seed, static_cast<std::size_t>(signature.device.index()));
                hash_combine(seed, signature.shape.size());
                for (const auto dimension: signature.shape) {
                    hash_combine(seed, static_cast<std::size_t>(dimension));
                }
            }

            static bool tensor_matches_signature(const torch::Tensor &tensor, const GraphTensorSignature &signature) {
                if (tensor.device() != signature.device || tensor.scalar_type() != signature.dtype) {
                    return false;
                }

                const auto dimensions = tensor.dim();
                if (static_cast<std::size_t>(dimensions) != signature.shape.size()) {
                    return false;
                }

                for (std::int64_t index = 0; index < dimensions; ++index) {
                    if (tensor.size(index) != signature.shape[index]) {
                        return false;
                    }
                }

                return true;
            }
        };

        void ensure_graph_regularization_metadata_capacity(GraphMode mode) const {
            if (mode == GraphMode::Disabled) {
                return;
            }

            bool needs_reset = graph_regularization_metadata_dirty_
                               || graph_global_regularization_metadata_.size() != global_regularization_bindings_.size()
                               || graph_layer_regularization_metadata_.size() != layer_regularization_bindings_.size();

            if (!needs_reset) {
                for (std::size_t index = 0; index < graph_layer_regularization_metadata_.size(); ++index) {
                    if (index >= layer_regularization_bindings_.size()) {
                        needs_reset = true;
                        break;
                    }
                    if (graph_layer_regularization_metadata_[index].size()
                        != layer_regularization_bindings_[index].size()) {
                        needs_reset = true;
                        break;
                    }
                }
            }

            if (needs_reset) {
                graph_global_regularization_metadata_.assign(global_regularization_bindings_.size(), {});
                graph_layer_regularization_metadata_.resize(layer_regularization_bindings_.size());
                for (std::size_t index = 0; index < layer_regularization_bindings_.size(); ++index) {
                    graph_layer_regularization_metadata_[index].
                            assign(layer_regularization_bindings_[index].size(), {});
                }
                graph_regularization_metadata_dirty_ = false;
            }
        }

        void ensure_graph_calibration_metadata_capacity(GraphMode mode) const {
            if (mode == GraphMode::Disabled) {
                return;
            }

            if (graph_calibration_metadata_dirty_
                || graph_calibration_metadata_.size() != calibration_methods_.size()) {
                graph_calibration_metadata_.assign(calibration_methods_.size(), {});
                graph_calibration_metadata_dirty_ = false;
            }
        }

        void mark_graph_regularization_metadata_dirty() const noexcept {
            graph_regularization_metadata_dirty_ = true;
        }

        void mark_graph_calibration_metadata_dirty() const noexcept {
            graph_calibration_metadata_dirty_ = true;
        }

        static std::string describe_activation(Activation::Type type);

        static std::string describe_module(const Layer::Details::RegisteredLayer &layer);

        enum class WorkspaceTensorPolicy {
            RebindStorage,
            PreserveStorage
        };

        static WorkspaceTensorPolicy workspace_tensor_policy(GraphMode mode) noexcept {
            switch (mode) {
                case GraphMode::Capture:
                case GraphMode::Replay:
                    return WorkspaceTensorPolicy::PreserveStorage;
                case GraphMode::Disabled:
                default:
                    return WorkspaceTensorPolicy::RebindStorage;
            }
        }

        static void copy_tensor_into(
            torch::Tensor &destination,
            const torch::Tensor &source,
            WorkspaceTensorPolicy policy = WorkspaceTensorPolicy::RebindStorage) {
            if (!source.defined()) {
                destination = torch::Tensor{};
                return;
            }

            if (!destination.defined()) {
                destination = source.clone(torch::MemoryFormat::Preserve);
                return;
            }


            if (policy == WorkspaceTensorPolicy::RebindStorage) {
                destination = source.clone(torch::MemoryFormat::Preserve);
                return;
            }

            if (destination.is_alias_of(source)) {
                if (destination.requires_grad() != source.requires_grad()) {
                    destination.requires_grad_(source.requires_grad());
                }
                return;
            }

            if (destination.device() != source.device()) {
                throw std::invalid_argument(
                    "copy_tensor_into requires destination and source to share the same device.");
            }

            if (destination.dtype() != source.dtype()) {
                throw std::invalid_argument(
                    "copy_tensor_into requires destination and source to share the same dtype.");
            }

            if (destination.sizes() != source.sizes()) {
                throw std::invalid_argument(
                    "copy_tensor_into requires destination and source to share the same shape.");
            }

            destination = destination.detach();
            destination.requires_grad_(source.requires_grad());
            destination.copy_(source);
        }

        void record_epoch_telemetry(TrainingTelemetry::EpochSnapshot snapshot) {
            telemetry_.append_epoch(std::move(snapshot));
        }

        void record_dataset_loss_telemetry(TrainingTelemetry::DatasetLossSnapshot snapshot) {
            telemetry_.append_dataset_loss(std::move(snapshot));
        }

        [[nodiscard]] std::vector<double> collect_learning_rates() {
            std::vector<double> learning_rates;

            auto append_from = [&](torch::optim::Optimizer *optimizer) {
                if (!optimizer) {
                    return;
                }
                for (auto &group: optimizer->param_groups()) {
                    learning_rates.push_back(group.options().get_lr());
                }
            };

            if (optimizer_) {
                append_from(optimizer_->instance.get());
            }
            for (auto &optimizer: local_optimizers_) {
                append_from(optimizer.instance.get());
            }

            return learning_rates;
        }

        [[nodiscard]] torch::Tensor ensure_input_memory_format(torch::Tensor tensor) const {
            if (!tensor.defined()) {
                return tensor;
            }

            auto ensure_contiguous = [&](torch::MemoryFormat format, int64_t min_dim) {
                if (tensor.dim() >= min_dim) {
                    if (!tensor.is_contiguous(format)) {
                        tensor = tensor.contiguous(format);
                    }
                } else if (!tensor.is_contiguous()) {
                    tensor = tensor.contiguous();
                }
            };

            switch (tensor_memory_format_) {
                case torch::MemoryFormat::ChannelsLast:
                    ensure_contiguous(torch::MemoryFormat::ChannelsLast, /*min_dim=*/4);
                    break;
                case torch::MemoryFormat::ChannelsLast3d:
                    ensure_contiguous(torch::MemoryFormat::ChannelsLast3d, /*min_dim=*/5);
                    break;
                default:
                    if (!tensor.is_contiguous()) {
                        tensor = tensor.contiguous();
                    }
                    break;
            }

            return tensor;
        }

        [[nodiscard]] torch::Tensor stage_tensor_for_execution(torch::Tensor tensor) const {
            if (!tensor.defined()) {
                return tensor;
            }

            tensor = ensure_input_memory_format(std::move(tensor));

            if (tensor.device().is_cpu() && device_.is_cuda() && !tensor.is_pinned()) {
                tensor = tensor.pin_memory();
            }

            if (tensor.device().is_cpu() && staging_observer_) {
                staging_observer_(tensor, device_.is_cuda());
            }

            if (tensor.device() != device_) {
                tensor = tensor.to(device_, /*non_blocking=*/device_.is_cuda());
                tensor = ensure_input_memory_format(std::move(tensor));
            }

            return tensor;
        }


        void copy_into_graph_input_buffer(torch::Tensor tensor, WorkspaceTensorPolicy policy) {
            constexpr std::size_t kInputNodeIndex = 0;
            copy_tensor_into(graph_workspace_.input, tensor, policy);
            graph_workspace_.bind_input(kInputNodeIndex);
        }

        [[nodiscard]] const torch::Tensor &graph_output_tensor() const noexcept {
            return graph_workspace_.output;
        }

        [[nodiscard]] std::size_t resolve_output_node_index() const noexcept {
            constexpr std::size_t kInputNodeIndex = 0;
            if (compiled_output_node_index_) {
                return *compiled_output_node_index_;
            }
            if (execution_steps_.empty()) {
                return kInputNodeIndex;
            }
            return execution_steps_.back().activation_index;
        }


        void clear_compiled_graph() noexcept {
            routing_active_ = false;
            compiled_nodes_.clear();
            compiled_steps_.clear();
            execution_steps_.clear();
            join_buffers_.clear();
            compiled_links_.clear();
            compiled_output_node_index_.reset();
            graph_capture_opt_in_ = false;
            invalidate_execution_workspace();
        }

        template<typename ConvolutionImpl>
        void apply_tensor_memory_format_to_convolution(ConvolutionImpl *convolution) {
            if (!convolution) {
                return;
            }

            auto apply_to_parameter = [&](torch::Tensor &parameter) {
                if (!parameter.defined()) {
                    return;
                }

                auto memory_format = tensor_memory_format_;
                if (tensor_memory_format_ == torch::MemoryFormat::ChannelsLast && parameter.dim() < 4) {
                    memory_format = torch::MemoryFormat::Contiguous;
                } else if (
                    tensor_memory_format_ == torch::MemoryFormat::ChannelsLast3d && parameter.dim() < 5) {
                    memory_format = torch::MemoryFormat::Contiguous;
                }
                parameter = parameter.to(
                    parameter.options(), /*non_blocking*/false, /*copy*/false, memory_format);
            };

            apply_to_parameter(convolution->weight);
            if (convolution->bias.defined()) {
                apply_to_parameter(convolution->bias);
            }
        }


        void register_layer_runtime(const Layer::Details::RegisteredLayer &layer) {
            layer_parameters_.push_back(collect_layer_parameters(layer));
            layer_regularization_bindings_.push_back(
                bind_local_regularization(layer.local.regularization, layer_parameters_.back()));
            mark_graph_regularization_metadata_dirty();
            invalidate_execution_workspace();
            if (layer.module) {
                if (auto *conv1d = dynamic_cast<torch::nn::Conv1dImpl *>(layer.module.get())) {
                    has_convolutional_layers_ = true;
                    apply_tensor_memory_format_to_convolution(conv1d);
                } else if (auto *conv2d = dynamic_cast<torch::nn::Conv2dImpl *>(layer.module.get())) {
                    has_convolutional_layers_ = true;
                    apply_tensor_memory_format_to_convolution(conv2d);
                }
            }
        }

        void refresh_layer_parameter_cache() {
            if (layer_parameters_.size() != layers_.size()) {
                layer_parameters_.resize(layers_.size());
            }

            for (std::size_t index = 0; index < layers_.size(); ++index) {
                layer_parameters_[index] = collect_layer_parameters(layers_[index]);
            }

            bool has_local_regularization = std::any_of(
                layer_regularization_bindings_.begin(),
                layer_regularization_bindings_.end(),
                [](const auto &bindings) { return !bindings.empty(); });

            if (regularization_configured_ || has_local_regularization) {
                global_regularization_parameters_ = collect_global_trainable_parameters();
                mark_graph_regularization_metadata_dirty();
            }
        }

        void set_tensor_memory_format(torch::MemoryFormat format) {
            if (tensor_memory_format_ == format) {
                return;
            }
            tensor_memory_format_ = format;
            apply_tensor_memory_format_to_convolutions();
        }

        void apply_tensor_memory_format_to_convolutions() {
            if (!has_convolutional_layers_) {
                return;
            }

            for (auto &layer: layers_) {
                if (!layer.module) {
                    continue;
                }
                if (auto *conv1d = dynamic_cast<torch::nn::Conv1dImpl *>(layer.module.get())) {
                    apply_tensor_memory_format_to_convolution(conv1d);
                } else if (auto *conv2d = dynamic_cast<torch::nn::Conv2dImpl *>(layer.module.get())) {
                    apply_tensor_memory_format_to_convolution(conv2d);
                }
            }
        }

        void invalidate_execution_workspace() noexcept {
            execution_workspace_dirty_ = true;
            graph_workspace_.invalidate();
            cached_layer_pointers_.clear();
            invalidate_graph_captures();
        }

        [[nodiscard]] GraphCaptureState &graph_capture_state(GraphExecutionPhase phase) noexcept {
            switch (phase) {
                case GraphExecutionPhase::Training:
                    return graph_capture_training_;
                case GraphExecutionPhase::Inference:
                    return graph_capture_inference_;
            }
            return graph_capture_training_;
        }

        [[nodiscard]] const GraphCaptureState &graph_capture_state(GraphExecutionPhase phase) const noexcept {
            switch (phase) {
                case GraphExecutionPhase::Training:
                    return graph_capture_training_;
                case GraphExecutionPhase::Inference:
                    return graph_capture_inference_;
            }
            return graph_capture_training_;
        }

        void invalidate_graph_capture(GraphExecutionPhase phase) noexcept {
            auto &state = graph_capture_state(phase);
#ifdef TORCH_CUDA_AVAILABLE
            state.graph.reset();
            state.capture_stream.reset();
#endif
            state.captured = false;
            state.dirty = true;
            state.loss_buffer = torch::Tensor{};
            state.target_buffer = torch::Tensor{};
        }

        void invalidate_graph_captures() noexcept {
            invalidate_graph_capture(GraphExecutionPhase::Training);
            invalidate_graph_capture(GraphExecutionPhase::Inference);
        }

        [[nodiscard]] bool graph_execution_enabled(GraphMode mode, GraphExecutionPhase phase) const noexcept {
            if (mode == GraphMode::Disabled) {
                return false;
            }
            if (!graph_capture_opt_in_ || !routing_active_) {
                return false;
            }
            if (!device_.is_cuda()) {
                return false;
            }
#ifdef TORCH_CUDA_AVAILABLE
            if (!torch::cuda::is_available()) {
                return false;
            }
            (void) phase;
            return true;
#else
            (void) phase;
            return false;
#endif
        }

        void ensure_execution_workspace() {
            if (execution_workspace_dirty_ || cached_layer_pointers_.size() != layers_.size()) {
                cached_layer_pointers_.resize(layers_.size());
                for (std::size_t index = 0; index < layers_.size(); ++index) {
                    cached_layer_pointers_[index] = &layers_[index];
                }
            }

            if (!has_compiled_routing() || execution_steps_.empty()) {
                execution_workspace_dirty_ = false;
                return;
            }

            auto &workspace = graph_workspace_;

            constexpr std::size_t kInputNodeIndex = 0;

            auto required_capacity = kInputNodeIndex + 1;
            auto consider_index = [&](std::size_t index) {
                required_capacity = std::max(required_capacity, index + 1);
            };

            for (const auto &step: execution_steps_) {
                consider_index(step.activation_index);
                switch (step.kind) {
                    case ExecutionStep::Kind::Module: {
                        consider_index(step.module.input_index);
                        break;
                    }
                    case ExecutionStep::Kind::Join: {
                        for (auto producer: step.join.producers) {
                            consider_index(producer);
                        }
                        break;
                    }
                    case ExecutionStep::Kind::Output: {
                        consider_index(step.output.input_index);
                        break;
                    }
                }
            }

            if (execution_workspace_dirty_ || workspace.node_buffers.size() != required_capacity) {
                workspace.ensure_node_capacity(required_capacity);
            }

            workspace.ensure_join_scratch(join_buffers_);

            workspace.bind_input(kInputNodeIndex);

            const auto output_index = resolve_output_node_index();
#ifndef NDEBUG
            assert(output_index < workspace.node_buffers.size());
#endif
            workspace.bind_output(output_index);

            for (auto &scratch: workspace.join_scratch) {
                scratch.clear();
            }

            execution_workspace_dirty_ = false;
        }

        [[nodiscard]] Layer::Details::RegisteredLayer *resolve_registered_layer(torch::nn::Module *module) noexcept {
            if (module == nullptr) {
                return nullptr;
            }

            for (auto &layer: layers_) {
                if (layer.module && layer.module.get() == module) {
                    return &layer;
                }
            }
            return nullptr;
        }

        static std::vector<torch::Tensor> collect_layer_parameters(const Layer::Details::RegisteredLayer &layer) {
            std::vector<torch::Tensor> parameters;
            if (!layer.module) {
                return parameters;
            }

            for (auto &parameter: layer.module->parameters()) {
                if (parameter.requires_grad()) {
                    parameters.push_back(parameter);
                }
            }

            return parameters;
        }

        std::vector<torch::Tensor> collect_global_trainable_parameters() const {
            std::vector<torch::Tensor> parameters;
            for (auto &parameter: this->parameters()) {
                if (parameter.requires_grad()) {
                    parameters.push_back(parameter);
                }
            }
            return parameters;
        }

        RegularizationStateStorage prepare_regularization_states(const Regularization::Descriptor &descriptor,
                                                                 const std::vector<torch::Tensor> &parameters) const {
            return std::visit(
                [&](const auto &concrete_descriptor) -> RegularizationStateStorage {
                    using DescriptorType = std::decay_t<decltype(concrete_descriptor)>;

                    if constexpr (std::is_same_v<DescriptorType, Regularization::EWCDescriptor>
                                  || std::is_same_v<DescriptorType, Regularization::MASDescriptor>
                                  || std::is_same_v<DescriptorType, Regularization::SIDescriptor>) {
                        auto storage = std::make_shared<std::vector<RegularizationState> >();
                        storage->reserve(parameters.size());
                        for (const auto &parameter: parameters) {
                            if constexpr (std::is_same_v<DescriptorType, Regularization::EWCDescriptor>) {
                                Regularization::Details::EWCState state{};
                                state.reference = parameter.detach().clone(torch::MemoryFormat::Preserve);
                                state.fisher_information = torch::zeros_like(parameter);
                                storage->emplace_back(std::move(state));
                            } else if constexpr (std::is_same_v<DescriptorType, Regularization::MASDescriptor>) {
                                Regularization::Details::MASState state{};
                                state.reference = parameter.detach().clone(torch::MemoryFormat::Preserve);
                                state.importance = torch::zeros_like(parameter);
                                storage->emplace_back(std::move(state));
                            } else if constexpr (std::is_same_v<DescriptorType, Regularization::SIDescriptor>) {
                                Regularization::Details::SIState state{};
                                state.reference = parameter.detach().clone(torch::MemoryFormat::Preserve);
                                state.importance = torch::zeros_like(parameter);
                                storage->emplace_back(std::move(state));
                            }
                        }
                        return storage;
                    } else if constexpr (std::is_same_v<DescriptorType, Regularization::SWAGDescriptor>) {
                        auto storage = std::make_shared<std::vector<RegularizationState> >();
                        storage->reserve(parameters.size());
                        for (std::size_t index = 0; index < parameters.size(); ++index) {
                            storage->emplace_back(Regularization::Details::SWAGState{});
                        }
                        return storage;
                    } else {
                        return {};
                    }
                },
                descriptor);
        }

        RegularizationBinding make_regularization_binding(Regularization::Descriptor descriptor,
                                                          const std::vector<torch::Tensor> &parameters) const {
            RegularizationBinding binding{};
            binding.descriptor = std::move(descriptor);
            binding.states = prepare_regularization_states(binding.descriptor, parameters);
            binding.accumulator = Regularization::bind_accumulator(binding.descriptor, binding.states);
            return binding;
        }

        std::vector<RegularizationBinding> bind_local_regularization(
            const std::vector<Regularization::Descriptor> &descriptors,
            const std::vector<torch::Tensor> &parameters) const {
            std::vector<RegularizationBinding> bindings;
            bindings.reserve(descriptors.size());
            for (const auto &descriptor: descriptors) {
                bindings.push_back(make_regularization_binding(descriptor, parameters));
            }
            return bindings;
        }

        void update_regularization_binding_states(RegularizationBinding &binding,
                                                  const std::vector<torch::Tensor> &parameters,
                                                  std::size_t step_index) {
            if (parameters.empty()) {
                return;
            }

            std::visit(
                [&](auto &concrete_descriptor) {
                    using DescriptorType = std::decay_t<decltype(concrete_descriptor)>;
                    if constexpr (std::is_same_v<DescriptorType, Regularization::SWAGDescriptor>) {
                        const auto &options = concrete_descriptor.options;
                        if (options.coefficient == 0.0) {
                            return;
                        }

                        const std::size_t stride = options.accumulation_stride == 0
                                                       ? std::size_t{1}
                                                       : options.accumulation_stride;
                        if (step_index < options.start_step) {
                            return;
                        }
                        const auto adjusted_step = step_index - options.start_step;
                        if (adjusted_step % stride != 0) {
                            return;
                        }
                        if (!binding.states) {
                            return;
                        }

                        auto &state_storage = *binding.states;
                        const auto limit = std::min(parameters.size(), state_storage.size());
                        for (std::size_t index = 0; index < limit; ++index) {
                            auto &state_variant = state_storage[index];
                            if (!std::holds_alternative<Regularization::Details::SWAGState>(state_variant)) {
                                continue;
                            }

                            auto &state = std::get<Regularization::Details::SWAGState>(state_variant);
                            if (options.max_snapshots > 0 && state.snapshot_count >= options.max_snapshots) {
                                continue;
                            }

                            auto snapshot = parameters[index].detach();
                            if (!snapshot.defined()) {
                                continue;
                            }

                            auto snapshot_tensor = snapshot.clone(torch::MemoryFormat::Preserve);
                            if (!snapshot_tensor.defined()) {
                                continue;
                            }

                            if (state.snapshot_count == 0 || !state.mean.defined()) {
                                state.mean = snapshot_tensor;
                                state.variance = torch::zeros_like(state.mean);
                                state.snapshot_count = 1;
                                continue;
                            }

                            if (!state.variance.defined()) {
                                state.variance = torch::zeros_like(state.mean);
                            }

                            if (snapshot_tensor.device() != state.mean.device()) {
                                snapshot_tensor = snapshot_tensor.to(state.mean.device());
                            }
                            if (snapshot_tensor.scalar_type() != state.mean.scalar_type()) {
                                snapshot_tensor = snapshot_tensor.to(state.mean.scalar_type());
                            }

                            auto delta = snapshot_tensor - state.mean;
                            const double next_count = static_cast<double>(state.snapshot_count + 1);
                            state.mean = state.mean + delta / next_count;
                            auto delta2 = snapshot_tensor - state.mean;
                            state.variance = state.variance + delta * delta2;
                            state.snapshot_count += 1;
                        }
                    }
                },
                binding.descriptor);
        }

        void update_regularization_states(std::size_t step_index, bool regularization_active = false) {
            if (!regularization_active && !has_regularization()) {
                return;
            }

            for (auto &binding: global_regularization_bindings_) {
                update_regularization_binding_states(binding, global_regularization_parameters_, step_index);
            }

            for (std::size_t index = 0; index < layer_regularization_bindings_.size(); ++index) {
                auto &bindings = layer_regularization_bindings_[index];
                auto &parameters = layer_parameters_[index];
                for (auto &binding: bindings) {
                    update_regularization_binding_states(binding, parameters, step_index);
                }
            }
        }

        torch::Tensor graph_train_step_impl(torch::Tensor batch_inputs, torch::Tensor batch_targets,
                                            GraphMode graph_mode, bool regularization_active, bool amp_enabled) {
            const auto phase = GraphExecutionPhase::Training;
            if (!graph_execution_enabled(graph_mode, phase)) {
                graph_mode = GraphMode::Disabled;
            }
            if (graph_mode != GraphMode::Disabled) {
                ensure_optimizer_graph_capability(graph_mode);
            }
            auto &state = graph_capture_state(phase);
            if (graph_mode != GraphMode::Disabled) {
                prepare_optimizers_for_graph(graph_mode);
            }


#ifdef TORCH_CUDA_AVAILABLE
            const bool use_amp = amp_enabled && device_.is_cuda();
            if (use_amp) {
                ensure_amp_scaler();
            }
#else
            (void) amp_enabled;
            const bool use_amp = false;
#endif
            const auto autocast_device_type = device_.type();
            const auto autocast_dtype = use_amp ? determine_autocast_dtype() : torch::kFloat32;
            auto run_training_step = [&](GraphMode mode, torch::Tensor inputs, torch::Tensor targets) {
                if (mode == GraphMode::Capture) {
                    targets = state.target_buffer;
                }

                if (mode != GraphMode::Disabled) {
                    ensure_graph_input_shape(mode, inputs);
                }
                torch::Tensor prediction;
                torch::Tensor loss;

                {
                    AutocastGuard autocast_guard(use_amp, autocast_device_type, autocast_dtype);
                    prediction = execute_plan(std::move(inputs), mode);

                    if (!prediction.sizes().equals(targets.sizes())) {
                        if (targets.numel() == prediction.numel()) {
                            targets = targets.reshape_as(prediction);
                        }
                    }

                    if (mode != GraphMode::Disabled) {
                        enforce_graph_shape(mode, targets, graph_target_shape_cache_, "target tensor");
                        auto detached_targets = targets.detach();
                        detached_targets.requires_grad_(false);
                        copy_tensor_into(
                            state.target_buffer,
                            detached_targets,
                            workspace_tensor_policy(mode));
                        targets = state.target_buffer;
                    }

                    loss = compute_loss(prediction, targets);
                    if (loss.dim() != 0) {
                        loss = loss.mean();
                    }


                    if (regularization_active) {
                        auto regularization_penalty = compute_regularization_penalty(mode);
                        if (regularization_penalty.defined()) {
                            if (mode == GraphMode::Disabled) {
                                if (regularization_penalty.device() != loss.device()) {
                                    regularization_penalty = regularization_penalty.to(loss.device());
                                }
                                if (regularization_penalty.scalar_type() != loss.scalar_type()) {
                                    regularization_penalty = regularization_penalty.to(loss.scalar_type());
                                }
                            } else {
                                if (regularization_penalty.device() != loss.device()) {
                                    throw std::runtime_error(
                                        "Regularisation penalty device changed during CUDA graph execution.");
                                }
                                if (regularization_penalty.scalar_type() != loss.scalar_type()) {
                                    throw std::runtime_error(
                                        "Regularisation penalty dtype changed during CUDA graph execution.");
                                }
                            }
                            loss = loss + regularization_penalty;
                        }
                    }
                }

                const bool retain_graph = (mode != GraphMode::Disabled);
#ifdef TORCH_CUDA_AVAILABLE
                if (use_amp) {
                    auto scaled_loss = amp_scaler_->scale(loss);
                    scaled_loss.backward({}, retain_graph);
                    step_optimizers_with_scaler(*amp_scaler_);
                    amp_scaler_->update();
                } else
#endif
                {
                    loss.backward({}, retain_graph);
                    step_optimizers();
                }

                zero_grad();

                loss.detach_();
                return loss;
            };

            switch (graph_mode) {
                case GraphMode::Disabled:
                    return run_training_step(GraphMode::Disabled, std::move(batch_inputs), std::move(batch_targets));
                case GraphMode::Capture: {
#ifdef TORCH_CUDA_AVAILABLE
                    if (state.dirty) {
                        reset_graph_shape_cache(GraphMode::Capture);
                    }

      } else {
                        state.graph->reset();
                    }


                    if (!state.capture_stream.has_value()) {
                        state.capture_stream = torch::cuda::getStreamFromPool();
                    }
                    ensure_execution_workspace();
                    ensure_graph_input_shape(GraphMode::Capture, batch_inputs);
                    copy_into_graph_input_buffer(std::move(batch_inputs), workspace_tensor_policy(GraphMode::Capture));
                    batch_inputs = graph_workspace_.input;

                    state.target_buffer = batch_targets.detach();
                    state.target_buffer.requires_grad_(false);
                    batch_targets = state.target_buffer;

                    state.captured = false;
                    torch::cuda::CUDAStreamGuard guard(*state.capture_stream);

                    bool capture_started = false;
                    try {
                        state.graph->capture_begin(*state.capture_stream);
                        capture_started = true;
                        auto loss = run_training_step(GraphMode::Capture, std::move(batch_inputs),
                                                      std::move(batch_targets));
                        state.graph->capture_end();
                        capture_started = false;
                        state.loss_buffer = loss.detach();
                        state.loss_buffer.requires_grad_(false);
                        loss.detach_();
                        state.captured = true;
                        state.dirty = false;
#ifdef TORCH_CUDA_AVAILABLE
                    if (use_amp && amp_scaler_) {
                        state.amp_scaler_state = amp_scaler_->state_dict();
                        state.amp_scaler_state_valid = true;
                    } else {
                        state.amp_scaler_state.clear();
                        state.amp_scaler_state_valid = false;
                    }
#endif
                    return state.loss_buffer;
        } catch (...) {
                        if (capture_started) {
                            try {
                                state.graph->capture_end();
                            } catch (...) {
                            }
                        }
                        state.graph->reset();
                        state.capture_stream.reset();
                        state.captured = false;
                        state.dirty = true;
                        state.loss_buffer = torch::Tensor{};
#ifdef TORCH_CUDA_AVAILABLE
                    state.amp_scaler_state.clear();
                    state.amp_scaler_state_valid = false;
#endif
                    throw;
        }
#else
                    throw std::runtime_error("CUDA graph capture requested but CUDA support is unavailable.");
#endif
                }
                case GraphMode::Replay: {
#ifdef TORCH_CUDA_AVAILABLE
                    if (!state.captured || state.dirty || !state.graph) {
                        throw std::runtime_error(
                            "CUDA graph replay requested for training before a capture was recorded.");
                    }
                    ensure_graph_input_shape(GraphMode::Replay, batch_inputs);
                    if (graph_target_shape_cache_) {
                        batch_targets = batch_targets.reshape(*graph_target_shape_cache_);
                    }
                    enforce_graph_shape(GraphMode::Replay, batch_targets, graph_target_shape_cache_, "target tensor");
                    ensure_execution_workspace();
                    copy_into_graph_input_buffer(std::move(batch_inputs), workspace_tensor_policy(GraphMode::Replay));
                    auto detached_targets = batch_targets.detach();
                    detached_targets.requires_grad_(false);
                    copy_tensor_into(
                        state.target_buffer,
                        detached_targets,
                        workspace_tensor_policy(GraphMode::Replay));
                    if (!state.capture_stream.has_value()) {
                        throw std::runtime_error(
                            "CUDA graph replay requested for training without an associated capture stream.");
                    }
      }
                    torch::cuda::CUDAStreamGuard guard(*state.capture_stream);
#ifdef TORCH_CUDA_AVAILABLE
                    if (use_amp && amp_scaler_ && state.amp_scaler_state_valid) {
                        amp_scaler_->load_state_dict(state.amp_scaler_state);
                    }
#endif
                    state.graph->replay();
#ifdef TORCH_CUDA_AVAILABLE
                    if (use_amp && amp_scaler_) {
                        state.amp_scaler_state = amp_scaler_->state_dict();
                        state.amp_scaler_state_valid = true;
                    }
#endif
                    return state.loss_buffer;
#else
                    throw std::runtime_error("CUDA graph replay requested but CUDA support is unavailable.");
#endif
                }
            }
            return torch::Tensor{};
        }

        void step_scheduler() {
            if (scheduler_) {
                scheduler_->step();
            }
        }


        struct TrainingDetails {
            struct TensorDataset {
                torch::Tensor inputs;
                torch::Tensor targets;
            };

            struct TrainingStepBinding {
                using Impl = torch::Tensor (Model::*)(torch::Tensor, torch::Tensor, GraphMode, bool, bool);

                Impl impl{&Model::graph_train_step_impl};

                [[nodiscard]] torch::Tensor operator()(Model &model, torch::Tensor inputs, torch::Tensor targets, GraphMode mode, bool regularization_active, bool amp_enabled) const {
                    return (model.*impl)(std::move(inputs), std::move(targets), mode, regularization_active, amp_enabled);
                }
            };

            [[nodiscard]] static TensorDataset prepare_tensor_dataset(torch::Tensor inputs, torch::Tensor targets,
                                                                      torch::MemoryFormat memory_format =
                                                                              torch::MemoryFormat::Contiguous) {
                auto prepared_inputs = std::move(inputs);
                if (memory_format == torch::MemoryFormat::ChannelsLast && prepared_inputs.defined()
                    && prepared_inputs.dim() >= 4) {
                    if (!prepared_inputs.is_contiguous(torch::MemoryFormat::ChannelsLast)) {
                        prepared_inputs = prepared_inputs.contiguous(torch::MemoryFormat::ChannelsLast);
                    }
                } else {
                    prepared_inputs = prepared_inputs.contiguous();
                }
                auto prepared_targets = std::move(targets).contiguous();
                if (prepared_inputs.device().is_cpu() && !prepared_inputs.is_pinned()) {
                    prepared_inputs = prepared_inputs.pin_memory();
                }
                if (prepared_targets.device().is_cpu() && !prepared_targets.is_pinned()) {
                    prepared_targets = prepared_targets.pin_memory();
                }
                return TensorDataset{std::move(prepared_inputs), std::move(prepared_targets)};
            }

            [[nodiscard]] static TensorDataset ensure_contiguous(TensorDataset dataset,
                                                                 torch::MemoryFormat memory_format =
                                                                         torch::MemoryFormat::Contiguous) {
                if (dataset.inputs.defined()) {
                    if (memory_format == torch::MemoryFormat::ChannelsLast && dataset.inputs.dim() >= 4) {
                        if (!dataset.inputs.is_contiguous(torch::MemoryFormat::ChannelsLast)) {
                            dataset.inputs = dataset.inputs.contiguous(torch::MemoryFormat::ChannelsLast);
                        }
                    } else if (!dataset.inputs.is_contiguous()) {
                        dataset.inputs = dataset.inputs.contiguous();
                    }
                }
                dataset.targets = dataset.targets.contiguous();
                return dataset;
            }

            [[nodiscard]] static TensorDataset ensure_cpu(TensorDataset dataset,
                                                          torch::MemoryFormat memory_format =
                                                                  torch::MemoryFormat::Contiguous) {
                if (!dataset.inputs.device().is_cpu()) {
                    if (memory_format == torch::MemoryFormat::ChannelsLast && dataset.inputs.dim() >= 4) {
                        auto options = dataset.inputs.options().device(torch::kCPU);
                        dataset.inputs = dataset.inputs.to(options, /*non_blocking*/false, /*copy*/false,
                                                           torch::MemoryFormat::ChannelsLast);
                    } else {
                        dataset.inputs = dataset.inputs.to(torch::kCPU);
                    }
                }
                if (!dataset.targets.device().is_cpu()) {
                    dataset.targets = dataset.targets.to(torch::kCPU);
                }
                if (dataset.inputs.device().is_cpu() && !dataset.inputs.is_pinned()) {
                    dataset.inputs = dataset.inputs.pin_memory();
                }
                if (dataset.targets.device().is_cpu() && !dataset.targets.is_pinned()) {
                    dataset.targets = dataset.targets.pin_memory();
                }
                return dataset;
            }

            template<class Dataset>
            [[nodiscard]] static TensorDataset pack_dataset(Dataset dataset) {
                if (dataset.empty()) {
                    return {};
                }

                std::vector<torch::Tensor> inputs;
                std::vector<torch::Tensor> targets;
                inputs.reserve(dataset.size());
                targets.reserve(dataset.size());

                for (auto &sample: dataset) {
                    inputs.push_back(std::move(sample.first));
                    targets.push_back(std::move(sample.second));
                }

                return TensorDataset{torch::stack(std::move(inputs)), torch::stack(std::move(targets))};
            }

            template<bool BufferVRAM, bool ShouldShuffle>
            static void run_epochs(Model &model, TensorDataset &train_dataset,
                                   const std::optional<TensorDataset> &test_dataset, const TrainOptions &options,
                                   const TrainingStepBinding &training_step) {
                const auto device = model.device();
                const auto total_samples = train_dataset.inputs.size(0);
                const auto batch_size = static_cast<std::int64_t>(options.batch_size);
                const auto requested_graph_mode = options.graph_mode;
                const bool graph_mode_active = model.graph_execution_enabled(
                    requested_graph_mode, GraphExecutionPhase::Training);
                const auto graph_mode = graph_mode_active ? requested_graph_mode : GraphMode::Disabled;
                const bool graph_mode_enabled = graph_mode != GraphMode::Disabled;
                const bool amp_enabled = model.is_amp_training_active();

                if (graph_mode == GraphMode::Capture) {
                    model.reset_graph_shape_cache(graph_mode);
                } else if (graph_mode == GraphMode::Replay) {
                    model.ensure_graph_replay_ready(graph_mode);
                }

                if (graph_mode_enabled) {
                    model.ensure_optimizer_graph_capability(graph_mode);
                }

                torch::TensorOptions index_options = torch::TensorOptions().dtype(torch::kLong);

                if (train_dataset.inputs.device().is_cpu() && !train_dataset.inputs.is_pinned()) {
                    train_dataset.inputs = train_dataset.inputs.pin_memory();
                }
                if (train_dataset.targets.device().is_cpu() && !train_dataset.targets.is_pinned()) {
                    train_dataset.targets = train_dataset.targets.pin_memory();
                }


                auto best_test = std::optional<double>{};
                std::vector<torch::Tensor> best_parameters;
                std::vector<torch::Tensor> best_buffers;
                bool best_state_captured = false;

                std::size_t step_index = 0;

                const bool regularization_active = model.has_regularization();

                std::unordered_map<BatchSignature, bool, BatchSignature::Hash> graph_capture_ready{};
                std::unordered_map<std::int64_t, BatchSignature> batch_signature_cache{};
                std::optional<BatchSignature> active_graph_signature{};

                struct PendingEpochLog {
                    std::size_t epoch_index{};
                    std::size_t total_epochs{};
                    TrainingTelemetry::DeferredScalar train_loss{};
                    std::optional<double> test_loss{};
                    std::optional<double> delta{};
                    bool improved{false};
                    double duration_seconds{0.0};
                };

                std::deque<PendingEpochLog> pending_epoch_logs{};

                auto flush_pending_epoch_logs = [&](bool drain) {
                    if (!options.monitor || options.stream == nullptr) {
                        return;
                    }

                    while (!pending_epoch_logs.empty()) {
                        auto &front = pending_epoch_logs.front();
                        if (!front.train_loss.is_ready()) {
                            if (!drain) {
                                break;
                            }
                            (void) front.train_loss.materialize();
                        }

                        const auto train_loss_value = front.train_loss.materialize();
                        log_epoch(*options.stream,
                                  front.epoch_index,
                                  front.total_epochs,
                                  train_loss_value,
                                  front.test_loss,
                                  front.delta,
                                  front.improved,
                                  front.duration_seconds);
                        pending_epoch_logs.pop_front();
                    }
                };


                torch::Tensor device_batch_inputs_buffer;
                torch::Tensor device_batch_targets_buffer;
                const bool channels_last_inputs = options.memory_format == torch::MemoryFormat::ChannelsLast;

                TrainingTelemetry::DeferredScalar last_train_loss_scalar;

#ifdef TORCH_CUDA_AVAILABLE
                struct PrefetchState {
                    explicit PrefetchState(int device_index) : stream(
                        torch::cuda::getStreamFromPool(/*isHighPriority=*/false, device_index)) {
                    }

                    torch::cuda::CUDAStream stream;
                    std::array<torch::Tensor, 2> inputs{};
                    std::array<torch::Tensor, 2> targets{};
                    std::array<at::cuda::CUDAEvent, 2> events{at::cuda::CUDAEvent{}, at::cuda::CUDAEvent{}};
                    std::array<bool, 2> pending{false, false};
                };

                std::optional<PrefetchState> prefetch_state{};
                if (device.is_cuda()) {
                    prefetch_state.emplace(device.index());
                }
#endif


                auto ensure_layout = [&](torch::Tensor tensor, bool apply_channels_last) {
                    if (!tensor.defined()) {
                        return tensor;
                    }
                    if (apply_channels_last && tensor.dim() >= 4) {
                        if (!tensor.is_contiguous(torch::MemoryFormat::ChannelsLast)) {
                            tensor = tensor.contiguous(torch::MemoryFormat::ChannelsLast);
                        }
                    } else if (!tensor.is_contiguous()) {
                        tensor = tensor.contiguous();
                    }
                    return tensor;
                };

                auto stage_to_device = [&](torch::Tensor tensor, torch::Tensor &buffer, bool apply_channels_last,
                                           bool force_non_blocking = false, bool use_prefetch_stream = false) {
                    tensor = ensure_layout(std::move(tensor), apply_channels_last);
                    if (!tensor.defined() || tensor.device() == device) {
                        return tensor;
                    }
                    auto options = tensor.options().device(device);

                    const bool non_blocking = force_non_blocking || tensor.is_pinned();

                    const bool requires_channels_last = apply_channels_last && tensor.dim() >= 4;

                    if (!buffer.defined() || buffer.device() != device || buffer.scalar_type() != tensor.scalar_type()
                        || !buffer.sizes().equals(tensor.sizes())
                        || (requires_channels_last
                                ? !buffer.is_contiguous(torch::MemoryFormat::ChannelsLast)
                                : !buffer.is_contiguous())) {
                        const auto memory_format = requires_channels_last
                                                       ? torch::MemoryFormat::ChannelsLast
                                                       : torch::MemoryFormat::Contiguous;
                        buffer = torch::empty(tensor.sizes(), options, memory_format);
                    }

                    (void) use_prefetch_stream;
#ifdef TORCH_CUDA_AVAILABLE
                    if (use_prefetch_stream && prefetch_state) {
                        torch::cuda::CUDAStreamGuard guard(prefetch_state->stream);
                        buffer.copy_(tensor, non_blocking);
                    } else {
                        buffer.copy_(tensor, non_blocking);
                    }
#else
                    buffer.copy_(tensor, non_blocking);
#endif
                    return buffer;
                };

#ifdef TORCH_CUDA_AVAILABLE
                auto wait_for_prefetch_slot = [&](std::size_t slot) {
                    if (!prefetch_state || !prefetch_state->pending[slot]) {
                        return;
                    }
                    auto current_stream = at::cuda::getCurrentCUDAStream(device.index());
                    current_stream.wait(prefetch_state->events[slot]);
                    prefetch_state->pending[slot] = false;
                };

                auto schedule_prefetch_from_provider = [&](auto &&provider, std::size_t slot) -> bool {
                    if (!prefetch_state) {
                        return false;
                    }

                    prefetch_state->pending[slot] = false;

                    auto batch = provider();
                    if (!batch.first.defined() || !batch.second.defined()) {
                        return false;
                    }

                    prefetch_state->inputs[slot] = stage_to_device(std::move(batch.first),
                                                                   prefetch_state->inputs[slot],
                                                                   channels_last_inputs,
                                                                   /*force_non_blocking=*/true,
                                                                   /*use_prefetch_stream=*/true);
                    prefetch_state->targets[slot] = stage_to_device(std::move(batch.second),
                                                                    prefetch_state->targets[slot],
                                                                    /*apply_channels_last=*/false,
                                                                    /*force_non_blocking=*/true,
                                                                    /*use_prefetch_stream=*/true);

                    {
                        torch::cuda::CUDAStreamGuard guard(prefetch_state->stream);
                        prefetch_state->events[slot].record(prefetch_state->stream);
                    }
                    prefetch_state->pending[slot] = true;
                    return true;
                };

#endif


                for (std::size_t epoch = 0; epoch < options.epoch; ++epoch) {
                    const auto epoch_start = std::chrono::steady_clock::now();

                    torch::Tensor epoch_indices;
                    if constexpr (ShouldShuffle) {
                        if (total_samples > 1) {
                            epoch_indices = torch::randperm(total_samples, index_options);
                        } else {
                            epoch_indices = torch::arange(total_samples, index_options);
                        }
                    }

                    auto accumulation = torch::zeros({}, torch::TensorOptions().dtype(torch::kFloat64).device(device));
                    std::int64_t weight = 0;
                    std::size_t processed_steps = 0;

#ifdef TORCH_CUDA_AVAILABLE
                    auto process_with_prefetch = [&](auto &&provider, std::size_t max_batches) {
                        if (!prefetch_state) {
                            return;
                        }

                        std::size_t current_slot = 0;
                        bool has_batch = schedule_prefetch_from_provider(provider, current_slot);
                        for (std::size_t batch_index = 0; batch_index < max_batches && has_batch; ++batch_index) {
                            wait_for_prefetch_slot(current_slot);
                            const auto processed = process_batch(prefetch_state->inputs[current_slot],
                                                                 prefetch_state->targets[current_slot]);
                            weight += processed;

                            const std::size_t next_slot = current_slot ^ 1U;
                            has_batch = schedule_prefetch_from_provider(provider, next_slot);
                            current_slot = next_slot;
                        }
                    };
#endif


                    const std::size_t total_batches = total_samples > 0
                                                          ? static_cast<std::size_t>(
                                                              (total_samples + batch_size - 1) / batch_size)
                                                          : 0;

                    auto fetch_batch = [&](std::size_t batch_index) {
                        NvtxRange data_range("Nott::Model::train::data_preparation");
                        const auto offset = static_cast<std::int64_t>(batch_index) * batch_size;
                        const auto remaining = total_samples - offset;
                        const auto current_batch = std::min<std::int64_t>(batch_size, remaining);
                        if (current_batch <= 0)
                            return std::pair<torch::Tensor, torch::Tensor>{torch::Tensor{}, torch::Tensor{}};

                        if (graph_mode_enabled && current_batch != batch_size) {
                            throw std::invalid_argument(
                                "Graph optimisation requires every batch to match the captured batch size ("
                                + std::to_string(batch_size)
                                + "). Received " + std::to_string(current_batch)
                                + " samples; pad or drop the remainder before enabling graph replay.");
                        }


                        torch::Tensor batch_inputs;
                        torch::Tensor batch_targets;

                        if constexpr (ShouldShuffle) {
                            auto batch_indices = epoch_indices.narrow(0, offset, current_batch);
                            if (!batch_indices.device().is_cpu()) {
                                batch_indices = batch_indices.to(torch::kCPU);
                            }
                            batch_inputs = train_dataset.inputs.index_select(0, batch_indices);
                            batch_targets = train_dataset.targets.index_select(0, batch_indices);
                            batch_inputs = ensure_layout(std::move(batch_inputs), channels_last_inputs);
                            batch_targets = ensure_layout(std::move(batch_targets), false);
                            if (batch_inputs.device().is_cpu() && !batch_inputs.is_pinned()) {
                                batch_inputs = batch_inputs.pin_memory();
                            }
                            if (batch_targets.device().is_cpu() && !batch_targets.is_pinned()) {
                                batch_targets = batch_targets.pin_memory();
                            }
                        } else {
                            batch_inputs = train_dataset.inputs.narrow(0, offset, current_batch);
                            batch_targets = train_dataset.targets.narrow(0, offset, current_batch);
                            batch_inputs = ensure_layout(std::move(batch_inputs), channels_last_inputs);
                            batch_targets = ensure_layout(std::move(batch_targets), false);
                            if (batch_inputs.device().is_cpu() && !batch_inputs.is_pinned()) {
                                batch_inputs = batch_inputs.pin_memory();
                            }
                            if (batch_targets.device().is_cpu() && !batch_targets.is_pinned()) {
                                batch_targets = batch_targets.pin_memory();
                            }
                        }

                        return std::pair<torch::Tensor, torch::Tensor>{
                            std::move(batch_inputs), std::move(batch_targets)
                        };
                    };

                    auto resolve_batch_signature = [&](std::int64_t batch_size,
                                                       const torch::Tensor &inputs,
                                                       const torch::Tensor &targets) -> const BatchSignature & {
                        auto iterator = batch_signature_cache.find(batch_size);
                        if (iterator == batch_signature_cache.end()) {
                            iterator = batch_signature_cache.emplace(
                                batch_size, BatchSignature::from_tensors(inputs, targets)).first;
                            return iterator->second;
                        }
                        if (!iterator->second.matches(inputs, targets)) {
                            iterator->second = BatchSignature::from_tensors(inputs, targets);
                        }

                        return iterator->second;
                    };

                    auto process_batch = [&](torch::Tensor batch_inputs, torch::Tensor batch_targets) {
                        if (!batch_inputs.defined() || !batch_targets.defined()) {
                            return std::int64_t{0};
                        }

                        const auto current_batch = batch_targets.size(0);
                        if (current_batch <= 0) {
                            return std::int64_t{0};
                        }

                        torch::Tensor loss;
                        bool replay_retry_attempted = false;


                        {
                            NvtxRange data_range("Nott::Model::train::data_preparation");
                            batch_inputs = stage_to_device(std::move(batch_inputs), device_batch_inputs_buffer,
                                                           channels_last_inputs);
                            batch_targets = stage_to_device(std::move(batch_targets), device_batch_targets_buffer,
                                                            false);
                        }


                        while (true) {
                            GraphMode batch_graph_mode = graph_mode;
                            const BatchSignature *batch_signature{nullptr};

                            if (graph_mode == GraphMode::Capture) {
                                const auto &resolved_signature = resolve_batch_signature(
                                    current_batch, batch_inputs, batch_targets);
                                batch_signature = &resolved_signature;

                                bool capture_ready = false;
                                if (batch_signature && active_graph_signature && *active_graph_signature == *
                                    batch_signature) {
                                    auto readiness = graph_capture_ready.find(*batch_signature);
                                    capture_ready = (readiness != graph_capture_ready.end()) && readiness->second;
                                }

                                if (!capture_ready) {
                                    batch_graph_mode = GraphMode::Capture;


                                    graph_capture_ready[*batch_signature] = false;

                                    if (!active_graph_signature || *active_graph_signature != *batch_signature) {
                                        if (active_graph_signature) {
                                            graph_capture_ready[*active_graph_signature] = false;
                                        }
                                        active_graph_signature.reset();
                                    }

                                    model.reset_graph_shape_cache(GraphMode::Capture);
                                } else {
                                    batch_graph_mode = GraphMode::Replay;
                                }
                            }

                            try {
                                if (graph_mode_enabled) {
                                    model.prepare_optimizers_for_graph(batch_graph_mode);
                                    model.ensure_graph_batch_shapes(batch_graph_mode, batch_inputs, batch_targets);
                                }

                                loss = training_step(model, batch_inputs, batch_targets, batch_graph_mode,
                                                     regularization_active, amp_enabled);

                                if (graph_mode == GraphMode::Capture) {
                                    if (batch_signature) {
                                        if (batch_graph_mode == GraphMode::Capture) {
                                            graph_capture_ready[*batch_signature] = true;
                                            active_graph_signature = *batch_signature;
                                        } else if (batch_graph_mode == GraphMode::Replay) {
                                            active_graph_signature = *batch_signature;
                                        }
                                    }
                                }

                                break;
                            } catch (const std::runtime_error &) {
                                if (!(graph_mode == GraphMode::Capture)) {
                                    throw;
                                }

                                if (batch_graph_mode == GraphMode::Replay && !replay_retry_attempted) {
                                    graph_capture_ready[*batch_signature] = false;
                                    if (active_graph_signature && *active_graph_signature == *batch_signature) {
                                        active_graph_signature.reset();
                                    }
                                    model.reset_graph_shape_cache(GraphMode::Capture);
                                    replay_retry_attempted = true;
                                    continue;
                                }

                                throw;
                            }
                        }

                        {
                            NvtxRange idle_range("Nott::Model::train::kernel_idle");
                            model.step_scheduler();


                            if (regularization_active) {
                                model.update_regularization_states(step_index, true);
                            }
                            ++step_index;
                            ++processed_steps;


                            auto loss_tensor = loss.detach();
                            if (loss_tensor.device() != accumulation.device()) {
                                loss_tensor = loss_tensor.to(accumulation.device());
                            }
                            loss_tensor = loss_tensor.to(torch::kFloat64);
                            loss_tensor.mul_(static_cast<double>(current_batch));
                            accumulation.add_(loss_tensor);
                        }
                        return current_batch;
                    };

                    if constexpr (BufferVRAM) {
                        if (graph_mode_enabled) {
#ifdef TORCH_CUDA_AVAILABLE
                            if (prefetch_state) {
                                std::size_t next_batch_index = 0;
                                auto provider = [&]() -> std::pair<torch::Tensor, torch::Tensor> {
                                    if (next_batch_index >= total_batches) {
                                        return {};
                                    }
                                    auto batch = fetch_batch(next_batch_index);
                                    ++next_batch_index;
                                    return batch;
                                };
                                process_with_prefetch(provider, total_batches);
                            } else
#endif
                            {
                                for (std::size_t batch_index = 0; batch_index < total_batches; ++batch_index) {
                                    auto batch_pair = fetch_batch(batch_index);
                                    const auto processed = process_batch(
                                        std::move(batch_pair.first), std::move(batch_pair.second));
                                    weight += processed;
                                }
                            }
                        } else {
                            std::deque<std::pair<torch::Tensor, torch::Tensor> > buffered_batches;
                            std::size_t next_batch_to_load = 0;
                            const std::size_t max_batches = total_batches == 0 ? 1 : total_batches;
                            const std::size_t buffer_limit = std::max<std::size_t>(1,
                                std::min<std::size_t>(options.buffer_vram + 1, max_batches));

                            auto maintain_buffer = [&](std::size_t current_index) {
                                const std::size_t desired_size = std::min(buffer_limit, total_batches - current_index);
                                while (buffered_batches.size() < desired_size && next_batch_to_load < total_batches) {
                                    auto batch = fetch_batch(next_batch_to_load);
                                    buffered_batches.push_back(std::move(batch));
                                    ++next_batch_to_load;
                                }
                            };

#ifdef TORCH_CUDA_AVAILABLE
                            if (prefetch_state) {
                                std::size_t processed_batches = 0;
                                auto provider = [&]() -> std::pair<torch::Tensor, torch::Tensor> {
                                    maintain_buffer(processed_batches);
                                    if (buffered_batches.empty()) {
                                        return {};
                                    }
                                    auto batch_pair = std::move(buffered_batches.front());
                                    buffered_batches.pop_front();
                                    ++processed_batches;
                                    maintain_buffer(processed_batches);
                                    return batch_pair;
                                };
                                process_with_prefetch(provider, total_batches);
                            } else
#endif
                            {
                                for (std::size_t batch_index = 0; batch_index < total_batches; ++batch_index) {
                                    maintain_buffer(batch_index);
                                    if (buffered_batches.empty()) {
                                        break;
                                    }

                                    auto batch_pair = std::move(buffered_batches.front());
                                    buffered_batches.pop_front();

                                    const auto processed = process_batch(
                                        std::move(batch_pair.first), std::move(batch_pair.second));
                                    weight += processed;


                                    maintain_buffer(batch_index + 1);
                                }
                            }
                        }
                    } else {
#ifdef TORCH_CUDA_AVAILABLE
                        if (prefetch_state) {
                            std::size_t next_batch_index = 0;
                            auto provider = [&]() -> std::pair<torch::Tensor, torch::Tensor> {
                                if (next_batch_index >= total_batches) {
                                    return {};
                                }
                                auto batch = fetch_batch(next_batch_index);
                                ++next_batch_index;
                                return batch;
                            };
                            process_with_prefetch(provider, total_batches);
                        } else
#endif
                        {
                            for (std::size_t batch_index = 0; batch_index < total_batches; ++batch_index) {
                                auto batch_pair = fetch_batch(batch_index);
                                const auto processed = process_batch(std::move(batch_pair.first),
                                                                     std::move(batch_pair.second));
                                weight += processed;
                            }
                        }
                    }

                    TrainingTelemetry::DeferredScalar train_loss_scalar;
                    if (weight > 0) {
                        auto averaged_loss_tensor = accumulation / static_cast<double>(weight);
                        train_loss_scalar = TrainingTelemetry::DeferredScalar::from_tensor(
                            std::move(averaged_loss_tensor), device);
                    } else {
                        auto zero_tensor = accumulation.detach();
                        zero_tensor.zero_();
                        train_loss_scalar = TrainingTelemetry::DeferredScalar::from_tensor(
                            std::move(zero_tensor), device);
                    }

                    last_train_loss_scalar = train_loss_scalar;

                    std::optional<TrainingTelemetry::DeferredScalar> test_loss_scalar{};

                    std::optional<double> test_loss{};
                    if (test_dataset) {
                        test_loss_scalar = compute_dataset_loss<BufferVRAM>(
                            model, *test_dataset, options.batch_size, options.buffer_vram);
                        if (test_loss_scalar) {
                            test_loss = test_loss_scalar->materialize();
                        }
                    }

                    bool improved = false;
                    std::optional<double> delta{};
                    if (test_loss) {
                        if (!best_test) {
                            improved = true;
                            best_test = test_loss;
                        } else {
                            const auto previous_best = *best_test;
                            delta = *test_loss - previous_best;
                            if (*test_loss < previous_best) {
                                improved = true;
                                best_test = test_loss;
                            }
                        }
                    }

                    if (improved && options.restore_best_state) {
                        best_parameters.clear();
                        best_buffers.clear();
                        best_parameters.reserve(model.parameters().size());
                        best_buffers.reserve(model.buffers().size());

                        for (auto &parameter: model.parameters()) {
                            if (parameter.defined()) {
                                best_parameters.push_back(parameter.detach().clone(torch::MemoryFormat::Preserve));
                            } else {
                                best_parameters.push_back({});
                            }
                        }

                        for (auto &buffer: model.buffers()) {
                            if (buffer.defined()) {
                                best_buffers.push_back(buffer.detach().clone(torch::MemoryFormat::Preserve));
                            } else {
                                best_buffers.push_back({});
                            }
                        }

                        best_state_captured = true;
                    }


                    const auto duration_seconds = std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - epoch_start).count();

                    const auto epoch_timestamp = std::chrono::system_clock::now();
                    auto learning_rates = model.collect_learning_rates();
                    TrainingTelemetry::DeferredScalar step_latency_scalar;
                    if (processed_steps > 0) {
                        const auto average_step_latency = duration_seconds / static_cast<double>(processed_steps);
                        auto latency_tensor = torch::tensor(average_step_latency,
                                                            torch::TensorOptions().dtype(torch::kFloat64));
                        const torch::Device cpu_device{torch::kCPU};
                        step_latency_scalar = TrainingTelemetry::DeferredScalar::from_tensor(
                            std::move(latency_tensor), cpu_device);
                    }
                    model.record_epoch_telemetry({
                        epoch + 1,
                        train_loss_scalar,
                        test_loss_scalar,
                        delta,
                        std::move(learning_rates),
                        epoch_timestamp,
                        duration_seconds,
                        step_latency_scalar
                    });

                    if (options.monitor && options.stream) {
                        pending_epoch_logs.push_back({
                            epoch + 1,
                            options.epoch,
                            train_loss_scalar,
                            test_loss,
                            delta,
                            improved,
                            duration_seconds
                        });
                        flush_pending_epoch_logs(false);
                    }
                }
                flush_pending_epoch_logs(true);
                if (options.restore_best_state && best_state_captured) {
                    [[maybe_unused]] const auto train_loss = last_train_loss_scalar.materialize();
                    std::cout << "[Nott] Reloading best state of the network..." << std::endl;
                    torch::NoGradGuard no_grad{};
                    auto parameters = model.parameters();
                    const auto parameter_limit = std::min(parameters.size(), best_parameters.size());
                    for (std::size_t index = 0; index < parameter_limit; ++index) {
                        auto &target = parameters[index];
                        const auto &source = best_parameters[index];
                        if (target.defined() && source.defined()) {
                            target.copy_(source);
                        }
                    }

                    auto buffers = model.buffers();
                    const auto buffer_limit = std::min(buffers.size(), best_buffers.size());
                    for (std::size_t index = 0; index < buffer_limit; ++index) {
                        auto &target = buffers[index];
                        const auto &source = best_buffers[index];
                        if (target.defined() && source.defined()) {
                            target.copy_(source);
                        }
                    }
                }
            }


            template<bool BufferVRAM>
            static std::optional<TrainingTelemetry::DeferredScalar> compute_dataset_loss(
                Model &model, const TensorDataset &dataset, std::size_t batch_size, std::size_t buffer_vram) {
                if (!dataset.inputs.defined() || !dataset.targets.defined()) {
                    return std::nullopt;
                }
                if (dataset.inputs.size(0) == 0) {
                    return std::nullopt;
                }

                if (batch_size == 0) {
                    throw std::invalid_argument("Batch size must be greater than zero when computing dataset loss.");
                }

                if constexpr (!BufferVRAM) {
                    (void) buffer_vram;
                }

                const auto device = model.device();
                const auto total_samples = dataset.inputs.size(0);
                const bool regularization_active = model.has_regularization();

                torch::NoGradGuard no_grad;
                const bool was_training = model.is_training();
                model.eval();

                torch::Tensor dataset_inputs = dataset.inputs;
                torch::Tensor dataset_targets = dataset.targets;

                if constexpr (BufferVRAM) {
                    if (!device.is_cuda()) {
                        throw std::runtime_error(
                            "VRAM buffering for dataset loss requires the model to be on a CUDA device.");
                    }
                    if (dataset_inputs.defined() && !dataset_inputs.device().is_cpu()) {
                        dataset_inputs = dataset_inputs.to(torch::kCPU);
                    }
                    if (dataset_targets.defined() && !dataset_targets.device().is_cpu()) {
                        dataset_targets = dataset_targets.to(torch::kCPU);
                    }
                }

                const bool channels_last_inputs =
                        model.preferred_tensor_memory_format() == torch::MemoryFormat::ChannelsLast;

                auto ensure_layout = [&](torch::Tensor tensor, bool apply_channels_last) {
                    if (!tensor.defined()) {
                        return tensor;
                    }
                    if (apply_channels_last && tensor.dim() >= 4) {
                        if (!tensor.is_contiguous(torch::MemoryFormat::ChannelsLast)) {
                            tensor = tensor.contiguous(torch::MemoryFormat::ChannelsLast);
                        }
                    } else if (!tensor.is_contiguous()) {
                        tensor = tensor.contiguous();
                    }
                    return tensor;
                };

                dataset_inputs = ensure_layout(std::move(dataset_inputs), channels_last_inputs);
                dataset_targets = ensure_layout(std::move(dataset_targets), false);

                if (dataset_inputs.device().is_cpu() && !dataset_inputs.is_pinned()) {
                    dataset_inputs = dataset_inputs.pin_memory();
                }
                if (dataset_targets.device().is_cpu() && !dataset_targets.is_pinned()) {
                    dataset_targets = dataset_targets.pin_memory();
                }

                const auto batch_extent = static_cast<std::int64_t>(batch_size);
                const std::size_t total_batches = total_samples > 0
                                                      ? static_cast<std::size_t>(
                                                          (total_samples + batch_extent - 1) / batch_extent)
                                                      : 0;


                auto accumulation = torch::zeros({}, torch::TensorOptions().dtype(torch::kFloat64).device(device));
                std::int64_t weight = 0;

                torch::Tensor device_batch_inputs_buffer;
                torch::Tensor device_batch_targets_buffer;

                auto stage_to_device = [&](torch::Tensor tensor, torch::Tensor &buffer, bool apply_channels_last) {
                    tensor = ensure_layout(std::move(tensor), apply_channels_last);
                    if (!tensor.defined() || tensor.device() == device) {
                        return tensor;
                    }

                    auto options = tensor.options().device(device);
                    const bool non_blocking = tensor.is_pinned();
                    const bool requires_channels_last = apply_channels_last && tensor.dim() >= 4;

                    if (!buffer.defined()
                        || buffer.device() != device
                        || buffer.scalar_type() != tensor.scalar_type()
                        || !buffer.sizes().equals(tensor.sizes())
                        || (requires_channels_last
                                ? !buffer.is_contiguous(torch::MemoryFormat::ChannelsLast)
                                : !buffer.is_contiguous())) {
                        const auto memory_format = requires_channels_last
                                                       ? torch::MemoryFormat::ChannelsLast
                                                       : torch::MemoryFormat::Contiguous;
                        buffer = torch::empty(tensor.sizes(), options, memory_format);
                    }

                    buffer.copy_(tensor, non_blocking);
                    return buffer;
                };


                auto prepare_batch = [&](torch::Tensor batch_inputs,
                                         torch::Tensor batch_targets) -> std::optional<StreamingBatch> {
                    if (!batch_inputs.defined() || !batch_targets.defined()) {
                        return std::nullopt;
                    }
                    batch_inputs = ensure_layout(std::move(batch_inputs), channels_last_inputs);
                    batch_targets = ensure_layout(std::move(batch_targets), false);

                    auto staged_inputs = stage_to_device(std::move(batch_inputs),
                                                         device_batch_inputs_buffer,
                                                         channels_last_inputs);
                    auto staged_targets = stage_to_device(std::move(batch_targets),
                                                          device_batch_targets_buffer,
                                                          false);

                    if (!staged_inputs.defined() || !staged_targets.defined()) {
                        return std::nullopt;
                    }

                    StreamingBatch batch{};
                    batch.inputs = std::move(staged_inputs);
                    batch.targets = std::move(staged_targets);
                    return batch;
                };

                auto consume_batch = [&](torch::Tensor prediction, StreamingBatch batch) {
                    if (!prediction.defined() || !batch.targets.defined()) {
                        return;
                    }

                    auto staged_targets = std::move(batch.targets);
                    const auto current_batch = staged_targets.size(0);
                    if (current_batch <= 0) {
                        return;
                    }


                    if (!prediction.sizes().equals(staged_targets.sizes())) {
                        if (staged_targets.numel() == prediction.numel()) {
                            staged_targets = staged_targets.reshape_as(prediction);
                        }
                    }

                    auto loss = model.compute_loss(prediction, staged_targets);
                    if (loss.dim() != 0) {
                        loss = loss.mean();
                    }
                    if (regularization_active) {
                        auto regularization_penalty = model.compute_regularization_penalty();
                        if (regularization_penalty.defined()) {
                            if (regularization_penalty.device() != loss.device()) {
                                regularization_penalty = regularization_penalty.to(loss.device());
                            }
                            if (regularization_penalty.scalar_type() != loss.scalar_type()) {
                                regularization_penalty = regularization_penalty.to(loss.scalar_type());
                            }
                            loss = loss + regularization_penalty;
                        }
                    }

                    auto loss_tensor = loss.detach();
                    if (loss_tensor.device() != accumulation.device()) {
                        loss_tensor = loss_tensor.to(accumulation.device());
                    }
                    loss_tensor = loss_tensor.to(torch::kFloat64);
                    loss_tensor.mul_(static_cast<double>(current_batch));
                    accumulation.add_(loss_tensor);
                    weight += current_batch;
                };


                StreamingOptions streaming_options{};
                streaming_options.batch_size = static_cast<std::size_t>(batch_size);

                if constexpr (BufferVRAM) {
                    const std::size_t max_batches = total_batches == 0 ? 1 : total_batches;
                    streaming_options.buffer_batches = std::max<std::size_t>(
                        std::size_t{1}, std::min<std::size_t>(buffer_vram + 1, max_batches));
                }
                model.stream_forward(std::move(dataset_inputs),
                                     std::move(dataset_targets),
                                     streaming_options,
                                     prepare_batch,
                                     consume_batch);
                if (was_training) {
                    model.train();
                } else {
                    model.eval();
                }

                if (weight == 0) {
                    return std::nullopt;
                }

                auto averaged_loss_tensor = accumulation / static_cast<double>(weight);
                auto loss_scalar = TrainingTelemetry::DeferredScalar::from_tensor(
                    std::move(averaged_loss_tensor), device);
                auto learning_rates = model.collect_learning_rates();
                const auto timestamp = std::chrono::system_clock::now();
                model.record_dataset_loss_telemetry({
                    loss_scalar,
                    static_cast<std::size_t>(dataset.inputs.size(0)),
                    std::move(learning_rates),
                    timestamp
                });

                return loss_scalar;
            }

            static void log_epoch(std::ostream &stream,
                                  std::size_t epoch_index,
                                  std::size_t total_epochs,
                                  double train_loss,
                                  const std::optional<double> &test_loss,
                                  const std::optional<double> &delta,
                                  bool improved,
                                  double duration_seconds) {
                using Utils::Terminal::ApplyColor;
                using Utils::Terminal::Colors::kBrightBlack;
                using Utils::Terminal::Colors::kBrightBlue;
                using Utils::Terminal::Colors::kBrightGreen;
                using Utils::Terminal::Colors::kBrightYellow;
                using Utils::Terminal::Colors::kReset;

                std::ostringstream line;
                line << "Epoch [" << epoch_index << "/" << total_epochs << "] | ";
                line << ApplyColor("Train", kBrightYellow) << " loss: "
                        << std::fixed << std::setprecision(6) << train_loss << " | ";
                line << ApplyColor("Test", kBrightBlue) << " loss: ";
                if (test_loss) {
                    line << std::fixed << std::setprecision(6) << *test_loss;
                } else {
                    line << "N/A";
                }

                line << " | ΔLoss: ";
                if (test_loss && delta) {
                    std::ostringstream delta_stream;
                    delta_stream << std::showpos << std::fixed << std::setprecision(6) << *delta;
                    line << delta_stream.str();
                } else if (test_loss) {
                    line << "N/A";
                } else {
                    line << "N/A";
                }

                const std::string nabla_symbol{"∇"};
                const std::string grey{kBrightBlack};
                const std::string green{kBrightGreen};
                const std::string reset{kReset};

                std::string nabla_indicator;

                if (improved)
                    line << grey << " (" << green << nabla_symbol << grey << ")" << reset;
                else
                    line << grey << " (" << nabla_symbol << ")" << reset;

                std::ostringstream duration_stream;
                duration_stream << std::fixed << std::setprecision(2) << duration_seconds << "sec";
                line << " | "
                        << ApplyColor("duration: " + duration_stream.str(), kBrightBlack);

                stream << line.str() << '\n';
            }
        };

        std::vector<Layer::Details::RegisteredLayer> layers_{};
        std::vector<NamedModuleDescriptor> module_descriptors_{};
        std::vector<CalibrationMethod> calibration_methods_{};
        std::vector<std::vector<torch::Tensor> > layer_parameters_{};
        std::vector<std::vector<RegularizationBinding> > layer_regularization_bindings_{};
        std::vector<torch::Tensor> global_regularization_parameters_{};
        std::vector<RegularizationBinding> global_regularization_bindings_{};
        mutable std::vector<GraphRegularizationBindingInfo> graph_global_regularization_metadata_{};
        mutable std::vector<std::vector<GraphRegularizationBindingInfo> > graph_layer_regularization_metadata_{};
        mutable std::vector<GraphCalibrationInfo> graph_calibration_metadata_{};
        mutable bool graph_regularization_metadata_dirty_{true};
        mutable bool graph_calibration_metadata_dirty_{true};
        mutable std::optional<std::vector<int64_t> > last_input_shape_{};
        mutable std::optional<std::vector<int64_t> > last_target_shape_{};
        mutable std::optional<std::vector<int64_t> > graph_input_shape_cache_{};
        mutable std::optional<std::vector<int64_t> > graph_target_shape_cache_{};
        TrainingTelemetry telemetry_{};
        std::size_t module_index_{0};
        std::unordered_map<std::string, ModuleNameBinding> module_name_index_{};
        std::vector<CompiledNode> compiled_nodes_{};
        std::vector<CompiledStep> compiled_steps_{};
        std::vector<ExecutionStep> execution_steps_{};
        std::vector<JoinBuffer> join_buffers_{};
        std::vector<LinkSpec> compiled_links_{};
        std::optional<std::size_t> compiled_output_node_index_{};
        std::vector<torch::Tensor> node_activations_{};
        std::vector<std::vector<torch::Tensor> > join_workspace_{};
        GraphExecutionWorkspace graph_workspace_{};
        GraphCaptureState graph_capture_training_{};
        GraphCaptureState graph_capture_inference_{};
        bool graph_capture_opt_in_{false};
        std::vector<Layer::Details::RegisteredLayer *> cached_layer_pointers_{};
        bool execution_workspace_dirty_{true};
        bool routing_active_{false};
        std::optional<OptimizerBinding> optimizer_{};
        std::vector<OptimizerBinding> local_optimizers_{};
        std::unique_ptr<LrScheduler::Details::Scheduler> scheduler_{};
        using StepImpl = void (Model::*)();
        StepImpl step_impl_{&Model::step_not_configured};
        using LossDescriptor = Loss::Descriptor;
        std::optional<LossDescriptor> loss_descriptor_{};
        std::string name_{};
        torch::Device device_{torch::kCPU, 0};
        bool regularization_configured_{false};
        std::string model_name_{};
        torch::MemoryFormat tensor_memory_format_{torch::MemoryFormat::Contiguous};
        std::function<void(const torch::Tensor &, bool)> staging_observer_{};
        bool has_convolutional_layers_{false};
        bool amp_training_active_{false};
#ifdef TORCH_CUDA_AVAILABLE
        std::optional<torch::cuda::amp::GradScaler> amp_scaler_{};
#endif


        void configure_step_impl() noexcept {
            if (!optimizer_ && local_optimizers_.empty()) {
                step_impl_ = &Model::step_not_configured;
                return;
            }
            step_impl_ = scheduler_ ? &Model::step_configured<true> : &Model::step_configured<false>;
        }

        void step_optimizers() {
            if (optimizer_) {
                optimizer_->instance->step();
            }
            for (auto &optimizer: local_optimizers_) {
                optimizer.instance->step();
            }
        }
#ifdef TORCH_CUDA_AVAILABLE
        void step_optimizers_with_scaler(torch::cuda::amp::GradScaler &scaler) {
            if (optimizer_) {
                scaler.step(*optimizer_->instance);
            }
            for (auto &optimizer: local_optimizers_) {
                scaler.step(*optimizer.instance);
            }
        }
#endif

        void step_not_configured() {
            throw std::logic_error("Optimizer has not been configured.");
        }

        template<bool WithScheduler>
        void step_configured() {
            if constexpr (WithScheduler) {
                step_scheduler();
            }
            step_optimizers();
        }

        [[nodiscard]] std::size_t next_module_index() noexcept { return module_index_++; }

        void reset_runtime_state() {
            auto preserved_name = name_;
            auto preserved_device = device_;
            auto preserved_model_name = model_name_;

            this->~Model();
            new(this) Model(preserved_name);

            device_ = preserved_device;
            model_name_ = std::move(preserved_model_name);
            module_name_index_.clear();
            clear_compiled_graph();
        }

        static std::string format_tensor_shape(const torch::Tensor &tensor) {
            std::ostringstream stream;
            stream << '(';
            const auto sizes = tensor.sizes();
            for (int64_t index = 0; index < sizes.size(); ++index) {
                if (index > 0) {
                    stream << ", ";
                }
                stream << sizes[index];
            }
            stream << ')';
            return stream.str();
        }

        struct AutocastGuard {
            AutocastGuard(bool enabled, c10::DeviceType device_type, torch::ScalarType dtype)
                : enabled_(enabled), device_type_(device_type) {
                if (enabled_) {
                    previous_enabled_ = at::autocast::is_autocast_enabled(device_type_);
                    previous_dtype_ = at::autocast::get_autocast_dtype(device_type_);
                    at::autocast::set_autocast_dtype(device_type_, dtype);
                    at::autocast::set_autocast_enabled(device_type_, true);
                }
            }

            AutocastGuard(const AutocastGuard &) = delete;

            AutocastGuard &operator=(const AutocastGuard &) = delete;

            AutocastGuard(AutocastGuard &&) = delete;

            AutocastGuard &operator=(AutocastGuard &&) = delete;

            ~AutocastGuard() {
                if (enabled_) {
                    at::autocast::set_autocast_enabled(device_type_, previous_enabled_);
                    at::autocast::set_autocast_dtype(device_type_, previous_dtype_);
                }
            }

        private:
            bool enabled_{false};
            c10::DeviceType device_type_{c10::DeviceType::CPU};
            bool previous_enabled_{false};
            torch::ScalarType previous_dtype_{torch::kFloat32};
        };

        [[nodiscard]] torch::ScalarType determine_autocast_dtype() const {
            for (const auto &parameter: this->parameters(/*recurse=*/false)) {
                if (parameter.defined()) {
                    return parameter.scalar_type();
                }
            }
            for (const auto &buffer: this->buffers(/*recurse=*/false)) {
                if (buffer.defined()) {
                    return buffer.scalar_type();
                }
            }
            return torch::kFloat32;
        }

#ifdef TORCH_CUDA_AVAILABLE
        void ensure_amp_scaler() {
            if (!amp_scaler_) {
                amp_scaler_.emplace();
            }
        }
#endif
    };
}

#include "plot/plot.hpp"

namespace Nott {
    template<class Descriptor, class... Args>
    decltype(auto) Model::plot(Descriptor descriptor, Args &&... args) {
        return Plot::Render(*this,
                            std::move(descriptor),
                            std::forward<Args>(args)...);
    }
}
#endif //Nott_CORE_HPP
