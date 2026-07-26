#ifndef Nott_CORE_HPP
#define Nott_CORE_HPP

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
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>
#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAStream.h>
#endif
#include <ATen/DeviceGuard.h>
#include <ATen/autocast_mode.h>


#include "../common/streaming.hpp"

#include "../common/graph.hpp"
#include "../common/save_load.hpp"
#include "../utils/terminal.hpp"
#include "../activation/activation.hpp"
#include "../activation/apply.hpp"
#include "../initialization/initialization.hpp"
#include "../attention/details/head.hpp"
#include "../block/block.hpp"
#include "../initialization/apply.hpp"
#include "../layer/layer.hpp"
#include "../block/details/blocks/residual.hpp"
#include "../block/details/blocks/sequential.hpp"
#include "../evaluation/evaluation.hpp"
#include "../loss/loss.hpp"
#include "../loss/details/mse.hpp"
#include "../optimizer/optimizer.hpp"
#include "../lrscheduler/lrscheduler.hpp"
#include "../block/block.hpp"
#include "../layer/details/positional_encoding.hpp"
#include "../regularization/regularization.hpp"
#include "../regularization/apply.hpp"
#include "../calibration/calibration.hpp"
#include "../training/kfold.hpp"
#include "../training/deferred_scalar.hpp"
#include "../training/training_policy.hpp"
#include "../training/training_preflight.hpp"
#include "../training/dataset_pipeline.hpp"
#include "../training/batch_iterator.hpp"
#include "../training/graph_coordinator.hpp"
#include "../training/epoch_runner.hpp"
#include "../training/compute_dataset_loss.hpp"

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
    bool drop_last{false};  // when true, silently skips incomplete final batch each epoch
                            /// Enable CUDA graph capture/replay; pad or drop remainder batches first.
    bool enable_amp{false}; // Enable TensorCores
    torch::MemoryFormat memory_format{torch::MemoryFormat::Contiguous};
  };


  class Model : public torch::nn::Module {
    using RegularizationState = Regularization::StateVariant;
    using RegularizationStateStorage = std::shared_ptr<std::vector<RegularizationState> >;
    using RegularizationAccumulator = Regularization::Accumulator;
    using CalibrationMethod = Calibration::MethodPtr;


    struct OptimizerBinding {
      std::unique_ptr<torch::optim::Optimizer> instance{};
      std::function<void(torch::optim::Optimizer &)> warmup{};
      bool capture_safe{false};
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
      std::unique_ptr<at::cuda::CUDAGraph> graph{};
      std::optional<c10::cuda::CUDAStream> capture_stream{};

      // true once a graph has been captured and nothing has invalidated it since
      [[nodiscard]] bool is_replay_ready() const noexcept;

      // replays the captured graph on its capture stream; throws if not captured
      void run_replay();

      // runs work() once under CUDA graph capture; resets state and rethrows on failure
      template <class Fn>
      torch::Tensor run_capture(Fn &&work);
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
      using DeferredScalar = Nott::Training::DeferredScalar;

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

    void links(std::vector<LinkSpec> specifications, bool enable_graph_capture);

    /// Updated multi-IO + params form
    void links(std::vector<LinkSpec> specifications, LinkParams params);


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
            binding.capture_safe = false;
            binding.warmup = [](torch::optim::Optimizer &optimizer) {
            if (auto *a = dynamic_cast<Optimizer::Details::Adam *>(&optimizer)) {
              a->ensure_state_initialized();
            }
            };
            } else if constexpr (std::is_same_v<DescriptorType, Optimizer::Details::AdamWDescriptor>) {
              auto options = Optimizer::Details::to_torch_options(concrete_descriptor.options);
              binding.instance = std::make_unique<Optimizer::Details::AdamW>(
                  std::move(parameters), options, warmup_buckets);
              binding.capture_safe = false;
              binding.warmup = [](torch::optim::Optimizer &optimizer) {
                if (auto *aw = dynamic_cast<Optimizer::Details::AdamW *>(&optimizer)) {
                  aw->ensure_state_initialized();
                }
              };
            } else if constexpr (std::is_same_v<DescriptorType, Optimizer::Details::SophiaGDescriptor>) {
              binding.instance = std::make_unique<Optimizer::Details::SophiaG>(
                  std::move(parameters), concrete_descriptor.options);
              binding.capture_safe = false;
              binding.warmup = [](torch::optim::Optimizer &optimizer) {
                if (auto *sophia = dynamic_cast<Optimizer::Details::SophiaG *>(&optimizer)) {
                  sophia->ensure_state_initialized();
                }
              };
            } else if constexpr (std::is_same_v<DescriptorType, Optimizer::Details::SophiaHDescriptor>) {
              binding.instance = std::make_unique<Optimizer::Details::SophiaH>(
                  std::move(parameters), concrete_descriptor.options);
              binding.capture_safe = false;
              binding.warmup = [](torch::optim::Optimizer &optimizer) {
                if (auto *sophia = dynamic_cast<Optimizer::Details::SophiaH *>(&optimizer)) {
                  sophia->ensure_state_initialized();
                }
              };
            } else if constexpr (std::is_same_v<DescriptorType, Optimizer::Details::MuonDescriptor>) {
              binding.instance = std::make_unique<Optimizer::Details::Muon>(
                  std::move(parameters), concrete_descriptor.options);
              binding.capture_safe = false;
              binding.warmup = [](torch::optim::Optimizer &optimizer) {
                if (auto *muon = dynamic_cast<Optimizer::Details::Muon *>(&optimizer)) {
                  muon->ensure_state_initialized();
                }
              };
            } else if constexpr (std::is_same_v<DescriptorType, Optimizer::Details::AdaMuonDescriptor>) {
              binding.instance = std::make_unique<Optimizer::Details::AdaMuon>(
                  std::move(parameters), concrete_descriptor.options);
              binding.capture_safe = false;
              binding.warmup = [](torch::optim::Optimizer &optimizer) {
                if (auto *AdaMuon = dynamic_cast<Optimizer::Details::AdaMuon *>(&optimizer)) {
                  AdaMuon->ensure_state_initialized();
                }
              };
            } else if constexpr (std::is_same_v<DescriptorType,
                Optimizer::Details::MuonManifoldDescriptor>) {
              binding.instance = std::make_unique<Optimizer::Details::MuonManifold>(
                  std::move(parameters), concrete_descriptor.options);
              binding.capture_safe = false;
              binding.warmup = [](torch::optim::Optimizer &optimizer) {
                if (auto *MuonManifold = dynamic_cast<Optimizer::Details::MuonManifold *>(&optimizer)) {
                  MuonManifold->ensure_state_initialized();
                }
              };
            } else if constexpr (std::is_same_v<DescriptorType, Optimizer::Details::AdafactorDescriptor>) {
              binding.instance = std::make_unique<Optimizer::Details::Adafactor>(
                  std::move(parameters), concrete_descriptor.options);
              binding.capture_safe = false;
              binding.warmup = [](torch::optim::Optimizer &optimizer) {
                if (auto *adafactor = dynamic_cast<Optimizer::Details::Adafactor *>(&optimizer)) {
                  adafactor->ensure_state_initialized();
                }
              };
            } else if constexpr (std::is_same_v<DescriptorType, Optimizer::Details::AdagradDescriptor>) {
              auto options = Optimizer::Details::to_torch_options(concrete_descriptor.options);
              binding.instance = std::make_unique<Optimizer::Details::Adagrad>(
                  std::move(parameters), options, warmup_buckets);
              binding.capture_safe = false;
              binding.warmup = [](torch::optim::Optimizer &optimizer) {
                if (auto *ada = dynamic_cast<Optimizer::Details::Adagrad *>(&optimizer)) {
                  ada->ensure_state_initialized();
                }
              };
            } else if constexpr (std::is_same_v<DescriptorType, Optimizer::Details::LAMBDescriptor>) {
              binding.instance = std::make_unique<Optimizer::Details::LAMB>(
                  std::move(parameters), concrete_descriptor.options);
              binding.capture_safe = false;
              binding.warmup = [](torch::optim::Optimizer &optimizer) {
                if (auto *lamb = dynamic_cast<Optimizer::Details::LAMB *>(&optimizer)) {
                  lamb->ensure_state_initialized();
                }
              };
            } else if constexpr (std::is_same_v<DescriptorType, Optimizer::Details::LionDescriptor>) {
              binding.instance = std::make_unique<Optimizer::Details::Lion>(
                  std::move(parameters), concrete_descriptor.options);
              binding.capture_safe = false;
              binding.warmup = [](torch::optim::Optimizer &optimizer) {
                if (auto *lion = dynamic_cast<Optimizer::Details::Lion *>(&optimizer)) {
                  lion->ensure_state_initialized();
                }
              };
            } else if constexpr (std::is_same_v<DescriptorType, Optimizer::Details::RMSpropDescriptor>) {
              auto options = Optimizer::Details::to_torch_options(concrete_descriptor.options);
              binding.instance = std::make_unique<Optimizer::Details::RMSProp>(
                  std::move(parameters), options, warmup_buckets);
              binding.capture_safe = false;
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
      cached_autocast_dtype_.reset();
      this->to(device_);
      return *this;
    }

    // TF32 and cuDNN benchmark are process-global, not per-layer; set them in one place
    Model &set_precision(bool allow_tf32, bool benchmark_cudnn = true) {
      at::globalContext().setAllowTF32CuDNN(allow_tf32);
      at::globalContext().setAllowTF32CuBLAS(allow_tf32);
      if (torch::cuda::is_available() && torch::cuda::cudnn_is_available()) {
        at::globalContext().setBenchmarkCuDNN(benchmark_cudnn);
      }
      return *this;
    }

    [[nodiscard]] const torch::Device &device() const noexcept { return device_; }


    [[nodiscard]] torch::Tensor forward(torch::Tensor input);

    [[nodiscard]] torch::Tensor forward(torch::Tensor input, ForwardOptions options);

    struct ForwardActivationCaptureResult {
      torch::Tensor logits{};
      torch::Tensor activation{};
    };

    [[nodiscard]] ForwardActivationCaptureResult forward_with_activation_capture(
        torch::Tensor input,
        torch::nn::Module *target_module,
        ForwardOptions options = {});

    [[nodiscard]] torch::Tensor forward_internal(torch::Tensor input, ForwardOptions options,
        Layer::Details::RegisteredLayer *capture_layer,
        torch::Tensor *captured_activation);

    torch::Tensor execute_plan(torch::Tensor tensor, GraphMode graph_mode,
        Layer::Details::RegisteredLayer *capture_layer = nullptr,
        torch::Tensor *captured_activation = nullptr);


    void set_model_name(std::string name) { model_name_ = std::move(name); }

    [[nodiscard]] std::string model_name() const;

    void save(const std::filesystem::path &directory) const;

    void load(const std::filesystem::path &directory);

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
      const torch::Tensor &aligned_target = target.device() == prediction.device() ? target : target.to(prediction.device());
      return std::visit(
          [&](const auto &descriptor) {
          return Loss::Details::compute(descriptor, prediction, aligned_target, weight);
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

    /// TODO: do it cleaner
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
      void train(Dataset dataset);

    void train(torch::Tensor train_inputs, torch::Tensor train_targets, TrainOptions options = {});

    void reset_training_state();

    [[nodiscard]] torch::MemoryFormat preferred_tensor_memory_format() const noexcept {
      return tensor_memory_format_;
    }


    void set_staging_observer(std::function<void(const torch::Tensor &, bool)> observer) {
      staging_observer_ = std::move(observer);
    }

    private:
    void ensure_optimizer_graph_capability(GraphMode mode) const;

    void prepare_optimizers_for_graph(GraphMode mode);

    public:
    void reset_graph_shape_cache(GraphMode mode) const;

    private:
    void ensure_graph_input_shape(GraphMode mode, const torch::Tensor &tensor) const;

    void ensure_graph_batch_shapes(GraphMode mode,
        const torch::Tensor &inputs,
        const torch::Tensor &targets) const;

    void ensure_graph_replay_ready(GraphMode mode) const;

    void enforce_graph_shape(GraphMode mode,
        const torch::Tensor &tensor,
        std::optional<std::vector<int64_t> > &storage,
        std::string_view tensor_label) const;

    static std::vector<int64_t> tensor_shape_vector(const torch::Tensor &tensor);

    static std::string format_shape_vector(const std::vector<int64_t> &shape);

    static std::string scalar_type_to_string(torch::ScalarType type);

    static GraphTensorSignature describe_tensor_signature(const torch::Tensor &tensor);

    static bool signatures_equal(const GraphTensorSignature &lhs, const GraphTensorSignature &rhs);

    static std::string format_signature(const GraphTensorSignature &signature);

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

    enum class WorkspaceTensorPolicy : std::uint8_t {
      RebindStorage,
      PreserveStorage
    };

    static WorkspaceTensorPolicy workspace_tensor_policy(GraphMode mode) noexcept;

    static void copy_tensor_into(
        torch::Tensor &destination,
        const torch::Tensor &source,
        WorkspaceTensorPolicy policy = WorkspaceTensorPolicy::RebindStorage);

    void record_epoch_telemetry(TrainingTelemetry::EpochSnapshot snapshot);

    public:
    void record_dataset_loss_telemetry(TrainingTelemetry::DatasetLossSnapshot snapshot);

    [[nodiscard]] std::vector<double> collect_learning_rates();

    private:
    [[nodiscard]] torch::Tensor ensure_input_memory_format(torch::Tensor tensor) const;

    [[nodiscard]] torch::Tensor stage_tensor_for_execution(torch::Tensor tensor) const;

    void copy_into_graph_input_buffer(const torch::Tensor &tensor, WorkspaceTensorPolicy policy);

    [[nodiscard]] const torch::Tensor &graph_output_tensor() const noexcept {
      return graph_workspace_.output;
    }

    [[nodiscard]] std::size_t resolve_output_node_index() const noexcept;


    void clear_compiled_graph() noexcept;

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
          const bool too_few_dims_for_2d = tensor_memory_format_ == torch::MemoryFormat::ChannelsLast && parameter.dim() < 4;
          const bool too_few_dims_for_3d = tensor_memory_format_ == torch::MemoryFormat::ChannelsLast3d && parameter.dim() < 5;
          if (too_few_dims_for_2d || too_few_dims_for_3d) {
            memory_format = torch::MemoryFormat::Contiguous;
          }
          // in-place swap; a plain reassign decouples this member from the registered param the optimizer holds
          parameter.set_data(parameter.to(
              parameter.options(), /*non_blocking*/false, /*copy*/false, memory_format));
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

    void invalidate_execution_workspace() noexcept;

    [[nodiscard]] GraphCaptureState &graph_capture_state(GraphExecutionPhase phase) noexcept;

    [[nodiscard]] const GraphCaptureState &graph_capture_state(GraphExecutionPhase phase) const noexcept;

    void invalidate_graph_capture(GraphExecutionPhase phase) noexcept;

    void invalidate_graph_captures() noexcept;

    [[nodiscard]] bool graph_execution_enabled(GraphMode mode, GraphExecutionPhase phase) const;

    void ensure_execution_workspace();

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
      binding.descriptor = descriptor;
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

    public:
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

    private:
    torch::Tensor graph_train_step_impl(torch::Tensor batch_inputs, torch::Tensor batch_targets,
        GraphMode graph_mode, bool regularization_active, bool amp_enabled);

    public:
    void step_scheduler();

    private:
    struct TrainingDetails;

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
    std::vector<std::size_t> node_last_consumer_step_{};
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
    mutable std::optional<torch::ScalarType> cached_autocast_dtype_{};


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

    struct AutocastGuard;

    [[nodiscard]] torch::ScalarType determine_autocast_dtype() const;
  };
}

#include "details/graph_utils.hpp"
#include "details/graph_builder.hpp"
#include "details/train.hpp"
#include "details/executor.hpp"
#include "details/model_io.hpp"
#include "../plot/plot.hpp"

namespace Nott {
  template<class Descriptor, class... Args>
    decltype(auto) Model::plot(Descriptor descriptor, Args &&... args) {
      return Plot::Render(*this,
          std::move(descriptor),
          std::forward<Args>(args)...);
    }
}
#endif //Nott_CORE_HPP
