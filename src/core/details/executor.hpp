#ifndef Nott_CORE_DETAILS_EXECUTOR_HPP
#define Nott_CORE_DETAILS_EXECUTOR_HPP

// needs Model::AutocastGuard complete, so train.hpp must be included first

namespace Nott {

#ifdef TORCH_CUDA_AVAILABLE
  // eager iterations run before a graph capture to warm up kernel selection and allocate state/buffers
  constexpr int kGraphWarmupIters = 3;

  inline bool Model::GraphCaptureState::is_replay_ready() const noexcept {
    return captured && !dirty && graph != nullptr;
  }

  inline void Model::GraphCaptureState::run_replay() {
    if (!capture_stream.has_value()) {
      throw std::runtime_error("CUDA graph replay requested without an associated capture stream.");
    }
    c10::cuda::CUDAStreamGuard guard(*capture_stream);
    graph->replay();
  }

  template <class Fn>
  torch::Tensor Model::GraphCaptureState::run_capture(Fn &&work) {
    if (!graph) {
      graph = std::make_unique<at::cuda::CUDAGraph>();
    } else {
      graph->reset();
    }
    if (!capture_stream.has_value()) {
      capture_stream = c10::cuda::getStreamFromPool();
    }
    captured = false;

    // callers run eager warmup steps first (to settle cuDNN autotune / optimizer state /
    // workspaces); make sure that is finished before recording, since capture cannot autotune
    // or allocate host memory
    torch::cuda::synchronize();

    c10::cuda::CUDAStreamGuard guard(*capture_stream);
    bool capture_started = false;
    try {
      graph->capture_begin();
      capture_started = true;
      auto result = work();
      graph->capture_end();
      capture_started = false;
      captured = true;
      dirty = false;
      // isolate the private graph pool from later work (else a later model can read its memory)
      torch::cuda::synchronize();
      return result;
    } catch (...) {
      if (capture_started) {
        try { graph->capture_end(); } catch (...) {}
      }
      graph->reset();
      capture_stream.reset();
      captured = false;
      dirty = true;
      throw;
    }
  }
#endif

  inline torch::Tensor Model::forward(torch::Tensor input) {
    return forward_internal(std::move(input), {}, nullptr, nullptr);
  }

  inline torch::Tensor Model::forward(torch::Tensor input, ForwardOptions options) {
    return forward_internal(std::move(input), options, nullptr, nullptr);
  }

  inline Model::ForwardActivationCaptureResult Model::forward_with_activation_capture(
      torch::Tensor input,
      torch::nn::Module *target_module,
      ForwardOptions options) {
    if (target_module == nullptr)
      throw std::invalid_argument("forward_with_activation_capture requires a valid module pointer.");

    auto *layer = resolve_registered_layer(target_module);
    if (layer == nullptr)
      throw std::runtime_error("Requested module is not part of the model graph.");

    torch::Tensor captured_activation;
    auto logits = forward_internal(std::move(input), options, layer, &captured_activation);
    if (!captured_activation.defined())
      throw std::runtime_error("Failed to capture activation for the requested module.");

    ForwardActivationCaptureResult result{};
    result.logits = std::move(logits);
    result.activation = std::move(captured_activation);
    return result;
  }

  inline torch::Tensor Model::forward_internal(torch::Tensor input, ForwardOptions options,
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
        if (!state.is_replay_ready()) {
          throw std::runtime_error(
              "CUDA graph replay requested for inference before a capture was recorded.");
        }
        ensure_graph_input_shape(GraphMode::Replay, input);
        ensure_execution_workspace();
        input = stage_tensor_for_execution(std::move(input));
        copy_into_graph_input_buffer(input, workspace_tensor_policy(GraphMode::Replay));
        state.run_replay();
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
        input = stage_tensor_for_execution(std::move(input));
        ensure_graph_input_shape(GraphMode::Capture, input);
        // eager warmup so cuDNN autotune is settled before capture
        for (int i = 0; i < kGraphWarmupIters; ++i) {
          execute(input, GraphMode::Disabled);
        }
        auto result = state.run_capture([&] { return execute(input, GraphMode::Capture); });
        state.loss_buffer = torch::Tensor{};
        return result;
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

  inline torch::Tensor Model::execute_plan(torch::Tensor tensor, GraphMode graph_mode,
      Layer::Details::RegisteredLayer *capture_layer,
      torch::Tensor *captured_activation) {
    tensor = stage_tensor_for_execution(std::move(tensor));
    if (graph_mode == GraphMode::Replay) {
      throw std::logic_error("Model::execute_plan cannot be invoked in replay mode.");
    }

    auto apply_calibrations = [&](torch::Tensor value) {
      if (calibration_methods_.empty()) return value;
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
    copy_into_graph_input_buffer(tensor, workspace_tensor_policy(graph_mode));

    auto &workspace = graph_workspace_;

    const auto output_index = resolve_output_node_index();
#ifndef NDEBUG
    assert(output_index < workspace.node_buffers.size());
#endif
    workspace.bind_output(output_index);

    for (std::size_t step_index = 0; step_index < execution_steps_.size(); ++step_index) {
      const auto &step = execution_steps_[step_index];
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

      // Disabled mode only: Capture/Replay need stable buffer addresses across replays
      if (graph_mode == GraphMode::Disabled) {
        auto release_if_last_use = [&](std::size_t node_index) {
          if (node_index < node_last_consumer_step_.size()
              && node_last_consumer_step_[node_index] == step_index) {
            workspace.node_buffers[node_index] = torch::Tensor{};
          }
        };
        switch (step.kind) {
          case ExecutionStep::Kind::Module: release_if_last_use(step.module.input_index); break;
          case ExecutionStep::Kind::Join: for (auto producer: step.join.producers) release_if_last_use(producer); break;
          case ExecutionStep::Kind::Output: release_if_last_use(step.output.input_index); break;
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
    if (!workspace.output.defined()) {
      /// Fallback: no Output execution step ran; pull directly from the node buffer.
#ifndef NDEBUG
      assert(output_index < workspace.node_buffers.size());
#endif

      auto output_tensor = workspace.node_buffers[output_index];
      if (!output_tensor.defined()) {
        throw std::runtime_error("Model::forward produced an undefined tensor at the output node.");
      }
      copy_tensor_into(workspace.output, output_tensor, workspace_tensor_policy(graph_mode));
      workspace.bind_output(output_index);
    }
    auto result = graph_output_tensor();

    return apply_calibrations(std::move(result));
  }

  inline void Model::reset_graph_shape_cache(GraphMode mode) const {
    if (mode == GraphMode::Disabled) {
      return;
    }

    graph_input_shape_cache_.reset();
    graph_target_shape_cache_.reset();
  }

  inline void Model::ensure_graph_input_shape(GraphMode mode, const torch::Tensor &tensor) const {
    if (mode == GraphMode::Disabled) {
      return;
    }

    enforce_graph_shape(mode, tensor, graph_input_shape_cache_, "input tensor");
  }

  inline void Model::ensure_graph_batch_shapes(GraphMode mode,
      const torch::Tensor &inputs,
      const torch::Tensor &targets) const {
    if (mode == GraphMode::Disabled) {
      return;
    }

    ensure_graph_input_shape(mode, inputs);
    enforce_graph_shape(mode, targets, graph_target_shape_cache_, "target tensor");
  }

  inline void Model::ensure_graph_replay_ready(GraphMode mode) const {
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

  inline void Model::enforce_graph_shape(GraphMode mode,
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

  inline torch::Tensor Model::ensure_input_memory_format(torch::Tensor tensor) const {
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

  inline torch::Tensor Model::stage_tensor_for_execution(torch::Tensor tensor) const {
    if (!tensor.defined()) {
      return tensor;
    }

    // Fast path: already on target device with the expected memory format, no observer.
    if (!staging_observer_ && tensor.device() == device_) {
      const bool ok =
        (tensor_memory_format_ == torch::MemoryFormat::Contiguous &&
         tensor.is_contiguous()) ||
        (tensor_memory_format_ == torch::MemoryFormat::ChannelsLast &&
         tensor.is_contiguous(torch::MemoryFormat::ChannelsLast)) ||
        (tensor_memory_format_ == torch::MemoryFormat::ChannelsLast3d &&
         tensor.is_contiguous(torch::MemoryFormat::ChannelsLast3d));
      if (ok) return tensor;
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
    }

    return tensor;
  }

  inline void Model::copy_into_graph_input_buffer(const torch::Tensor &tensor, WorkspaceTensorPolicy policy) {
    constexpr std::size_t kInputNodeIndex = 0;
    copy_tensor_into(graph_workspace_.input, tensor, policy);
    graph_workspace_.bind_input(kInputNodeIndex);
  }

  inline std::size_t Model::resolve_output_node_index() const noexcept {
    constexpr std::size_t kInputNodeIndex = 0;
    if (compiled_output_node_index_) {
      return *compiled_output_node_index_;
    }
    if (execution_steps_.empty()) {
      return kInputNodeIndex;
    }
    return execution_steps_.back().activation_index;
  }

  inline void Model::invalidate_execution_workspace() noexcept {
    execution_workspace_dirty_ = true;
    graph_workspace_.invalidate();
    cached_layer_pointers_.clear();
    invalidate_graph_captures();
  }

  inline Model::GraphCaptureState &Model::graph_capture_state(GraphExecutionPhase phase) noexcept {
    switch (phase) {
      case GraphExecutionPhase::Training:
        return graph_capture_training_;
      case GraphExecutionPhase::Inference:
        return graph_capture_inference_;
    }
    return graph_capture_training_;
  }

  inline const Model::GraphCaptureState &Model::graph_capture_state(GraphExecutionPhase phase) const noexcept {
    switch (phase) {
      case GraphExecutionPhase::Training:
        return graph_capture_training_;
      case GraphExecutionPhase::Inference:
        return graph_capture_inference_;
    }
    return graph_capture_training_;
  }

  inline void Model::invalidate_graph_capture(GraphExecutionPhase phase) noexcept {
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

  inline void Model::invalidate_graph_captures() noexcept {
    invalidate_graph_capture(GraphExecutionPhase::Training);
    invalidate_graph_capture(GraphExecutionPhase::Inference);
  }

  inline bool Model::graph_execution_enabled(GraphMode mode, GraphExecutionPhase phase) const {
    if (mode == GraphMode::Disabled) {
      return false;
    }
    // Capture is not supported: the training capture threads autograd through fixed node buffers
    // via an in-place copy_, which throws on repeated capture-mode execution and cannot preserve
    // the backward linkage. Fixing it needs a capture that only pins input/output/grad buffers and
    // lets intermediates be ordinary autograd tensors in the graph's private pool.
    if (mode == GraphMode::Capture || mode == GraphMode::Replay) {
      throw std::runtime_error("Nott: GraphMode capture/replay is not supported; use GraphMode::Disabled.");
    }
    if (!graph_capture_opt_in_) {
      return false;
    }
    // routing_active_ is set by links() for multi-IO models; sequential models
    // (layers_ non-empty) are also static and can be captured without explicit links.
    if (!routing_active_ && layers_.empty()) {
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

  inline void Model::ensure_execution_workspace() {
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

  inline torch::Tensor Model::graph_train_step_impl(torch::Tensor batch_inputs, torch::Tensor batch_targets,
      GraphMode graph_mode, bool regularization_active, bool amp_enabled) {
    // Fast path: no graph, no AMP, no regularisation, skip all graph/scaler setup.
#ifdef TORCH_CUDA_AVAILABLE
    if (graph_mode == GraphMode::Disabled && !amp_enabled && !regularization_active) {
      auto prediction = execute_plan(std::move(batch_inputs), GraphMode::Disabled);
      auto loss = compute_loss(prediction, batch_targets);
      if (loss.dim() != 0) loss = loss.mean();
      loss.backward();
      step_optimizers();
      zero_grad();
      loss.detach_();
      return loss;
    }
#endif

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

      // bf16 autocast needs no gradient scaling (fp32 exponent range), so the AMP and
      // non-AMP steps are identical here; autocast is applied around forward via AutocastGuard.
      const bool retain_graph = (mode != GraphMode::Disabled);
      loss.backward({}, retain_graph);
      step_optimizers();

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
                                 ensure_execution_workspace();
                                 // stage inputs and targets to the device before capture begins; the
                                 // device transfer pins host memory, which is illegal once capturing.
                                 batch_inputs = stage_tensor_for_execution(std::move(batch_inputs));
                                 if (batch_targets.defined() && batch_targets.device() != device_) {
                                   batch_targets = batch_targets.to(device_, /*non_blocking=*/device_.is_cuda());
                                 }
                                 ensure_graph_input_shape(GraphMode::Capture, batch_inputs);
                                 copy_into_graph_input_buffer(batch_inputs, workspace_tensor_policy(GraphMode::Capture));
                                 batch_inputs = graph_workspace_.input;

                                 state.target_buffer = batch_targets.detach();
                                 state.target_buffer.requires_grad_(false);
                                 batch_targets = state.target_buffer;

                                 // eager warmup so cuDNN autotune and optimizer state are settled
                                 // before capture (Disabled mode rebinds buffers, no in-place on leaves)
                                 for (int i = 0; i < kGraphWarmupIters; ++i) {
                                   run_training_step(GraphMode::Disabled, graph_workspace_.input, state.target_buffer);
                                 }

                                 try {
                                   auto loss = state.run_capture([&] {
                                     return run_training_step(GraphMode::Capture, graph_workspace_.input,
                                         state.target_buffer);
                                   });
                                   state.loss_buffer = loss.detach();
                                   state.loss_buffer.requires_grad_(false);
                                   loss.detach_();
                                   return state.loss_buffer;
                                 } catch (...) {
                                   state.loss_buffer = torch::Tensor{};
                                   throw;
                                 }
#else
                                 throw std::runtime_error("CUDA graph capture requested but CUDA support is unavailable.");
#endif
                               }
      case GraphMode::Replay: {
#ifdef TORCH_CUDA_AVAILABLE
                                if (!state.is_replay_ready()) {
                                  throw std::runtime_error(
                                      "CUDA graph replay requested for training before a capture was recorded.");
                                }
                                batch_inputs = stage_tensor_for_execution(std::move(batch_inputs));
                                if (batch_targets.defined() && batch_targets.device() != device_) {
                                  batch_targets = batch_targets.to(device_, /*non_blocking=*/device_.is_cuda());
                                }
                                ensure_graph_input_shape(GraphMode::Replay, batch_inputs);
                                if (graph_target_shape_cache_) {
                                  batch_targets = batch_targets.reshape(*graph_target_shape_cache_);
                                }
                                enforce_graph_shape(GraphMode::Replay, batch_targets, graph_target_shape_cache_, "target tensor");
                                ensure_execution_workspace();
                                copy_into_graph_input_buffer(batch_inputs, workspace_tensor_policy(GraphMode::Replay));
                                auto detached_targets = batch_targets.detach();
                                detached_targets.requires_grad_(false);
                                copy_tensor_into(state.target_buffer, detached_targets, workspace_tensor_policy(GraphMode::Replay));
                                state.run_replay();
                                return state.loss_buffer;
#else
                                throw std::runtime_error("CUDA graph replay requested but CUDA support is unavailable.");
#endif
                              }
    }
    return torch::Tensor{};
  }

}

#endif //Nott_CORE_DETAILS_EXECUTOR_HPP
