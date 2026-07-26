#ifndef Nott_CORE_DETAILS_TRAIN_HPP
#define Nott_CORE_DETAILS_TRAIN_HPP

// must be included before executor.hpp: graph_train_step_impl needs Model::AutocastGuard complete

namespace Nott {

  struct Model::TrainingDetails {
    struct TensorDataset {
      torch::Tensor inputs;
      torch::Tensor targets;
    };

    [[nodiscard]] static TensorDataset prepare_tensor_dataset(torch::Tensor inputs, const torch::Tensor &targets,
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
      auto prepared_targets = targets.contiguous();
      prepared_inputs = Nott::Training::ensure_pinned(std::move(prepared_inputs));
      prepared_targets = Nott::Training::ensure_pinned(std::move(prepared_targets));
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
      dataset.inputs = Nott::Training::ensure_pinned(std::move(dataset.inputs));
      dataset.targets = Nott::Training::ensure_pinned(std::move(dataset.targets));
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

        return TensorDataset{torch::stack(inputs), torch::stack(targets)};
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
      } else {
        line << "N/A";
      }

      const std::string nabla_symbol{"∇"};
      const std::string grey{kBrightBlack};
      const std::string green{kBrightGreen};
      const std::string reset{kReset};

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

  struct Model::AutocastGuard {
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
        // must clear the autocast weight cache on exit; otherwise the next step reuses stale
        // low-precision weight copies from this forward and the model never learns
        at::autocast::clear_cache();
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

  // autocast compute dtype for AMP; bf16 keeps the fp32 exponent range so no loss scaling
  // is needed. only reached on CUDA (use_amp gates on it).
  inline torch::ScalarType Model::determine_autocast_dtype() const {
    if (cached_autocast_dtype_) {
      return *cached_autocast_dtype_;
    }
    cached_autocast_dtype_ = device_.is_cuda() ? torch::kBFloat16 : torch::kFloat32;
    return *cached_autocast_dtype_;
  }

  template<class Config, class Dataset>
    void Model::train(Dataset dataset) {
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

  inline void Model::train(torch::Tensor train_inputs, torch::Tensor train_targets, TrainOptions options) {
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
      auto dataset = TrainingDetails::prepare_tensor_dataset(std::move(inputs), targets,
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

    if (test_dataset) {
      *test_dataset = TrainingDetails::ensure_contiguous(std::move(*test_dataset),
          effective_options.memory_format);
      *test_dataset = TrainingDetails::ensure_cpu(std::move(*test_dataset), effective_options.memory_format);
    }

    /// Resolve graph mode once before the loop
    const auto req_graph_mode   = effective_options.graph_mode;
    // Auto-enable capture for sequential models: links() sets routing_active_ for
    // multi-IO topologies; sequential models are equally static and don't need it.
    if (req_graph_mode != GraphMode::Disabled && !layers_.empty() && !graph_capture_opt_in_) {
      graph_capture_opt_in_ = true;
    }
    const bool graph_active     = graph_execution_enabled(req_graph_mode, GraphExecutionPhase::Training);
    const GraphMode eff_graph   = graph_active ? req_graph_mode : GraphMode::Disabled;
    const bool graph_enabled    = eff_graph != GraphMode::Disabled;

    if (eff_graph == GraphMode::Capture)
      reset_graph_shape_cache(eff_graph);
    else if (eff_graph == GraphMode::Replay)
      ensure_graph_replay_ready(eff_graph);
    if (graph_enabled)
      ensure_optimizer_graph_capability(eff_graph);

    /// Build training policy
    Training::TrainingPolicy policy = Training::make_training_policy(
        effective_options,
        device_.is_cuda(),
        has_convolutional_layers_,
        has_regularization(),
        /*prefetch_possible=*/device_.is_cuda(),
        channels_last_applicable,
        eff_graph);

    /// CUDA prefetch state
#ifdef TORCH_CUDA_AVAILABLE
    std::optional<Training::PrefetchState> prefetch_state_opt;
    if (device_.is_cuda())
      prefetch_state_opt.emplace(device_.index());
    Training::PrefetchState* prefetch_ptr =
      prefetch_state_opt ? &*prefetch_state_opt : nullptr;
#endif

    const bool regularization_active = has_regularization();
    const bool amp_enabled           = is_amp_training_active();

    /// Graph coordinator (persists across all epochs)
    Training::GraphModeCoordinator graph_coord{eff_graph};

    /// Per-batch training step
    auto training_step = [this, &graph_coord, graph_enabled,
         regularization_active, amp_enabled, &policy]
           (Model& m, torch::Tensor inputs, torch::Tensor targets) -> torch::Tensor
           {
             if (graph_enabled &&
                 static_cast<std::size_t>(inputs.size(0)) != policy.batch_size) {
               if (policy.drop_last) {
                 // Silently skip the incomplete final batch: graph replay requires
                 // a fixed batch size and cannot handle the remainder.
                 return torch::zeros({}, torch::TensorOptions()
                     .dtype(torch::kFloat32).device(inputs.device()));
               }
               throw std::invalid_argument(
                   "Graph optimisation requires every batch to match the captured batch "
                   "size (" + std::to_string(policy.batch_size) + "). Received "
                   + std::to_string(inputs.size(0))
                   + " samples; set drop_last=true or use a batch size that divides "
                   "the dataset evenly.");
             }

             if (graph_coord.requested != GraphMode::Capture) {
               if (graph_enabled) {
                 m.prepare_optimizers_for_graph(graph_coord.requested);
                 m.ensure_graph_batch_shapes(graph_coord.requested, inputs, targets);
               }
               return m.graph_train_step_impl(
                   std::move(inputs), std::move(targets),
                   graph_coord.requested, regularization_active, amp_enabled);
             }

             /// Capture path, may need one retry on shape change.
             bool retry_done = false;
             while (true) {
               const GraphMode batch_mode = graph_coord.resolve(m, inputs, targets);
               m.prepare_optimizers_for_graph(batch_mode);
               m.ensure_graph_batch_shapes(batch_mode, inputs, targets);
               try {
                 auto loss = m.graph_train_step_impl(
                     inputs, targets, batch_mode, regularization_active, amp_enabled);
                 if (batch_mode == GraphMode::Capture)
                   graph_coord.on_captured();
                 return loss;
               } catch (const std::runtime_error&) {
                 if (batch_mode == GraphMode::Replay && !retry_done) {
                   graph_coord.on_replay_failed(m);
                   retry_done = true;
                   continue;
                 }
                 throw;
               }
             }
           };

    /// Epoch-end callback: telemetry + console
    auto on_epoch_end = [&](const Training::EpochLogEntry& entry) {
      auto lrs = collect_learning_rates();

      auto wrap_double = [](double v) {
        auto t = torch::tensor(v, torch::TensorOptions().dtype(torch::kFloat64));
        return TrainingTelemetry::DeferredScalar::from_tensor(
            std::move(t), torch::Device{torch::kCPU});
      };

      std::optional<TrainingTelemetry::DeferredScalar> deferred_test;
      if (entry.test_loss) deferred_test = wrap_double(*entry.test_loss);

      const double latency = entry.processed_steps > 0
        ? entry.duration_seconds / static_cast<double>(entry.processed_steps)
        : 0.0;

      record_epoch_telemetry({
          entry.epoch_index,
          entry.train_loss,        // already a DeferredScalar, no extra wrap
          std::move(deferred_test),
          entry.delta,
          std::move(lrs),
          entry.timestamp,
          entry.duration_seconds,
          wrap_double(latency)
          });

      if (effective_options.monitor && effective_options.stream) {
        TrainingDetails::log_epoch(
            *effective_options.stream,
            entry.epoch_index,
            effective_options.epoch,
            entry.train_loss.materialize(), // resolves lazily; transfer likely done
            entry.test_loss,
            entry.delta,
            entry.improved,
            entry.duration_seconds);
      }
    };

    /// Test-loss callback
    auto compute_test_loss_fn = [&](auto& m, const auto& ds,
        const Training::TrainingPolicy& pol)
      -> std::optional<double>
      {
        return Training::compute_dataset_loss(
            m, ds, pol.batch_size,
            pol.use_buffer(), pol.buffer_vram,
            pol.memory_format);
      };

    std::size_t global_step = 0;

    auto run_training_dataset = [&](typename TrainingDetails::TensorDataset dataset,
        const std::optional<typename TrainingDetails::TensorDataset>&
        evaluation_dataset) {
      Training::run_epochs(
          *this, dataset, evaluation_dataset, policy,
          compute_test_loss_fn,
          on_epoch_end,
          training_step,
          global_step
#ifdef TORCH_CUDA_AVAILABLE
          , prefetch_ptr
#endif
          );
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

  inline void Model::reset_training_state() {
    clear_training_telemetry();
    invalidate_graph_capture(GraphExecutionPhase::Training);
    graph_workspace_.invalidate();
  }

  inline void Model::ensure_optimizer_graph_capability(GraphMode mode) const {
    if (mode == GraphMode::Disabled) {
      return;
    }

    // a scheduler changes the learning rate on the host each step; capture would freeze it
    if (scheduler_) {
      throw std::runtime_error(
          "CUDA graph capture cannot be used with a learning-rate scheduler; the schedule would be frozen at capture time.");
    }

    auto build_error_message = [](const OptimizerBinding &) {
      return std::string(
          "Optimizer does not support CUDA graph capture (its step depends on host-side state such as a step "
          "counter or bias correction); use SGD, or GraphMode::Disabled.");
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

  inline void Model::prepare_optimizers_for_graph(GraphMode mode) {
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

  inline void Model::record_epoch_telemetry(TrainingTelemetry::EpochSnapshot snapshot) {
    telemetry_.append_epoch(std::move(snapshot));
  }

  inline void Model::record_dataset_loss_telemetry(TrainingTelemetry::DatasetLossSnapshot snapshot) {
    telemetry_.append_dataset_loss(std::move(snapshot));
  }

  inline std::vector<double> Model::collect_learning_rates() {
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

  inline void Model::step_scheduler() {
    if (scheduler_) {
      scheduler_->step();
    }
  }

}

#endif //Nott_CORE_DETAILS_TRAIN_HPP
