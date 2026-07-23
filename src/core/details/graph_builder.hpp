#ifndef Nott_CORE_DETAILS_GRAPH_BUILDER_HPP
#define Nott_CORE_DETAILS_GRAPH_BUILDER_HPP

namespace Nott {

  inline void Model::links(std::vector<LinkSpec> specifications, bool enable_graph_capture) {
    LinkParams params{};
    params.enable_graph_capture = enable_graph_capture;
    links(std::move(specifications), std::move(params));
  }

  /// Updated multi-IO + params form
  inline void Model::links(std::vector<LinkSpec> specifications, LinkParams params) {
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

    /// Build N input nodes (@input, @input[1], ...)
    std::vector<std::size_t> input_node_indices(num_inputs, std::numeric_limits<std::size_t>::max());
    for (std::size_t i = 0; i < num_inputs; ++i) {
      CompiledNode n{};
      n.kind = CompiledNode::Kind::Input;
      n.label = "@input";
      if (num_inputs > 1) { n.label += "[" + std::to_string(i) + "]"; }
      nodes.push_back(std::move(n));
      input_node_indices[i] = nodes.size() - 1;
    }

    /// Modules
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

    /// Outputs created lazily; track per-index
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

    /// Multi-IO
    auto resolve_port = [&](Port &port, PortRole role) -> std::size_t {
      switch (port.kind) {
        case Port::Kind::Input: {
                                  /// default single input semantics
                                  if (port.identifier.empty() || iequals(port.identifier, "@input")) {
                                    port.assign_node(input_node_indices.front());
                                    return input_node_indices.front();
                                  }
                                  /// numeric?
                                  if (auto n = parse_numeric_identifier(port.identifier)) {
                                    if (*n >= input_node_indices.size())
                                      throw std::invalid_argument("Input index " + std::to_string(*n) + " out of range.");
                                    port.assign_node(input_node_indices[*n]);
                                    return input_node_indices[*n];
                                  }
                                  /// alias?
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
                                   /// default single output
                                   if (port.identifier.empty() || iequals(port.identifier, "@output")) {
                                     const auto idx = ensure_output_node_k(0);
                                     port.assign_node(idx);
                                     return idx;
                                   }
                                   /// numeric?
                                   if (auto n = parse_numeric_identifier(port.identifier)) {
                                     const auto idx = ensure_output_node_k(*n);
                                     port.assign_node(idx);
                                     return idx;
                                   }
                                   /// alias?
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

    /// track explicit join edges to avoid duplicate inferred edges
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

    /// Apply links
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

    /// Toposort
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
        /// changed
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

    /// Emit execution steps
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

    /// Decide final terminal: if multiple distinct Output nodes referenced, auto-stack to keep single terminal
    std::vector<std::size_t> referenced_outputs;
    for (std::size_t k = 0; k < output_node_indices.size(); ++k)
      if (output_node_indices[k] != std::numeric_limits<std::size_t>::max()) referenced_outputs.push_back(
          output_node_indices[k]);

    std::optional<std::size_t> output_node_index{};
    if (referenced_outputs.size() == 1) {
      output_node_index = referenced_outputs.front();
    } else if (referenced_outputs.size() > 1) {
      /// hidden stack join on dim=1
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
        /// Join exec
        ExecutionStep exJ{};
        exJ.activation_index = join_idx;
        exJ.kind = ExecutionStep::Kind::Join;
        exJ.join.policy = MergePolicy::Stack;
        exJ.join.producers = referenced_outputs;
        exJ.join.workspace_index = buf_idx;
        exJ.join.concat_dimension = static_cast<int64_t>(1);
        execution_steps.push_back(std::move(exJ));
        /// Output exec
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

    std::vector<std::size_t> last_consumer_step(nodes.size(), std::numeric_limits<std::size_t>::max());
    for (std::size_t step_index = 0; step_index < execution_steps.size(); ++step_index) {
      const auto &step = execution_steps[step_index];
      switch (step.kind) {
        case ExecutionStep::Kind::Module:
          last_consumer_step[step.module.input_index] = step_index;
          break;
        case ExecutionStep::Kind::Join:
          for (auto producer: step.join.producers) last_consumer_step[producer] = step_index;
          break;
        case ExecutionStep::Kind::Output:
          last_consumer_step[step.output.input_index] = step_index;
          break;
      }
    }

    compiled_nodes_ = std::move(nodes);
    compiled_steps_ = std::move(steps);
    execution_steps_ = std::move(execution_steps);
    join_buffers_ = std::move(joins);
    compiled_links_ = std::move(resolved_links);
    compiled_output_node_index_ = output_node_index;
    node_last_consumer_step_ = std::move(last_consumer_step);
    routing_active_ = true;
    invalidate_execution_workspace();
  }

  inline void Model::clear_compiled_graph() noexcept {
    routing_active_ = false;
    compiled_nodes_.clear();
    compiled_steps_.clear();
    execution_steps_.clear();
    join_buffers_.clear();
    compiled_links_.clear();
    compiled_output_node_index_.reset();
    node_last_consumer_step_.clear();
    graph_capture_opt_in_ = false;
    invalidate_execution_workspace();
  }

}

#endif //Nott_CORE_DETAILS_GRAPH_BUILDER_HPP
