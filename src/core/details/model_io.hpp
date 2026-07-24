#ifndef Nott_CORE_DETAILS_MODEL_IO_HPP
#define Nott_CORE_DETAILS_MODEL_IO_HPP

namespace Nott {

  namespace Details {
    // module.save() nests each submodule as its own sub-archive, not a flat "fc_0.weight" key
    inline bool read_nested_tensor(torch::serialize::InputArchive &root,
        const std::string &dotted_key, torch::Tensor &out) {
      std::vector<std::string> parts;
      std::size_t start = 0;
      while (true) {
        const auto pos = dotted_key.find('.', start);
        if (pos == std::string::npos) {
          parts.push_back(dotted_key.substr(start));
          break;
        }
        parts.push_back(dotted_key.substr(start, pos - start));
        start = pos + 1;
      }
      if (parts.empty()) {
        return false;
      }

      std::vector<torch::serialize::InputArchive> levels(parts.size() - 1);
      torch::serialize::InputArchive *current = &root;
      for (std::size_t index = 0; index + 1 < parts.size(); ++index) {
        if (!current->try_read(parts[index], levels[index])) {
          return false;
        }
        current = &levels[index];
      }
      return current->try_read(parts.back(), out);
    }
  }

  inline std::string Model::model_name() const {
    if (model_name_.empty()) {
      return this->name();
    }
    return model_name_;
  }

  inline void Model::save(const std::filesystem::path &directory) const {
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

  inline void Model::load(const std::filesystem::path &directory) {
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
      if (!Details::read_nested_tensor(validation_archive, item.key(), stored)) {
        throw std::runtime_error("Checkpoint is missing parameter '" + item.key() + "'.");
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
      if (!Details::read_nested_tensor(validation_archive, item.key(), stored)) {
        throw std::runtime_error("Checkpoint is missing buffer '" + item.key() + "'.");
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

}

#endif //Nott_CORE_DETAILS_MODEL_IO_HPP
