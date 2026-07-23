#ifndef Nott_CORE_DETAILS_GRAPH_UTILS_HPP
#define Nott_CORE_DETAILS_GRAPH_UTILS_HPP

namespace Nott {

  inline std::vector<int64_t> Model::tensor_shape_vector(const torch::Tensor &tensor) {
    const auto sizes = tensor.sizes();
    return std::vector<int64_t>(sizes.begin(), sizes.end());
  }

  inline std::string Model::format_shape_vector(const std::vector<int64_t> &shape) {
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

  inline std::string Model::scalar_type_to_string(torch::ScalarType type) {
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

  inline Model::GraphTensorSignature Model::describe_tensor_signature(const torch::Tensor &tensor) {
    GraphTensorSignature signature{};
    signature.device = tensor.device();
    signature.dtype = tensor.scalar_type();
    signature.shape = tensor_shape_vector(tensor);
    return signature;
  }

  inline bool Model::signatures_equal(const GraphTensorSignature &lhs, const GraphTensorSignature &rhs) {
    return lhs.device == rhs.device && lhs.dtype == rhs.dtype && lhs.shape == rhs.shape;
  }

  inline std::string Model::format_signature(const GraphTensorSignature &signature) {
    std::ostringstream stream;
    stream << "shape=" << format_shape_vector(signature.shape)
      << ", dtype=" << scalar_type_to_string(signature.dtype)
      << ", device=" << signature.device.str();
    return stream.str();
  }

  inline Model::WorkspaceTensorPolicy Model::workspace_tensor_policy(GraphMode mode) noexcept {
    switch (mode) {
      case GraphMode::Capture:
      case GraphMode::Replay:
        return WorkspaceTensorPolicy::PreserveStorage;
      case GraphMode::Disabled:
      default:
        return WorkspaceTensorPolicy::RebindStorage;
    }
  }

  inline void Model::copy_tensor_into(
      torch::Tensor &destination,
      const torch::Tensor &source,
      WorkspaceTensorPolicy policy) {
    if (!source.defined()) {
      destination = torch::Tensor{};
      return;
    }

    if (policy == WorkspaceTensorPolicy::RebindStorage) {
      /// Rebind the destination handle to share the source's storage
      destination = source;
      return;
    }

    if (!destination.defined()) {
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

}

#endif //Nott_CORE_DETAILS_GRAPH_UTILS_HPP
