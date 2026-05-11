#ifndef Nott_COMMON_SAVE_LOAD_HPP
#define Nott_COMMON_SAVE_LOAD_HPP
#include <algorithm>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

#include "../activation/activation.hpp"
#include "../attention/attention.hpp"
#include "../block/block.hpp"
#include "../block/details/transformers/classic.hpp"
#include "../block/details/transformers/mamba.hpp"
#include "../block/details/transformers/plusplus.hpp"
#include "../common/local.hpp"
#include "../initialization/initialization.hpp"
#include "../layer/layer.hpp"
#include "../optimizer/optimizer.hpp"
#include "../regularization/regularization.hpp"

namespace Nott::Common::SaveLoad {
    using PropertyTree = boost::property_tree::ptree;
    using ModuleDescriptor = std::variant<Layer::Descriptor, Block::Descriptor>;

    struct NamedModuleDescriptor {
        ModuleDescriptor descriptor{};
        std::string name{};

        NamedModuleDescriptor() = default;

        NamedModuleDescriptor(ModuleDescriptor descriptor, std::string name = {})
            : descriptor(std::move(descriptor)), name(std::move(name))
        {}
    };


    PropertyTree serialize_attention(const Attention::Descriptor& descriptor);
    Attention::Descriptor deserialize_attention(const PropertyTree& tree, const std::string& context);
    PropertyTree serialize_layer_descriptor(const Layer::Descriptor& descriptor);
    Layer::Descriptor deserialize_layer_descriptor(const PropertyTree& tree, const std::string& context);

    namespace Detail {
        inline std::string to_lower(std::string value)
        {
            std::transform(value.begin(), value.end(), value.begin(), [](unsigned char character) {
                return static_cast<char>(std::tolower(character));
            });
            return value;
        }

        template <class Numeric>
        Numeric get_numeric(const PropertyTree& tree, const std::string& key, const std::string& context)
        {
            static_assert(std::is_arithmetic_v<Numeric>, "Numeric type required for property tree extraction.");
            const auto value = tree.get_optional<Numeric>(key);
            if (!value) {
                std::ostringstream message;
                message << "Missing numeric field '" << key << "' in " << context;
                throw std::runtime_error(message.str());
            }
            return *value;
        }

        inline bool get_boolean(const PropertyTree& tree, const std::string& key, const std::string& context)
        {
            const auto value = tree.get_optional<bool>(key);
            if (!value) {
                std::ostringstream message;
                message << "Missing boolean field '" << key << "' in " << context;
                throw std::runtime_error(message.str());
            }
            return *value;
        }

        inline std::string get_string(const PropertyTree& tree, const std::string& key, const std::string& context)
        {
            const auto value = tree.get_optional<std::string>(key);
            if (!value) {
                std::ostringstream message;
                message << "Missing string field '" << key << "' in " << context;
                throw std::runtime_error(message.str());
            }
            return *value;
        }

        template <class T>
        std::vector<T> read_array(const PropertyTree& tree, const std::string& context)
        {
            std::vector<T> values;
            values.reserve(tree.size());
            for (const auto& child : tree) {
                try {
                    if constexpr (std::is_same_v<T, bool>) {
                        values.push_back(static_cast<bool>(child.second.get_value<int>()));
                    } else {
                        values.push_back(child.second.get_value<T>());
                    }
                } catch (const boost::property_tree::ptree_bad_data&) {
                    std::ostringstream message;
                    message << "Invalid array element in " << context;
                    throw std::runtime_error(message.str());
                }
            }
            return values;
        }

        template <class T>
        PropertyTree write_array(const std::vector<T>& values)
        {
            PropertyTree array;
            for (const auto& value : values) {
                PropertyTree element;
                element.put("", value);
                array.push_back({"", element});
            }
            return array;
        }

        inline std::string loss_reduction_to_string(Loss::Reduction reduction)
        {
            switch (reduction) {
                case Loss::Reduction::None: return "none";
                case Loss::Reduction::Sum: return "sum";
                case Loss::Reduction::Mean:
                default: return "mean";
            }
        }

        inline Loss::Reduction loss_reduction_from_string(const std::string& value, const std::string& context)
        {
            const auto lowered = to_lower(value);
            if (lowered == "mean") return Loss::Reduction::Mean;
            if (lowered == "sum") return Loss::Reduction::Sum;
            if (lowered == "none") return Loss::Reduction::None;
            std::ostringstream message;
            message << "Unknown loss reduction '" << value << "' in " << context;
            throw std::runtime_error(message.str());
        }

        namespace Mamba = Block::Details::Transformer::Mamba;
        namespace PlusPlus = Block::Details::Transformer::PlusPlus;
        inline Activation::Descriptor deserialize_activation_descriptor(const PropertyTree& tree, const std::string& context);


        inline PropertyTree serialize_mamba_rms_norm_options(const Mamba::RMSNormOptions& options)
        {
            PropertyTree tree;
            tree.put("eps", options.eps);
            tree.put("learnable", options.learnable);
            return tree;
        }

        inline Mamba::RMSNormOptions deserialize_mamba_rms_norm_options(const PropertyTree& tree, const std::string& context)
        {
            Mamba::RMSNormOptions options;
            options.eps = get_numeric<double>(tree, "eps", context + " rms_norm");
            options.learnable = get_boolean(tree, "learnable", context + " rms_norm");
            return options;
        }

        inline PropertyTree serialize_mamba_selective_state_options(const Mamba::SelectiveStateSpaceOptions& options)
        {
            PropertyTree tree;
            tree.put("embed_dim", options.embed_dim);
            tree.put("state_expansion", options.state_expansion);
            tree.put("ssm_layers", static_cast<std::uint64_t>(options.ssm_layers));
            tree.put("conv_kernel_size", static_cast<std::uint64_t>(options.conv_kernel_size));
            tree.put("dropout", options.dropout);
            tree.put("batch_first", options.batch_first);
            return tree;
        }

        inline Mamba::SelectiveStateSpaceOptions deserialize_mamba_selective_state_options(const PropertyTree& tree,
                                                                                          const std::string& context)
        {
            Mamba::SelectiveStateSpaceOptions options;
            options.embed_dim = get_numeric<std::int64_t>(tree, "embed_dim", context + " selective_state");
            options.state_expansion = get_numeric<double>(tree, "state_expansion", context + " selective_state");
            options.ssm_layers = get_numeric<std::int64_t>(tree, "ssm_layers", context + " selective_state");
            options.conv_kernel_size = get_numeric<std::int64_t>(tree, "conv_kernel_size", context + " selective_state");
            options.dropout = get_numeric<double>(tree, "dropout", context + " selective_state");
            options.batch_first = get_boolean(tree, "batch_first", context + " selective_state");
            return options;
        }

        inline PropertyTree serialize_mamba_feed_forward_options(const Mamba::FeedForwardOptions& options)
        {
            PropertyTree tree;
            tree.put("embed_dim", options.embed_dim);
            tree.put("expansion_ratio", options.expansion_ratio);
            tree.put("dropout", options.dropout);
            tree.put("gated", options.gated);
            return tree;
        }

        inline std::string attention_variant_to_string(Attention::Variant variant)
        {
            switch (variant) {
                case Attention::Variant::Full: return "full";
                case Attention::Variant::Causal: return "causal";
            }
            throw std::runtime_error("Unsupported attention variant during serialisation.");
        }
        inline Attention::Variant attention_variant_from_string(const std::string& value)
        {
            const auto lowered = to_lower(value);
            if (lowered == "full") return Attention::Variant::Full;
            if (lowered == "causal") return Attention::Variant::Causal;
            std::ostringstream message;
            message << "Unknown attention variant '" << value << "'.";
            throw std::runtime_error(message.str());
        }
        inline std::string soft_dropout_noise_type_to_string(Layer::Details::SoftDropoutOptions::NoiseType type)
        {
            using NoiseType = Layer::Details::SoftDropoutOptions::NoiseType;
            switch (type) {
                case NoiseType::Gaussian: return "gaussian";
                case NoiseType::Poisson: return "poisson";
                case NoiseType::Dithering: return "dithering";
                case NoiseType::InterleavedGradientNoise: return "interleaved_gradient_noise";
                case NoiseType::BlueNoise: return "blue_noise";
                case NoiseType::Bayer: return "bayer";
            }
            throw std::runtime_error("Unsupported SoftDropout noise type during serialisation.");
        }

        inline Layer::Details::SoftDropoutOptions::NoiseType soft_dropout_noise_type_from_string(const std::string& value)
        {
            using NoiseType = Layer::Details::SoftDropoutOptions::NoiseType;
            const auto lowered = to_lower(value);
            if (lowered == "gaussian") return NoiseType::Gaussian;
            if (lowered == "poisson") return NoiseType::Poisson;
            if (lowered == "dithering") return NoiseType::Dithering;
            if (lowered == "interleaved_gradient_noise") return NoiseType::InterleavedGradientNoise;
            if (lowered == "blue_noise") return NoiseType::BlueNoise;
            if (lowered == "bayer") return NoiseType::Bayer;
            std::ostringstream message;
            message << "Unknown SoftDropout noise type '" << value << "'.";
            throw std::runtime_error(message.str());
        }

        inline std::string reduce_op_to_string(Layer::Details::ReduceOp op)
        {
            switch (op) {
                case Layer::Details::ReduceOp::Sum: return "sum";
                case Layer::Details::ReduceOp::Mean: return "mean";
                case Layer::Details::ReduceOp::Max: return "max";
                case Layer::Details::ReduceOp::Min: return "min";
            }
            throw std::runtime_error("Unsupported reduce operation during serialisation.");
        }

        inline Layer::Details::ReduceOp reduce_op_from_string(const std::string& value)
        {
            const auto lowered = to_lower(value);
            if (lowered == "sum") return Layer::Details::ReduceOp::Sum;
            if (lowered == "mean") return Layer::Details::ReduceOp::Mean;
            if (lowered == "max") return Layer::Details::ReduceOp::Max;
            if (lowered == "min") return Layer::Details::ReduceOp::Min;
            std::ostringstream message;
            message << "Unknown reduce operation '" << value << "'.";
            throw std::runtime_error(message.str());
        }

        inline Mamba::FeedForwardOptions deserialize_mamba_feed_forward_options(const PropertyTree& tree,
                                                                                const std::string& context)
        {
            Mamba::FeedForwardOptions options;
            options.embed_dim = get_numeric<std::int64_t>(tree, "embed_dim", context + " feed_forward");
            options.expansion_ratio = get_numeric<double>(tree, "expansion_ratio", context + " feed_forward");
            options.dropout = get_numeric<double>(tree, "dropout", context + " feed_forward");
            options.gated = get_boolean(tree, "gated", context + " feed_forward");
            return options;
        }

        inline std::string serialize_mamba_normalization_order(Mamba::NormalizationOrder order)
        {
            return order == Mamba::NormalizationOrder::Pre ? "pre" : "post";
        }

        inline Mamba::NormalizationOrder deserialize_mamba_normalization_order(const std::string& value,
                                                                               const std::string& context)
        {
            const auto lowered = to_lower(value);
            if (lowered == "pre") {
                return Mamba::NormalizationOrder::Pre;
            }
            if (lowered == "post") {
                return Mamba::NormalizationOrder::Post;
            }
            std::ostringstream message;
            message << "Unknown normalization order '" << value << "' in " << context;
            throw std::runtime_error(message.str());
        }

        inline PropertyTree serialize_bert_attention_options(const ::Nott::Block::Details::Transformer::Bert::AttentionOptions &o) {
            PropertyTree t;
            t.put("embed_dim", o.embed_dim);
            t.put("num_heads", o.num_heads);
            t.put("dropout", o.dropout);
            t.put("bias", o.bias);
            t.put("batch_first", o.batch_first);
            t.put("variant", attention_variant_to_string(o.variant));
            return t;
        }

        inline PropertyTree serialize_bert_feed_forward_options(const ::Nott::Block::Details::Transformer::Bert::FeedForwardOptions &o) {
            PropertyTree t;
            t.put("embed_dim", o.embed_dim);
            t.put("mlp_ratio", o.mlp_ratio);

            /// If you have a project-level activation serializer, replace the next line with that call.
            /// I serialize the activation descriptor by its 'type' enum value (as integer).
            t.put("activation.type", static_cast<std::uint64_t>(o.activation.type));
            t.put("bias", o.bias);
            return t;
        }

        inline PropertyTree serialize_bert_layer_norm_options(const ::Nott::Block::Details::Transformer::Bert::LayerNormOptions &o) {
            PropertyTree t;
            t.put("eps", o.eps);
            t.put("elementwise_affine", o.elementwise_affine);
            return t;
        }

        inline PropertyTree serialize_bert_embedding_options(const ::Nott::Block::Details::Transformer::Bert::EmbeddingOptions &o) {
            PropertyTree t;
            t.put("vocab_size", o.vocab_size);
            t.put("type_vocab_size", o.type_vocab_size);
            t.put("max_position_embeddings", o.max_position_embeddings);
            t.put("dropout", o.dropout);
            t.put("use_token_type", o.use_token_type);
            t.put("use_position_embeddings", o.use_position_embeddings);
            return t;
        }

        inline PropertyTree serialize_bert_encoder_layer_descriptor(const ::Nott::Block::Details::Transformer::Bert::EncoderLayerDescriptor &d) {
            PropertyTree t;
            t.add_child("attention", serialize_bert_attention_options(d.attention));
            t.add_child("feed_forward", serialize_bert_feed_forward_options(d.feed_forward));
            return t;
        }

        inline PropertyTree serialize_bert_encoder_options(const ::Nott::Block::Details::Transformer::Bert::EncoderOptions &o) {
            PropertyTree t;
            t.put("layers", static_cast<std::uint64_t>(o.layers));
            t.put("embed_dim", o.embed_dim);
            t.add_child("attention", serialize_bert_attention_options(o.attention));
            t.add_child("feed_forward", serialize_bert_feed_forward_options(o.feed_forward));
            t.add_child("layer_norm", serialize_bert_layer_norm_options(o.layer_norm));
            t.add_child("embedding", serialize_bert_embedding_options(o.embedding));
            t.put("residual_dropout", o.residual_dropout);
            t.put("attention_dropout", o.attention_dropout);
            t.put("feed_forward_dropout", o.feed_forward_dropout);
            t.put("pre_norm", o.pre_norm);
            t.put("final_layer_norm", o.final_layer_norm);
            return t;
        }

        inline Activation::Type activation_type_from_index(std::uint64_t value, const std::string& context)
        {
            switch (value) {
                case static_cast<std::uint64_t>(Activation::Type::Identity): return Activation::Type::Identity;
                case static_cast<std::uint64_t>(Activation::Type::ReLU): return Activation::Type::ReLU;
                case static_cast<std::uint64_t>(Activation::Type::Sigmoid): return Activation::Type::Sigmoid;
                case static_cast<std::uint64_t>(Activation::Type::Tanh): return Activation::Type::Tanh;
                case static_cast<std::uint64_t>(Activation::Type::LeakyReLU): return Activation::Type::LeakyReLU;
                case static_cast<std::uint64_t>(Activation::Type::Softmax): return Activation::Type::Softmax;
                case static_cast<std::uint64_t>(Activation::Type::SiLU): return Activation::Type::SiLU;
                case static_cast<std::uint64_t>(Activation::Type::GeLU): return Activation::Type::GeLU;
                case static_cast<std::uint64_t>(Activation::Type::GLU): return Activation::Type::GLU;
                case static_cast<std::uint64_t>(Activation::Type::SwiGLU): return Activation::Type::SwiGLU;
                case static_cast<std::uint64_t>(Activation::Type::dSiLU): return Activation::Type::dSiLU;
                case static_cast<std::uint64_t>(Activation::Type::PSiLU): return Activation::Type::PSiLU;
                case static_cast<std::uint64_t>(Activation::Type::Mish): return Activation::Type::Mish;
                case static_cast<std::uint64_t>(Activation::Type::Swish): return Activation::Type::Swish;
            }
            std::ostringstream message;
            message << "Unknown activation type index '" << value << "' in " << context;
            throw std::runtime_error(message.str());
        }



        inline ::Nott::Block::Details::Transformer::Bert::LayerNormOptions deserialize_bert_layer_norm_options(
            const PropertyTree& tree, const std::string& context)
        {
            ::Nott::Block::Details::Transformer::Bert::LayerNormOptions options;
            options.eps = get_numeric<double>(tree, "eps", context);
            options.elementwise_affine = get_boolean(tree, "elementwise_affine", context);
            return options;
        }

        inline ::Nott::Block::Details::Transformer::Bert::EmbeddingOptions deserialize_bert_embedding_options(
            const PropertyTree& tree, const std::string& context)
        {
            ::Nott::Block::Details::Transformer::Bert::EmbeddingOptions options;
            options.vocab_size = get_numeric<std::int64_t>(tree, "vocab_size", context);
            options.type_vocab_size = get_numeric<std::int64_t>(tree, "type_vocab_size", context);
            options.max_position_embeddings = get_numeric<std::int64_t>(tree, "max_position_embeddings", context);
            options.dropout = get_numeric<double>(tree, "dropout", context);
            options.use_token_type = get_boolean(tree, "use_token_type", context);
            options.use_position_embeddings = get_boolean(tree, "use_position_embeddings", context);
            return options;
        }


        inline ::Nott::Block::Details::Transformer::Bert::AttentionOptions deserialize_bert_attention_options(const PropertyTree& tree, const std::string& context) {
            ::Nott::Block::Details::Transformer::Bert::AttentionOptions options;
            options.embed_dim = get_numeric<std::int64_t>(tree, "embed_dim", context);
            options.num_heads = get_numeric<std::int64_t>(tree, "num_heads", context);
            options.dropout = get_numeric<double>(tree, "dropout", context);
            options.bias = get_boolean(tree, "bias", context);
            options.batch_first = get_boolean(tree, "batch_first", context);
            options.variant = attention_variant_from_string(get_string(tree, "variant", context));
            return options;
        }
        inline ::Nott::Block::Details::Transformer::Bert::FeedForwardOptions deserialize_bert_feed_forward_options(
        const PropertyTree& tree, const std::string& context)
        {
            ::Nott::Block::Details::Transformer::Bert::FeedForwardOptions options;
            options.embed_dim = get_numeric<std::int64_t>(tree, "embed_dim", context);
            options.mlp_ratio = get_numeric<double>(tree, "mlp_ratio", context);
            if (const auto activation_node = tree.get_child_optional("activation")) {
                options.activation = deserialize_activation_descriptor(*activation_node, context + " activation");
            } else {
                const auto activation_index =
                    get_numeric<std::uint64_t>(tree, "activation.type", context + " activation");
                options.activation.type = activation_type_from_index(activation_index, context + " activation");
            }
            options.bias = get_boolean(tree, "bias", context);
            return options;
        }

        inline ::Nott::Block::Details::Transformer::Bert::EncoderLayerDescriptor
        deserialize_bert_encoder_layer_descriptor(const PropertyTree& tree, const std::string& context)
        {
            ::Nott::Block::Details::Transformer::Bert::EncoderLayerDescriptor descriptor;
            descriptor.attention = deserialize_bert_attention_options(tree.get_child("attention"), context + " attention");
            descriptor.feed_forward =
                deserialize_bert_feed_forward_options(tree.get_child("feed_forward"), context + " feed_forward");
            return descriptor;
        }

        inline ::Nott::Block::Details::Transformer::Bert::EncoderOptions deserialize_bert_encoder_options(
            const PropertyTree& tree, const std::string& context)
        {
            ::Nott::Block::Details::Transformer::Bert::EncoderOptions options;
            options.layers = static_cast<std::size_t>(get_numeric<std::uint64_t>(tree, "layers", context));
            options.embed_dim = get_numeric<std::int64_t>(tree, "embed_dim", context);
            options.attention = deserialize_bert_attention_options(tree.get_child("attention"), context + " attention");
            options.feed_forward = deserialize_bert_feed_forward_options(tree.get_child("feed_forward"), context + " feed_forward");
            options.layer_norm = deserialize_bert_layer_norm_options(tree.get_child("layer_norm"), context + " layer_norm");
            options.embedding = deserialize_bert_embedding_options(tree.get_child("embedding"), context + " embedding");
            options.residual_dropout = get_numeric<double>(tree, "residual_dropout", context);
            options.attention_dropout = get_numeric<double>(tree, "attention_dropout", context);
            options.feed_forward_dropout = get_numeric<double>(tree, "feed_forward_dropout", context);
            options.pre_norm = get_boolean(tree, "pre_norm", context);
            options.final_layer_norm = get_boolean(tree, "final_layer_norm", context);
            return options;
        }


        inline std::string activation_type_to_string(Activation::Type type)
        {
            switch (type) {
                case Activation::Type::Identity: return "identity";
                case Activation::Type::ReLU: return "relu";
                case Activation::Type::Sigmoid: return "sigmoid";
                case Activation::Type::Tanh: return "tanh";
                case Activation::Type::LeakyReLU: return "leaky_relu";
                case Activation::Type::Softmax: return "softmax";
                case Activation::Type::SiLU: return "silu";
                case Activation::Type::GeLU: return "gelu";
                case Activation::Type::GLU: return "glu";
                case Activation::Type::SwiGLU: return "swiglu";
                case Activation::Type::dSiLU: return "dsilu";
                case Activation::Type::PSiLU: return "psilu";
                case Activation::Type::Mish: return "mish";
                case Activation::Type::Swish: return "swish";
            }
            throw std::runtime_error("Unsupported activation type during serialisation.");
        }

        inline Activation::Type activation_type_from_string(const std::string& value)
        {
            const auto lowered = to_lower(value);
            if (lowered == "identity") return Activation::Type::Identity;
            if (lowered == "relu") return Activation::Type::ReLU;
            if (lowered == "sigmoid") return Activation::Type::Sigmoid;
            if (lowered == "tanh") return Activation::Type::Tanh;
            if (lowered == "leaky_relu") return Activation::Type::LeakyReLU;
            if (lowered == "softmax") return Activation::Type::Softmax;
            if (lowered == "silu") return Activation::Type::SiLU;
            if (lowered == "gelu") return Activation::Type::GeLU;
            if (lowered == "glu") return Activation::Type::GLU;
            if (lowered == "swiglu") return Activation::Type::SwiGLU;
            if (lowered == "dsilu") return Activation::Type::dSiLU;
            if (lowered == "psilu") return Activation::Type::PSiLU;
            if (lowered == "mish") return Activation::Type::Mish;
            if (lowered == "swish") return Activation::Type::Swish;
            std::ostringstream message;
            message << "Unknown activation type '" << value << "'.";
            throw std::runtime_error(message.str());
        }
        inline Activation::Descriptor deserialize_activation_descriptor(const PropertyTree& tree, const std::string& context)
        {
            Activation::Descriptor descriptor;
            descriptor.type = activation_type_from_string(get_string(tree, "type", context));
            return descriptor;
        }

        inline std::string initialization_type_to_string(Initialization::Type type)
        {
            switch (type) {
                case Initialization::Type::Default: return "default";
                case Initialization::Type::XavierNormal: return "xavier_normal";
                case Initialization::Type::XavierUniform: return "xavier_uniform";
                case Initialization::Type::HeNormal: return "he_normal";
                case Initialization::Type::HeUniform: return "he_uniform";
                case Initialization::Type::ZeroBias: return "zero_bias";
                case Initialization::Type::Dirac: return "dirac";
                case Initialization::Type::Lyapunov: return "lyapunov";
            }
            throw std::runtime_error("Unsupported initialisation type during serialisation.");
        }

        inline Initialization::Type initialization_type_from_string(const std::string& value)
        {
            const auto lowered = to_lower(value);
            if (lowered == "default") return Initialization::Type::Default;
            if (lowered == "xavier_normal") return Initialization::Type::XavierNormal;
            if (lowered == "xavier_uniform") return Initialization::Type::XavierUniform;
            if (lowered == "he_normal") return Initialization::Type::HeNormal;
            if (lowered == "he_uniform") return Initialization::Type::HeUniform;
            if (lowered == "zero_bias") return Initialization::Type::ZeroBias;
            if (lowered == "dirac") return Initialization::Type::Dirac;
            if (lowered == "lyapunov") return Initialization::Type::Lyapunov;
            std::ostringstream message;
            message << "Unknown initialisation type '" << value << "'.";
            throw std::runtime_error(message.str());
        }

        inline std::string s4_initialization_to_string(Layer::Details::S4Initialization initialization)
        {
            switch (initialization) {
                case Layer::Details::S4Initialization::HiPPO: return "hippo";
                case Layer::Details::S4Initialization::S4D: return "s4d";
            }
            throw std::runtime_error("Unsupported S4 initialization during serialisation.");
        }

        inline Layer::Details::S4Initialization s4_initialization_from_string(const std::string& value)
        {
            const auto lowered = to_lower(value);
            if (lowered == "hippo") return Layer::Details::S4Initialization::HiPPO;
            if (lowered == "s4d") return Layer::Details::S4Initialization::S4D;
            std::ostringstream message;
            message << "Unknown S4 initialization '" << value << "'.";
            throw std::runtime_error(message.str());
        }




        namespace Classic = Block::Details::Transformer::Classic;
        namespace EBT = Block::Details::Transformer::EBT;

        inline std::string positional_encoding_type_to_string(Classic::PositionalEncodingType type)
        {
            switch (type) {
                case Classic::PositionalEncodingType::None: return "none";
                case Classic::PositionalEncodingType::Sinusoidal: return "sinusoidal";
                case Classic::PositionalEncodingType::Learned: return "learned";
            }
            throw std::runtime_error("Unsupported positional encoding type during serialisation.");
        }

        inline Classic::PositionalEncodingType positional_encoding_type_from_string(const std::string& value)
        {
            const auto lowered = to_lower(value);
            if (lowered == "none") return Classic::PositionalEncodingType::None;
            if (lowered == "sinusoidal") return Classic::PositionalEncodingType::Sinusoidal;
            if (lowered == "learned") return Classic::PositionalEncodingType::Learned;
            std::ostringstream message;
            message << "Unknown positional encoding type '" << value << "'.";
            throw std::runtime_error(message.str());
        }

        inline PropertyTree serialize_activation_descriptor(const Activation::Descriptor& descriptor)
        {
            PropertyTree tree;
            tree.put("type", activation_type_to_string(descriptor.type));
            return tree;
        }


        inline PropertyTree serialize_initialization_descriptor(const Initialization::Descriptor& descriptor)
        {
            PropertyTree tree;
            tree.put("type", initialization_type_to_string(descriptor.type));
            return tree;
        }

        inline Initialization::Descriptor deserialize_initialization_descriptor(const PropertyTree& tree,
                                                                                const std::string& context)
        {
            Initialization::Descriptor descriptor;
            descriptor.type = initialization_type_from_string(get_string(tree, "type", context));
            return descriptor;
        }

        inline std::string ebt_modality_type_to_string(EBT::ModalityType type)
        {
            switch (type) {
                case EBT::ModalityType::Discrete: return "discrete";
                case EBT::ModalityType::Continuous: return "continuous";
            }
            throw std::runtime_error("Unsupported EBT modality type during serialisation.");
        }

        inline EBT::ModalityType ebt_modality_type_from_string(const std::string& value)
        {
            const auto lowered = to_lower(value);
            if (lowered == "discrete") {
                return EBT::ModalityType::Discrete;
            }
            if (lowered == "continuous") {
                return EBT::ModalityType::Continuous;
            }
            std::ostringstream message;
            message << "Unknown EBT modality type '" << value << "'.";
            throw std::runtime_error(message.str());
        }

        inline PropertyTree serialize_ebt_modality_options(const EBT::ModalityOptions& options)
        {
            PropertyTree tree;
            tree.put("type", ebt_modality_type_to_string(options.type));
            tree.put("vocab_size", static_cast<std::int64_t>(options.vocab_size));
            tree.put("input_dim", static_cast<std::int64_t>(options.input_dim));
            tree.put("embed_dim", static_cast<std::int64_t>(options.embed_dim));
            return tree;
        }

        inline EBT::ModalityOptions deserialize_ebt_modality_options(const PropertyTree& tree, const std::string& context)
        {
            EBT::ModalityOptions options;
            options.type = ebt_modality_type_from_string(get_string(tree, "type", context));
            options.vocab_size = get_numeric<std::int64_t>(tree, "vocab_size", context);
            options.input_dim = get_numeric<std::int64_t>(tree, "input_dim", context);
            options.embed_dim = get_numeric<std::int64_t>(tree, "embed_dim", context);
            return options;
        }

        inline PropertyTree serialize_ebt_energy_options(const EBT::EnergyScorerOptions& options)
        {
            PropertyTree tree;
            tree.put("depth", static_cast<std::uint64_t>(options.depth));
            tree.put("hidden_size", static_cast<std::int64_t>(options.hidden_size));
            tree.put("modality_heads", static_cast<std::int64_t>(options.modality_heads));
            return tree;
        }

        inline EBT::EnergyScorerOptions deserialize_ebt_energy_options(const PropertyTree& tree, const std::string& context)
        {
            EBT::EnergyScorerOptions options;
            options.depth = static_cast<std::size_t>(get_numeric<std::uint64_t>(tree, "depth", context));
            options.hidden_size = get_numeric<std::int64_t>(tree, "hidden_size", context);
            options.modality_heads = get_numeric<std::int64_t>(tree, "modality_heads", context);
            return options;
        }

        inline PropertyTree serialize_ebt_optimizer_options(const EBT::OptimizerOptions& options)
        {
            PropertyTree tree;
            tree.put("learning_rate", options.learning_rate);
            tree.put("momentum", options.momentum);
            tree.put("gradient_clip_norm", options.gradient_clip_norm);
            return tree;
        }

        inline EBT::OptimizerOptions deserialize_ebt_optimizer_options(const PropertyTree& tree, const std::string& context)
        {
            EBT::OptimizerOptions options;
            options.learning_rate = get_numeric<double>(tree, "learning_rate", context);
            options.momentum = get_numeric<double>(tree, "momentum", context);
            options.gradient_clip_norm = get_numeric<double>(tree, "gradient_clip_norm", context);
            return options;
        }

        inline PropertyTree serialize_ebt_refinement_options(const EBT::RefinementOptions& options)
        {
            PropertyTree tree;
            tree.put("max_steps", static_cast<std::uint64_t>(options.max_steps));
            tree.put("tolerance", options.tolerance);
            tree.put("stop_on_plateau", options.stop_on_plateau);
            return tree;
        }

        inline EBT::RefinementOptions deserialize_ebt_refinement_options(const PropertyTree& tree, const std::string& context)
        {
            EBT::RefinementOptions options;
            options.max_steps = static_cast<std::size_t>(get_numeric<std::uint64_t>(tree, "max_steps", context));
            options.tolerance = get_numeric<double>(tree, "tolerance", context);
            options.stop_on_plateau = get_boolean(tree, "stop_on_plateau", context);
            return options;
        }

        inline PropertyTree serialize_classic_attention_options(const Classic::AttentionOptions& options)
        {
            PropertyTree tree;
            tree.put("embed_dim", options.embed_dim);
            tree.put("num_heads", options.num_heads);
            tree.put("harddropout", options.dropout);
            tree.put("bias", options.bias);
            tree.put("batch_first", options.batch_first);
            tree.put("variant", attention_variant_to_string(options.variant));
            return tree;
        }

        inline Classic::AttentionOptions deserialize_classic_attention_options(const PropertyTree& tree,
                                                                               const std::string& context)
        {
            Classic::AttentionOptions options;
            options.embed_dim = get_numeric<std::int64_t>(tree, "embed_dim", context);
            options.num_heads = get_numeric<std::int64_t>(tree, "num_heads", context);
            options.dropout = get_numeric<double>(tree, "harddropout", context);
            options.bias = get_boolean(tree, "bias", context);
            options.batch_first = get_boolean(tree, "batch_first", context);
            options.variant = attention_variant_from_string(get_string(tree, "variant", context));
            return options;
        }

        inline PropertyTree serialize_classic_feed_forward_options(const Classic::FeedForwardOptions& options)
        {
            PropertyTree tree;
            tree.put("embed_dim", options.embed_dim);
            tree.put("mlp_ratio", options.mlp_ratio);
            tree.put("bias", options.bias);
            tree.add_child("activation", serialize_activation_descriptor(options.activation));
            tree.add_child("initialization", serialize_initialization_descriptor(options.initialization));
            return tree;
        }

        inline Classic::FeedForwardOptions deserialize_classic_feed_forward_options(const PropertyTree& tree,
                                                                                    const std::string& context)
        {
            Classic::FeedForwardOptions options;
            options.embed_dim = get_numeric<std::int64_t>(tree, "embed_dim", context);
            options.mlp_ratio = get_numeric<double>(tree, "mlp_ratio", context);
            options.bias = get_boolean(tree, "bias", context);
            options.activation = deserialize_activation_descriptor(tree.get_child("activation"), context);
            options.initialization = deserialize_initialization_descriptor(tree.get_child("initialization"), context);
            return options;
        }

        inline PropertyTree serialize_classic_layer_norm_options(const Classic::LayerNormOptions& options)
        {
            PropertyTree tree;
            tree.put("eps", options.eps);
            tree.put("elementwise_affine", options.elementwise_affine);
            return tree;
        }

        inline Classic::LayerNormOptions deserialize_classic_layer_norm_options(const PropertyTree& tree,
                                                                                const std::string& context)
        {
            Classic::LayerNormOptions options;
            options.eps = get_numeric<double>(tree, "eps", context);
            options.elementwise_affine = get_boolean(tree, "elementwise_affine", context);
            return options;
        }

        inline PropertyTree serialize_classic_positional_encoding_options(const Classic::PositionalEncodingOptions& options)
        {
            PropertyTree tree;
            tree.put("type", positional_encoding_type_to_string(options.type));
            tree.put("harddropout", options.dropout);
            tree.put("max_length", static_cast<std::uint64_t>(options.max_length));
            tree.put("batch_first", options.batch_first);
            return tree;
        }

        inline Classic::PositionalEncodingOptions deserialize_classic_positional_encoding_options(const PropertyTree& tree,
                                                                                                  const std::string& context)
        {
            Classic::PositionalEncodingOptions options;
            options.type = positional_encoding_type_from_string(get_string(tree, "type", context));
            options.dropout = get_numeric<double>(tree, "harddropout", context);
            options.max_length = static_cast<std::size_t>(get_numeric<std::uint64_t>(tree, "max_length", context));
            options.batch_first = get_boolean(tree, "batch_first", context);
            return options;
        }

        inline PropertyTree serialize_transformer_pp_auxiliary_head_options(const PlusPlus::AuxiliaryHeadOptions& options)
        {
            PropertyTree tree;
            tree.put("enabled", options.enabled);
            tree.put("num_classes", options.num_classes);
            tree.put("dropout", options.dropout);
            return tree;
        }

        inline PropertyTree serialize_transformer_pp_hybrid_attention_options(const PlusPlus::HybridAttentionOptions& options)
        {
            PropertyTree tree;
            tree.put("embed_dim", options.embed_dim);
            tree.put("num_heads", options.num_heads);
            tree.put("dropout", options.dropout);
            tree.put("bias", options.bias);
            tree.put("batch_first", options.batch_first);
            tree.put("variant", attention_variant_to_string(options.variant));
            tree.put("use_convolution", options.use_convolution);
            tree.put("convolution_kernel_size", options.convolution_kernel_size);
            tree.put("convolution_groups", options.convolution_groups);
            tree.put("convolution_dropout", options.convolution_dropout);
            return tree;
        }

        inline PropertyTree serialize_transformer_pp_feed_forward_options(const PlusPlus::FeedForwardOptions& options)
        {
            PropertyTree tree;
            tree.put("embed_dim", options.embed_dim);
            tree.put("mlp_ratio", options.mlp_ratio);
            tree.put("bias", options.bias);
            tree.add_child("activation", serialize_activation_descriptor(options.activation));
            tree.add_child("initialization", serialize_initialization_descriptor(options.initialization));
            return tree;
        }

        inline PropertyTree serialize_transformer_pp_layer_norm_options(const PlusPlus::LayerNormOptions& options)
        {
            PropertyTree tree;
            tree.put("eps", options.eps);
            tree.put("elementwise_affine", options.elementwise_affine);
            return tree;
        }

        inline PropertyTree serialize_transformer_pp_hybrid_attention_descriptor(
            const PlusPlus::HybridAttentionDescriptor& descriptor)
        {
            PropertyTree tree;
            tree.add_child("attention", serialize_attention(descriptor.attention));
            tree.put("use_convolution", descriptor.use_convolution);
            tree.put("convolution_kernel_size", descriptor.convolution_kernel_size);
            tree.put("convolution_groups", descriptor.convolution_groups);
            tree.put("convolution_dropout", descriptor.convolution_dropout);
            return tree;
        }

        inline PropertyTree serialize_transformer_pp_encoder_layer_descriptor(const PlusPlus::EncoderLayerDescriptor& layer)
        {
            PropertyTree tree;
            tree.add_child("hybrid_attention", serialize_transformer_pp_hybrid_attention_descriptor(layer.hybrid_attention));
            tree.add_child("attention_dropout", serialize_layer_descriptor(layer.attention_dropout));
            PropertyTree feed_forward_layers;
            for (const auto& ff : layer.feed_forward) {
                feed_forward_layers.push_back({"", serialize_layer_descriptor(ff)});
            }
            tree.add_child("feed_forward", feed_forward_layers);
            tree.add_child("feed_forward_dropout", serialize_layer_descriptor(layer.feed_forward_dropout));
            return tree;
        }

        inline PropertyTree serialize_transformer_pp_decoder_layer_descriptor(const PlusPlus::DecoderLayerDescriptor& layer)
        {
            PropertyTree tree;
            tree.add_child("self_attention", serialize_transformer_pp_hybrid_attention_descriptor(layer.self_attention));
            tree.add_child("self_attention_dropout", serialize_layer_descriptor(layer.self_attention_dropout));
            tree.add_child("cross_attention", serialize_attention(layer.cross_attention));
            tree.add_child("cross_attention_dropout", serialize_layer_descriptor(layer.cross_attention_dropout));
            PropertyTree feed_forward_layers;
            for (const auto& ff : layer.feed_forward) {
                feed_forward_layers.push_back({"", serialize_layer_descriptor(ff)});
            }
            tree.add_child("feed_forward", feed_forward_layers);
            tree.add_child("feed_forward_dropout", serialize_layer_descriptor(layer.feed_forward_dropout));
            return tree;
        }

        inline PlusPlus::AuxiliaryHeadOptions deserialize_transformer_pp_auxiliary_head_options(const PropertyTree& tree,
                                                                                                 const std::string& context)
        {
            PlusPlus::AuxiliaryHeadOptions options;
            options.enabled = get_boolean(tree, "enabled", context);
            options.num_classes = get_numeric<std::int64_t>(tree, "num_classes", context);
            options.dropout = get_numeric<double>(tree, "dropout", context);
            return options;
        }

        inline PlusPlus::HybridAttentionOptions deserialize_transformer_pp_hybrid_attention_options(const PropertyTree& tree,
                                                                                                     const std::string& context)
        {
            PlusPlus::HybridAttentionOptions options;
            options.embed_dim = get_numeric<std::int64_t>(tree, "embed_dim", context);
            options.num_heads = get_numeric<std::int64_t>(tree, "num_heads", context);
            options.dropout = get_numeric<double>(tree, "dropout", context);
            options.bias = get_boolean(tree, "bias", context);
            options.batch_first = get_boolean(tree, "batch_first", context);
            options.variant = attention_variant_from_string(get_string(tree, "variant", context));
            options.use_convolution = get_boolean(tree, "use_convolution", context);
            options.convolution_kernel_size = get_numeric<std::int64_t>(tree, "convolution_kernel_size", context);
            options.convolution_groups = get_numeric<std::int64_t>(tree, "convolution_groups", context);
            options.convolution_dropout = get_numeric<double>(tree, "convolution_dropout", context);
            return options;
        }

        inline PlusPlus::FeedForwardOptions deserialize_transformer_pp_feed_forward_options(const PropertyTree& tree,
                                                                                            const std::string& context)
        {
            PlusPlus::FeedForwardOptions options;
            options.embed_dim = get_numeric<std::int64_t>(tree, "embed_dim", context);
            options.mlp_ratio = get_numeric<double>(tree, "mlp_ratio", context);
            options.bias = get_boolean(tree, "bias", context);
            options.activation = deserialize_activation_descriptor(tree.get_child("activation"), context);
            options.initialization = deserialize_initialization_descriptor(tree.get_child("initialization"), context);
            return options;
        }

        inline PlusPlus::LayerNormOptions deserialize_transformer_pp_layer_norm_options(const PropertyTree& tree,
                                                                                        const std::string& context)
        {
            PlusPlus::LayerNormOptions options;
            options.eps = get_numeric<double>(tree, "eps", context);
            options.elementwise_affine = get_boolean(tree, "elementwise_affine", context);
            return options;
        }

        inline PlusPlus::HybridAttentionDescriptor deserialize_transformer_pp_hybrid_attention_descriptor(
            const PropertyTree& tree, const std::string& context)
        {
            PlusPlus::HybridAttentionDescriptor descriptor;
            descriptor.attention = deserialize_attention(tree.get_child("attention"), context);
            descriptor.use_convolution = get_boolean(tree, "use_convolution", context);
            descriptor.convolution_kernel_size = get_numeric<std::int64_t>(tree, "convolution_kernel_size", context);
            descriptor.convolution_groups = get_numeric<std::int64_t>(tree, "convolution_groups", context);
            descriptor.convolution_dropout = get_numeric<double>(tree, "convolution_dropout", context);
            return descriptor;
        }

        inline PlusPlus::EncoderLayerDescriptor deserialize_transformer_pp_encoder_layer_descriptor(const PropertyTree& tree,
                                                                                                     const std::string& context)
        {
            PlusPlus::EncoderLayerDescriptor layer;
            layer.hybrid_attention =
                deserialize_transformer_pp_hybrid_attention_descriptor(tree.get_child("hybrid_attention"), context);
            layer.attention_dropout = deserialize_layer_descriptor(tree.get_child("attention_dropout"), context);
            for (const auto& node : tree.get_child("feed_forward")) {
                layer.feed_forward.push_back(deserialize_layer_descriptor(node.second, context));
            }
            layer.feed_forward_dropout = deserialize_layer_descriptor(tree.get_child("feed_forward_dropout"), context);
            return layer;
        }

        inline PlusPlus::DecoderLayerDescriptor deserialize_transformer_pp_decoder_layer_descriptor(const PropertyTree& tree,
                                                                                                     const std::string& context)
        {
            PlusPlus::DecoderLayerDescriptor layer;
            layer.self_attention =
                deserialize_transformer_pp_hybrid_attention_descriptor(tree.get_child("self_attention"), context);
            layer.self_attention_dropout = deserialize_layer_descriptor(tree.get_child("self_attention_dropout"), context);
            layer.cross_attention = deserialize_attention(tree.get_child("cross_attention"), context);
            layer.cross_attention_dropout =
                deserialize_layer_descriptor(tree.get_child("cross_attention_dropout"), context);
            for (const auto& node : tree.get_child("feed_forward")) {
                layer.feed_forward.push_back(deserialize_layer_descriptor(node.second, context));
            }
            layer.feed_forward_dropout = deserialize_layer_descriptor(tree.get_child("feed_forward_dropout"), context);
            return layer;
        }

        inline PropertyTree serialize_perceiver_attention_options(
        const ::Nott::Block::Details::Transformer::Perceiver::AttentionOptions &o)
        {
            PropertyTree t;
            t.put("query_dim", o.query_dim);
            t.put("key_dim", o.key_dim);
            t.put("num_heads", o.num_heads);
            t.put("dropout", o.dropout);
            t.put("bias", o.bias);
            t.put("batch_first", o.batch_first);
            return t;
        }

        inline PropertyTree serialize_perceiver_feed_forward_options(
            const ::Nott::Block::Details::Transformer::Perceiver::FeedForwardOptions &o)
        {
            PropertyTree t;
            t.put("embed_dim", o.embed_dim);
            t.put("mlp_ratio", o.mlp_ratio);

            /// If you have a project-level activation serializer, prefer that:
            // t.add_child("activation", serialize_activation_descriptor(o.activation));
            t.put("activation.type", static_cast<std::uint64_t>(o.activation.type));

            t.put("bias", o.bias);
            t.put("dropout", o.dropout);
            return t;
        }

        inline PropertyTree serialize_perceiver_encoder_layer_descriptor(
            const ::Nott::Block::Details::Transformer::Perceiver::EncoderLayerDescriptor &d)
        {
            PropertyTree t;
            t.add_child("feed_forward", serialize_perceiver_feed_forward_options(d.feed_forward));
            return t;
        }

        inline PropertyTree serialize_perceiver_encoder_options(
            const ::Nott::Block::Details::Transformer::Perceiver::EncoderOptions &o)
        {
            PropertyTree t;
            t.put("layers", static_cast<std::uint64_t>(o.layers));
            t.put("self_layers", static_cast<std::uint64_t>(o.self_layers));
            t.put("repeats", static_cast<std::uint64_t>(o.repeats));
            t.put("latent_dim", o.latent_dim);
            t.put("input_dim", o.input_dim);
            t.put("latent_slots", static_cast<std::uint64_t>(o.latent_slots));
            t.add_child("cross_attention", serialize_perceiver_attention_options(o.cross_attention));
            t.add_child("self_attention", serialize_perceiver_attention_options(o.self_attention));
            t.add_child("feed_forward", serialize_perceiver_feed_forward_options(o.feed_forward));
            t.put("residual_dropout", o.residual_dropout);
            t.put("attention_dropout", o.attention_dropout);
            return t;
        }

        inline std::string serialize_vision_variant(::Nott::Block::Details::Transformer::Vision::Variant v) {
            switch (v) {
                case ::Nott::Block::Details::Transformer::Vision::Variant::ViT:  return "vit";
                case ::Nott::Block::Details::Transformer::Vision::Variant::Swin: return "swin";
                default: return "unknown";
            }
        }

        inline PropertyTree serialize_vision_attention_options(
            const ::Nott::Block::Details::Transformer::Vision::AttentionOptions &o)
        {
            PropertyTree t;
            t.put("embed_dim", o.embed_dim);
            t.put("num_heads", o.num_heads);
            t.put("dropout", o.dropout);
            t.put("bias", o.bias);
            t.put("batch_first", o.batch_first);
            t.put("variant", Detail::attention_variant_to_string(o.variant));
            return t;
        }

        inline PropertyTree serialize_vision_feed_forward_options(
            const ::Nott::Block::Details::Transformer::Vision::FeedForwardOptions &o)
        {
            PropertyTree t;
            t.put("embed_dim", o.embed_dim);
            t.put("mlp_ratio", o.mlp_ratio);

            /// Prefer calling your project-level activation serializer if present:
            // t.add_child("activation", Detail::serialize_activation_descriptor(o.activation));
            t.put("activation.type", static_cast<std::uint64_t>(o.activation.type));
            t.put("bias", o.bias);
            return t;
        }

        inline PropertyTree serialize_vision_layer_norm_options(
            const ::Nott::Block::Details::Transformer::Vision::LayerNormOptions &o)
        {
            PropertyTree t;
            t.put("eps", o.eps);
            t.put("elementwise_affine", o.elementwise_affine);
            return t;
        }

        inline PropertyTree serialize_vision_patch_embedding_options(
            const ::Nott::Block::Details::Transformer::Vision::PatchEmbeddingOptions &o)
        {
            PropertyTree t;
            t.put("in_channels", o.in_channels);
            t.put("embed_dim", o.embed_dim);
            t.put("patch_size", o.patch_size);
            t.put("add_class_token", o.add_class_token);
            t.put("normalize", o.normalize);
            t.put("dropout", o.dropout);
            return t;
        }

        inline PropertyTree serialize_vision_window_options(
            const ::Nott::Block::Details::Transformer::Vision::WindowOptions &o)
        {
            PropertyTree t;
            t.put("size", o.size);
            t.put("shift", o.shift);
            return t;
        }

        /// Positional encoding serializer, simple mapping, safe fallback if you don't already have one.
        inline std::string serialize_positional_encoding_type(::Nott::Layer::Details::PositionalEncodingType t) {
            switch (t) {
                case ::Nott::Layer::Details::PositionalEncodingType::None: return "none";
                case ::Nott::Layer::Details::PositionalEncodingType::Sinusoidal: return "sinusoidal";
                case ::Nott::Layer::Details::PositionalEncodingType::Learned: return "learned";
                default: return "unknown";
            }
        }

        inline PropertyTree serialize_positional_encoding_options(
            const ::Nott::Layer::Details::PositionalEncodingOptions &o)
        {
            PropertyTree t;
            t.put("type", serialize_positional_encoding_type(o.type));
            t.put("dropout", o.dropout);
            // if PositionalEncodingOptions contains extra fields, add them here
            return t;
        }

        inline PropertyTree serialize_vision_encoder_layer_descriptor(
            const ::Nott::Block::Details::Transformer::Vision::EncoderLayerDescriptor &d)
        {
            PropertyTree t;
            t.add_child("attention", serialize_vision_attention_options(d.attention));
            t.add_child("feed_forward", serialize_vision_feed_forward_options(d.feed_forward));
            t.add_child("window", serialize_vision_window_options(d.window));
            return t;
        }

        inline PropertyTree serialize_vision_encoder_options(
            const ::Nott::Block::Details::Transformer::Vision::EncoderOptions &o)
        {
            PropertyTree t;
            t.put("layers", static_cast<std::uint64_t>(o.layers));
            t.put("embed_dim", o.embed_dim);
            t.put("variant", serialize_vision_variant(o.variant));
            t.add_child("attention", serialize_vision_attention_options(o.attention));
            t.add_child("feed_forward", serialize_vision_feed_forward_options(o.feed_forward));
            t.add_child("layer_norm", serialize_vision_layer_norm_options(o.layer_norm));
            t.add_child("patch_embedding", serialize_vision_patch_embedding_options(o.patch_embedding));
            t.add_child("window", serialize_vision_window_options(o.window));
            t.add_child("positional_encoding", serialize_positional_encoding_options(o.positional_encoding));
            t.put("residual_dropout", o.residual_dropout);
            t.put("attention_dropout", o.attention_dropout);
            t.put("feed_forward_dropout", o.feed_forward_dropout);
            t.put("pre_norm", o.pre_norm);
            t.put("final_layer_norm", o.final_layer_norm);
            return t;
        }

        inline PropertyTree serialize_longformer_attention_options(
    const ::Nott::Block::Details::Transformer::LongformerXL::AttentionOptions &o)
        {
            PropertyTree t;
            t.put("embed_dim", o.embed_dim);
            t.put("num_heads", o.num_heads);
            t.put("dropout", o.dropout);
            t.put("bias", o.bias);
            t.put("batch_first", o.batch_first);
            return t;
        }

        inline PropertyTree serialize_longformer_feed_forward_options(
            const ::Nott::Block::Details::Transformer::LongformerXL::FeedForwardOptions &o)
        {
            PropertyTree t;
            t.put("embed_dim", o.embed_dim);
            t.put("mlp_ratio", o.mlp_ratio);

            /// Prefer your project-level activation serializer if present:
            // t.add_child("activation", serialize_activation_descriptor(o.activation));
            t.put("activation.type", static_cast<std::uint64_t>(o.activation.type));

            t.put("bias", o.bias);
            t.put("dropout", o.dropout);
            return t;
        }

        inline PropertyTree serialize_longformer_layer_norm_options(
            const ::Nott::Block::Details::Transformer::LongformerXL::LayerNormOptions &o)
        {
            PropertyTree t;
            t.put("eps", o.eps);
            t.put("elementwise_affine", o.elementwise_affine);
            return t;
        }

        inline PropertyTree serialize_longformer_encoder_layer_descriptor(
            const ::Nott::Block::Details::Transformer::LongformerXL::EncoderLayerDescriptor &d)
        {
            PropertyTree t;
            t.add_child("attention", serialize_longformer_attention_options(d.attention));
            t.add_child("feed_forward", serialize_longformer_feed_forward_options(d.feed_forward));
            return t;
        }

        inline PropertyTree serialize_longformer_encoder_options(
            const ::Nott::Block::Details::Transformer::LongformerXL::EncoderOptions &o)
        {
            PropertyTree t;
            t.put("layers", static_cast<std::uint64_t>(o.layers));
            t.put("embed_dim", o.embed_dim);
            t.add_child("attention", serialize_longformer_attention_options(o.attention));
            t.add_child("feed_forward", serialize_longformer_feed_forward_options(o.feed_forward));
            t.add_child("layer_norm", serialize_longformer_layer_norm_options(o.layer_norm));
            t.put("window_size", o.window_size);
            t.put("global_tokens", static_cast<std::uint64_t>(o.global_tokens));
            t.put("causal", o.causal);
            t.put("use_memory", o.use_memory);
            t.put("memory_size", static_cast<std::uint64_t>(o.memory_size));
            t.put("residual_dropout", o.residual_dropout);
            t.put("attention_dropout", o.attention_dropout);
            t.put("feed_forward_dropout", o.feed_forward_dropout);
            t.put("pre_norm", o.pre_norm);
            t.put("final_layer_norm", o.final_layer_norm);
            return t;
        }




        inline std::string pooling_variant_to_string(const Layer::Details::PoolingOptions& options)
        {
            return std::visit(
                [](const auto& concrete) -> std::string {
                    using OptionType = std::decay_t<decltype(concrete)>;
                    if constexpr (std::is_same_v<OptionType, Layer::Details::MaxPool1dOptions>) {
                        return "max1d";
                    } else if constexpr (std::is_same_v<OptionType, Layer::Details::AvgPool1dOptions>) {
                        return "avg1d";
                    } else if constexpr (std::is_same_v<OptionType, Layer::Details::AdaptiveAvgPool1dOptions>) {
                        return "adaptive_avg1d";
                    } else if constexpr (std::is_same_v<OptionType, Layer::Details::AdaptiveMaxPool1dOptions>) {
                        return "adaptive_max1d";
                    } else if constexpr (std::is_same_v<OptionType, Layer::Details::MaxPool2dOptions>) {
                        return "max2d";
                    } else if constexpr (std::is_same_v<OptionType, Layer::Details::AvgPool2dOptions>) {
                        return "avg2d";
                    } else if constexpr (std::is_same_v<OptionType, Layer::Details::AdaptiveAvgPool2dOptions>) {
                        return "adaptive_avg2d";
                    } else if constexpr (std::is_same_v<OptionType, Layer::Details::AdaptiveMaxPool2dOptions>) {
                        return "adaptive_max2d";
                    } else {
                        static_assert(sizeof(OptionType) == 0, "Unsupported pooling options variant.");
                    }
                },
                options);
        }

        inline Layer::Details::PoolingOptions pooling_variant_from_string(const std::string& value)
        {
            const auto lowered = to_lower(value);
            if (lowered == "max1d") return Layer::Details::PoolingOptions{Layer::Details::MaxPool1dOptions{}};
            if (lowered == "avg1d") return Layer::Details::PoolingOptions{Layer::Details::AvgPool1dOptions{}};
            if (lowered == "adaptive_avg1d") return Layer::Details::PoolingOptions{Layer::Details::AdaptiveAvgPool1dOptions{}};
            if (lowered == "adaptive_max1d") return Layer::Details::PoolingOptions{Layer::Details::AdaptiveMaxPool1dOptions{}};
            if (lowered == "max2d") return Layer::Details::PoolingOptions{Layer::Details::MaxPool2dOptions{}};
            if (lowered == "avg2d") return Layer::Details::PoolingOptions{Layer::Details::AvgPool2dOptions{}};
            if (lowered == "adaptive_avg2d") return Layer::Details::PoolingOptions{Layer::Details::AdaptiveAvgPool2dOptions{}};
            if (lowered == "adaptive_max2d") return Layer::Details::PoolingOptions{Layer::Details::AdaptiveMaxPool2dOptions{}};
            std::ostringstream message;
            message << "Unknown pooling variant '" << value << "'.";
            throw std::runtime_error(message.str());
        }
        inline std::string upsample_mode_to_string(::Nott::UpsampleMode mode)
        {
            switch (mode) {
                case ::Nott::UpsampleMode::Bilinear: return "bilinear";
                case ::Nott::UpsampleMode::Bicubic: return "bicubic";
                case ::Nott::UpsampleMode::Nearest:
                default: return "nearest";
            }
        }

        inline ::Nott::UpsampleMode upsample_mode_from_string(const std::string& value, const std::string& context)
        {
            const auto lowered = to_lower(value);
            if (lowered == "nearest") return ::Nott::UpsampleMode::Nearest;
            if (lowered == "bilinear") return ::Nott::UpsampleMode::Bilinear;
            if (lowered == "bicubic") return ::Nott::UpsampleMode::Bicubic;
            std::ostringstream message;
            message << "Unknown upsample mode '" << value << "' in " << context;
            throw std::runtime_error(message.str());
        }

        inline std::string downsample_mode_to_string(::Nott::DownsampleMode mode)
        {
            switch (mode) {
                case ::Nott::DownsampleMode::Bilinear: return "bilinear";
                case ::Nott::DownsampleMode::Bicubic: return "bicubic";
                case ::Nott::DownsampleMode::Nearest:
                default: return "nearest";
            }
        }

        inline ::Nott::DownsampleMode downsample_mode_from_string(const std::string& value, const std::string& context)
        {
            const auto lowered = to_lower(value);
            if (lowered == "nearest") return ::Nott::DownsampleMode::Nearest;
            if (lowered == "bilinear") return ::Nott::DownsampleMode::Bilinear;
            if (lowered == "bicubic") return ::Nott::DownsampleMode::Bicubic;
            std::ostringstream message;
            message << "Unknown downsample mode '" << value << "' in " << context;
            throw std::runtime_error(message.str());
        }

        inline std::string manifold_kind_to_string(Optimizer::Details::ManifoldKind kind)
        {
            switch (kind) {
                case Optimizer::Details::ManifoldKind::Euclidean:
                    return "euclidean";
                case Optimizer::Details::ManifoldKind::UnitSphere:
                    return "unit_sphere";
                case Optimizer::Details::ManifoldKind::Stiefel:
                    return "stiefel";
            }
            throw std::logic_error("Unknown Muon manifold kind encountered during serialization.");
        }

        inline Optimizer::Details::ManifoldKind manifold_kind_from_string(const std::string& value,
                                                                          const std::string& context)
        {
            const auto lowered = to_lower(value);
            if (lowered == "euclidean") return Optimizer::Details::ManifoldKind::Euclidean;
            if (lowered == "unit_sphere") return Optimizer::Details::ManifoldKind::UnitSphere;
            if (lowered == "stiefel") return Optimizer::Details::ManifoldKind::Stiefel;
            std::ostringstream message;
            message << "Unknown Muon manifold kind '" << value << "' in " << context;
            throw std::runtime_error(message.str());
        }


    }

    inline PropertyTree serialize_optimizer(const Optimizer::Descriptor& descriptor)
    {
        PropertyTree tree;
        std::visit(
            [&](const auto& concrete) {
                using DescriptorType = std::decay_t<decltype(concrete)>;
                if constexpr (std::is_same_v<DescriptorType, Optimizer::Details::SGDDescriptor>) {
                    const auto& options = concrete.options;
                    tree.put("type", "sgd");
                    tree.put("options.learning_rate", options.learning_rate);
                    tree.put("options.momentum", options.momentum);
                    tree.put("options.dampening", options.dampening);
                    tree.put("options.weight_decay", options.weight_decay);
                    tree.put("options.nesterov", options.nesterov);
                    tree.put("options.maximize", options.maximize);
                } else if constexpr (std::is_same_v<DescriptorType, Optimizer::Details::AdamDescriptor>) {
                    const auto& options = concrete.options;
                    tree.put("type", "adam");
                    tree.put("options.learning_rate", options.learning_rate);
                    tree.put("options.beta1", options.beta1);
                    tree.put("options.beta2", options.beta2);
                    tree.put("options.eps", options.eps);
                    tree.put("options.weight_decay", options.weight_decay);
                    tree.put("options.amsgrad", options.amsgrad);
                } else if constexpr (std::is_same_v<DescriptorType, Optimizer::Details::AdamWDescriptor>) {
                    const auto& options = concrete.options;
                    tree.put("type", "adamw");
                    tree.put("options.learning_rate", options.learning_rate);
                    tree.put("options.beta1", options.beta1);
                    tree.put("options.beta2", options.beta2);
                    tree.put("options.eps", options.eps);
                    tree.put("options.weight_decay", options.weight_decay);
                    tree.put("options.amsgrad", options.amsgrad);
                } else if constexpr (std::is_same_v<DescriptorType, Optimizer::Details::SophiaGDescriptor>) {
                    const auto& options = concrete.options;
                    tree.put("type", "sophia_g");
                    tree.put("options.learning_rate", options.lr());
                    tree.put("options.beta1", options.beta1());
                    tree.put("options.beta2", options.beta2());
                    tree.put("options.rho", options.rho());
                    tree.put("options.eps", options.eps());
                    tree.put("options.weight_decay", options.weight_decay());
                    tree.put("options.clip", options.clip());
                    tree.put("options.hessian_update_interval", options.hessian_update_interval());
                } else if constexpr (std::is_same_v<DescriptorType, Optimizer::Details::SophiaHDescriptor>) {
                    const auto& options = concrete.options;
                    tree.put("type", "sophia_h");
                    tree.put("options.learning_rate", options.lr());
                    tree.put("options.beta1", options.beta1());
                    tree.put("options.beta2", options.beta2());
                    tree.put("options.rho", options.rho());
                    tree.put("options.eps", options.eps());
                    tree.put("options.weight_decay", options.weight_decay());
                    tree.put("options.clip", options.clip());
                    tree.put("options.hessian_update_interval", options.hessian_update_interval());
                } else if constexpr (std::is_same_v<DescriptorType, Optimizer::Details::MuonDescriptor>) {
                    const auto& options = concrete.options;
                    tree.put("type", "muon");
                    tree.put("options.learning_rate", options.lr());
                    tree.put("options.beta", options.beta());
                    tree.put("options.weight_decay", options.weight_decay());
                    tree.put("options.eps", options.eps());
                    tree.put("options.max_update_norm", options.max_update_norm());
                } else if constexpr (std::is_same_v<DescriptorType, Optimizer::Details::AdaMuonDescriptor>) {
                    const auto& options = concrete.options;
                    tree.put("type", "ada_muon");
                    tree.put("options.learning_rate", options.lr());
                    tree.put("options.beta", options.beta());
                    tree.put("options.beta2", options.beta2());
                    tree.put("options.weight_decay", options.weight_decay());
                    tree.put("options.eps", options.eps());
                    tree.put("options.max_update_norm", options.max_update_norm());
                } else if constexpr (std::is_same_v<DescriptorType, Optimizer::Details::MuonManifoldDescriptor>) {
                    const auto& options = concrete.options;
                    tree.put("type", "muon_manifold");
                    tree.put("options.learning_rate", options.lr());
                    tree.put("options.beta", options.beta());
                    tree.put("options.beta2", options.beta2());
                    tree.put("options.weight_decay", options.weight_decay());
                    tree.put("options.eps", options.eps());
                    tree.put("options.max_update_norm", options.max_update_norm());
                    tree.put("options.retraction_epsilon", options.retraction_epsilon());
                    tree.put("options.renormalize", options.renormalize());
                    tree.put("options.manifold", Detail::manifold_kind_to_string(options.manifold()));
                } else if constexpr (std::is_same_v<DescriptorType, Optimizer::Details::AdafactorDescriptor>) {
                    const auto& options = concrete.options;
                    tree.put("type", "adafactor");
                    tree.put("options.learning_rate", options.lr());
                    tree.put("options.eps1", options.eps1());
                    tree.put("options.eps2", options.eps2());
                    tree.put("options.clip_threshold", options.clip_threshold());
                    tree.put("options.decay_rate", options.decay_rate());
                    tree.put("options.beta1", options.beta1());
                    tree.put("options.use_first_moment", options.use_first_moment());
                    tree.put("options.weight_decay", options.weight_decay());
                    tree.put("options.scale_parameter", options.scale_parameter());
                    tree.put("options.relative_step", options.relative_step());
                    tree.put("options.warmup_init", options.warmup_init());
                } else if constexpr (std::is_same_v<DescriptorType, Optimizer::Details::AdagradDescriptor>) {
                    const auto& options = concrete.options;
                    tree.put("type", "adagrad");
                    tree.put("options.learning_rate", options.learning_rate);
                    tree.put("options.lr_decay", options.lr_decay);
                    tree.put("options.weight_decay", options.weight_decay);
                    tree.put("options.initial_accumulator_value", options.initial_accumulator_value);
                    tree.put("options.eps", options.eps);
                } else if constexpr (std::is_same_v<DescriptorType, Optimizer::Details::LAMBDescriptor>) {
                    const auto& options = concrete.options;
                    tree.put("type", "lamb");
                    tree.put("options.learning_rate", options.lr());
                    tree.put("options.beta1", options.beta1());
                    tree.put("options.beta2", options.beta2());
                    tree.put("options.eps", options.eps());
                    tree.put("options.weight_decay", options.weight_decay());
                    tree.put("options.adam", options.adam());
                } else if constexpr (std::is_same_v<DescriptorType, Optimizer::Details::LionDescriptor>) {
                    const auto& options = concrete.options;
                    tree.put("type", "lion");
                    tree.put("options.learning_rate", options.lr());
                    tree.put("options.beta1", options.beta1());
                    tree.put("options.beta2", options.beta2());
                    tree.put("options.weight_decay", options.weight_decay());
                } else if constexpr (std::is_same_v<DescriptorType, Optimizer::Details::RMSpropDescriptor>) {
                    const auto& options = concrete.options;
                    tree.put("type", "rmsprop");
                    tree.put("options.learning_rate", options.learning_rate);
                    tree.put("options.alpha", options.alpha);
                    tree.put("options.eps", options.eps);
                    tree.put("options.weight_decay", options.weight_decay);
                    tree.put("options.momentum", options.momentum);
                    tree.put("options.centered", options.centered);
                } else {
                    static_assert(sizeof(DescriptorType) == 0, "Unsupported optimizer descriptor supplied.");
                }
            },
            descriptor);
        return tree;
    }

    inline Optimizer::Descriptor deserialize_optimizer(const PropertyTree& tree, const std::string& context)
    {
        const auto type = Detail::to_lower(Detail::get_string(tree, "type", context));
        if (type == "sgd") {
            Optimizer::Details::SGDOptions options;
            options.learning_rate = Detail::get_numeric<double>(tree, "options.learning_rate", context);
            options.momentum = Detail::get_numeric<double>(tree, "options.momentum", context);
            options.dampening = Detail::get_numeric<double>(tree, "options.dampening", context);
            options.weight_decay = Detail::get_numeric<double>(tree, "options.weight_decay", context);
            options.nesterov = Detail::get_boolean(tree, "options.nesterov", context);
            options.maximize = Detail::get_boolean(tree, "options.maximize", context);
            return Optimizer::Descriptor{Optimizer::Details::SGDDescriptor{options}};
        } if (type == "adam") {
            Optimizer::Details::AdamOptions options;
            options.learning_rate = Detail::get_numeric<double>(tree, "options.learning_rate", context);
            options.beta1 = Detail::get_numeric<double>(tree, "options.beta1", context);
            options.beta2 = Detail::get_numeric<double>(tree, "options.beta2", context);
            options.eps = Detail::get_numeric<double>(tree, "options.eps", context);
            options.weight_decay = Detail::get_numeric<double>(tree, "options.weight_decay", context);
            options.amsgrad = Detail::get_boolean(tree, "options.amsgrad", context);
            return Optimizer::Descriptor{Optimizer::Details::AdamDescriptor{options}};
        } if (type == "adamw") {
            Optimizer::Details::AdamWOptions options;
            options.learning_rate = Detail::get_numeric<double>(tree, "options.learning_rate", context);
            options.beta1 = Detail::get_numeric<double>(tree, "options.beta1", context);
            options.beta2 = Detail::get_numeric<double>(tree, "options.beta2", context);
            options.eps = Detail::get_numeric<double>(tree, "options.eps", context);
            options.weight_decay = Detail::get_numeric<double>(tree, "options.weight_decay", context);
            options.amsgrad = Detail::get_boolean(tree, "options.amsgrad", context);
            return Optimizer::Descriptor{Optimizer::Details::AdamWDescriptor{options}};
        } if (type == "sophia_g") {
            Optimizer::Details::SophiaGOptions options;
            options.lr(Detail::get_numeric<double>(tree, "options.learning_rate", context));
            options.beta1(Detail::get_numeric<double>(tree, "options.beta1", context));
            options.beta2(Detail::get_numeric<double>(tree, "options.beta2", context));
            options.rho(Detail::get_numeric<double>(tree, "options.rho", context));
            options.eps(Detail::get_numeric<double>(tree, "options.eps", context));
            options.weight_decay(Detail::get_numeric<double>(tree, "options.weight_decay", context));
            options.clip(Detail::get_numeric<double>(tree, "options.clip", context));
            options.hessian_update_interval(Detail::get_numeric<int64_t>(tree, "options.hessian_update_interval", context));
            return Optimizer::Descriptor{Optimizer::Details::SophiaGDescriptor{options}};
        } if (type == "sophia_h") {
            Optimizer::Details::SophiaHOptions options;
            options.lr(Detail::get_numeric<double>(tree, "options.learning_rate", context));
            options.beta1(Detail::get_numeric<double>(tree, "options.beta1", context));
            options.beta2(Detail::get_numeric<double>(tree, "options.beta2", context));
            options.rho(Detail::get_numeric<double>(tree, "options.rho", context));
            options.eps(Detail::get_numeric<double>(tree, "options.eps", context));
            options.weight_decay(Detail::get_numeric<double>(tree, "options.weight_decay", context));
            options.clip(Detail::get_numeric<double>(tree, "options.clip", context));
            options.hessian_update_interval(Detail::get_numeric<int64_t>(tree, "options.hessian_update_interval", context));
            return Optimizer::Descriptor{Optimizer::Details::SophiaHDescriptor{options}};
        } if (type == "muon") {
            Optimizer::Details::MuonOptions options;
            options.lr(Detail::get_numeric<double>(tree, "options.learning_rate", context));
            options.beta(Detail::get_numeric<double>(tree, "options.beta", context));
            options.weight_decay(Detail::get_numeric<double>(tree, "options.weight_decay", context));
            options.eps(Detail::get_numeric<double>(tree, "options.eps", context));
            options.max_update_norm(Detail::get_numeric<double>(tree, "options.max_update_norm", context));
            return Optimizer::Descriptor{Optimizer::Details::MuonDescriptor{options}};
        } if (type == "ada_muon") {
            Optimizer::Details::AdaMuonOptions options;
            options.lr(Detail::get_numeric<double>(tree, "options.learning_rate", context));
            options.beta(Detail::get_numeric<double>(tree, "options.beta", context));
            options.beta2(Detail::get_numeric<double>(tree, "options.beta2", context));
            options.weight_decay(Detail::get_numeric<double>(tree, "options.weight_decay", context));
            options.eps(Detail::get_numeric<double>(tree, "options.eps", context));
            options.max_update_norm(Detail::get_numeric<double>(tree, "options.max_update_norm", context));
            return Optimizer::Descriptor{Optimizer::Details::AdaMuonDescriptor{options}};
        } if (type == "muon_manifold") {
            Optimizer::Details::MuonManifoldOptions options;
            options.lr(Detail::get_numeric<double>(tree, "options.learning_rate", context));
            options.beta(Detail::get_numeric<double>(tree, "options.beta", context));
            options.beta2(Detail::get_numeric<double>(tree, "options.beta2", context));
            options.weight_decay(Detail::get_numeric<double>(tree, "options.weight_decay", context));
            options.eps(Detail::get_numeric<double>(tree, "options.eps", context));
            options.max_update_norm(Detail::get_numeric<double>(tree, "options.max_update_norm", context));
            options.retraction_epsilon(Detail::get_numeric<double>(tree, "options.retraction_epsilon", context));
            options.renormalize(Detail::get_boolean(tree, "options.renormalize", context));
            const auto manifold = Detail::get_string(tree, "options.manifold", context);
            options.manifold(Detail::manifold_kind_from_string(manifold, context));
            return Optimizer::Descriptor{Optimizer::Details::MuonManifoldDescriptor{options}};
        } if (type == "adafactor") {
            Optimizer::Details::AdafactorOptions options;
            options.lr(Detail::get_numeric<double>(tree, "options.learning_rate", context));
            options.eps1(Detail::get_numeric<double>(tree, "options.eps1", context));
            options.eps2(Detail::get_numeric<double>(tree, "options.eps2", context));
            options.clip_threshold(Detail::get_numeric<double>(tree, "options.clip_threshold", context));
            options.decay_rate(Detail::get_numeric<double>(tree, "options.decay_rate", context));
            options.beta1(Detail::get_numeric<double>(tree, "options.beta1", context));
            options.use_first_moment(Detail::get_numeric<double>(tree, "options.use_first_moment", context));
            options.weight_decay(Detail::get_boolean(tree, "options.weight_decay", context));
            options.scale_parameter(Detail::get_boolean(tree, "options.scale_parameter", context));
            options.relative_step(Detail::get_boolean(tree, "options.relative_step", context));
            options.warmup_init(Detail::get_boolean(tree, "options.warmup_init", context));
            return Optimizer::Descriptor{Optimizer::Details::AdafactorDescriptor{options}};
        } if (type == "adagrad") {
            Optimizer::Details::AdagradOptions options;
            options.learning_rate = Detail::get_numeric<double>(tree, "options.learning_rate", context);
            options.lr_decay = Detail::get_numeric<double>(tree, "options.lr_decay", context);
            options.weight_decay = Detail::get_numeric<double>(tree, "options.weight_decay", context);
            options.initial_accumulator_value = Detail::get_numeric<double>(tree, "options.initial_accumulator_value", context);
            options.eps = Detail::get_numeric<double>(tree, "options.eps", context);
            return Optimizer::Descriptor{Optimizer::Details::AdagradDescriptor{options}};
        } if (type == "lamb") {
            Optimizer::Details::LAMBOptions options;
            options.lr(Detail::get_numeric<double>(tree, "options.lr", context));
            options.beta1(Detail::get_numeric<double>(tree, "options.beta1", context));
            options.beta2(Detail::get_numeric<double>(tree, "options.beta2", context));
            options.eps(Detail::get_numeric<double>(tree, "options.eps", context));
            options.weight_decay(Detail::get_numeric<double>(tree, "options.weight_decay", context));
            options.adam(Detail::get_boolean(tree, "options.adam", context));
            return Optimizer::Descriptor{Optimizer::Details::LAMBDescriptor{options}};
        } if (type == "lion") {
            Optimizer::Details::LionOptions options;
            options.lr(Detail::get_numeric<double>(tree, "options.lr", context));
            options.beta1(Detail::get_numeric<double>(tree, "options.beta1", context));
            options.beta2(Detail::get_numeric<double>(tree, "options.beta2", context));
            options.weight_decay(Detail::get_numeric<double>(tree, "options.weight_decay", context));
            return Optimizer::Descriptor{Optimizer::Details::LionDescriptor{options}};
        } if (type == "rmsprop") {
            Optimizer::Details::RMSpropOptions options;
            options.learning_rate = Detail::get_numeric<double>(tree, "options.learning_rate", context);
            options.alpha = Detail::get_numeric<double>(tree, "options.alpha", context);
            options.eps = Detail::get_numeric<double>(tree, "options.eps", context);
            options.weight_decay = Detail::get_numeric<double>(tree, "options.weight_decay", context);
            options.momentum = Detail::get_numeric<double>(tree, "options.momentum", context);
            options.centered = Detail::get_boolean(tree, "options.centered", context);
            return Optimizer::Descriptor{Optimizer::Details::RMSpropDescriptor{options}};
        }
        std::ostringstream message;
        message << "Unknown optimizer descriptor '" << type << "' in " << context;
        throw std::runtime_error(message.str());
    }

    inline PropertyTree serialize_regularization(const Regularization::Descriptor& descriptor) {
        PropertyTree tree;
        std::visit(
            [&](const auto& concrete) {
                using DescriptorType = std::decay_t<decltype(concrete)>;
                const auto& options = concrete.options;
                if constexpr (std::is_same_v<DescriptorType, Regularization::L1Descriptor>) {
                    tree.put("type", "l1");
                    tree.put("options.coefficient", options.coefficient);
                } else if constexpr (std::is_same_v<DescriptorType, Regularization::ElasticNetDescriptor>) {
                    tree.put("type", "elastic_net");
                    tree.put("options.l1_coefficient", options.l1_coefficient);
                    tree.put("options.l2_coefficient", options.l2_coefficient);
                } else if constexpr (std::is_same_v<DescriptorType, Regularization::GroupLassoDescriptor>) {
                    tree.put("type", "group_lasso");
                    tree.put("options.coefficient", options.coefficient);
                    tree.put("options.group_dim", static_cast<std::int64_t>(options.group_dim));
                    tree.put("options.epsilon", options.epsilon);
                } else if constexpr (std::is_same_v<DescriptorType, Regularization::StructuredL2Descriptor>) {
                    tree.put("type", "structured_l2");
                    tree.put("options.coefficient", options.coefficient);
                    tree.put("options.group_dim", static_cast<std::int64_t>(options.group_dim));
                } else if constexpr (std::is_same_v<DescriptorType, Regularization::L0HardConcreteDescriptor>) {
                    tree.put("type", "l0_hard_concrete");
                    tree.put("options.coefficient", options.coefficient);
                    tree.put("options.beta", options.beta);
                    tree.put("options.gamma", options.gamma);
                    tree.put("options.zeta", options.zeta);
                } else if constexpr (std::is_same_v<DescriptorType, Regularization::OrthogonalityDescriptor>) {
                    tree.put("type", "orthogonality");
                    tree.put("options.coefficient", options.coefficient);
                } else if constexpr (std::is_same_v<DescriptorType, Regularization::SpectralNormDescriptor>) {
                    tree.put("type", "spectral_norm");
                    tree.put("options.coefficient", options.coefficient);
                    tree.put("options.target", options.target);
                } else if constexpr (std::is_same_v<DescriptorType, Regularization::MaxNormDescriptor>) {
                    tree.put("type", "max_norm");
                    tree.put("options.coefficient", options.coefficient);
                    tree.put("options.max_norm", options.max_norm);
                    tree.put("options.dim", static_cast<std::int64_t>(options.dim));
                } else if constexpr (std::is_same_v<DescriptorType, Regularization::KLSparsityDescriptor>) {
                    tree.put("type", "kl_sparsity");
                    tree.put("options.coefficient", options.coefficient);
                    tree.put("options.target", options.target);
                    tree.put("options.epsilon", options.epsilon);
                } else if constexpr (std::is_same_v<DescriptorType, Regularization::DeCovDescriptor>) {
                    tree.put("type", "decov");
                    tree.put("options.coefficient", options.coefficient);
                    tree.put("options.epsilon", options.epsilon);
                } else if constexpr (std::is_same_v<DescriptorType, Regularization::CenteringVarianceDescriptor>) {
                    tree.put("type", "centering_variance");
                    tree.put("options.coefficient", options.coefficient);
                    tree.put("options.target_std", options.target_std);
                } else if constexpr (std::is_same_v<DescriptorType, Regularization::JacobianNormDescriptor>) {
                    tree.put("type", "jacobian_norm");
                    tree.put("options.coefficient", options.coefficient);
                } else if constexpr (std::is_same_v<DescriptorType, Regularization::WGANGPDescriptor>) {
                    tree.put("type", "wgan_gp");
                    tree.put("options.coefficient", options.coefficient);
                    tree.put("options.target", options.target);
                } else if constexpr (std::is_same_v<DescriptorType, Regularization::R1Descriptor>) {
                    tree.put("type", "r1");
                    tree.put("options.coefficient", options.coefficient);
                } else if constexpr (std::is_same_v<DescriptorType, Regularization::R2Descriptor>) {
                    tree.put("type", "r2");
                    tree.put("options.coefficient", options.coefficient);
                } else if constexpr (std::is_same_v<DescriptorType, Regularization::TRADESDescriptor>) {
                    tree.put("type", "trades");
                    tree.put("options.coefficient", options.coefficient);
                } else if constexpr (std::is_same_v<DescriptorType, Regularization::VATDescriptor>) {
                    tree.put("type", "vat");
                    tree.put("options.coefficient", options.coefficient);
                } else if constexpr (std::is_same_v<DescriptorType, Regularization::L2Descriptor>) {
                    tree.put("type", "l2");
                    tree.put("options.coefficient", options.coefficient);
                } else if constexpr (std::is_same_v<DescriptorType, Regularization::EWCDescriptor>) {
                    tree.put("type", "ewc");
                    tree.put("options.strength", options.strength);
                } else if constexpr (std::is_same_v<DescriptorType, Regularization::MASDescriptor>) {
                    tree.put("type", "mas");
                    tree.put("options.strength", options.strength);
                } else if constexpr (std::is_same_v<DescriptorType, Regularization::SIDescriptor>) {
                    tree.put("type", "si");
                    tree.put("options.strength", options.strength);
                    tree.put("options.damping", options.damping);
                } else if constexpr (std::is_same_v<DescriptorType, Regularization::NuclearNormDescriptor>) {
                    tree.put("type", "nuclear_norm");
                    tree.put("options.strength", options.strength);
                } else if constexpr (std::is_same_v<DescriptorType, Regularization::SWADescriptor>) {
                    tree.put("type", "swa");
                    tree.put("options.coefficient", options.coefficient);
                } else if constexpr (std::is_same_v<DescriptorType, Regularization::SWAGDescriptor>) {
                    tree.put("type", "swag");
                    tree.put("options.coefficient", options.coefficient);
                    tree.put("options.variance_epsilon", options.variance_epsilon);
                    tree.put("options.start_step", static_cast<std::uint64_t>(options.start_step));
                    tree.put("options.accumulation_stride", static_cast<std::uint64_t>(options.accumulation_stride));
                    tree.put("options.max_snapshots", static_cast<std::uint64_t>(options.max_snapshots));
                } else if constexpr (std::is_same_v<DescriptorType, Regularization::FGEDescriptor>) {
                    tree.put("type", "fge");
                    tree.put("options.coefficient", options.coefficient);
                } else if constexpr (std::is_same_v<DescriptorType, Regularization::SFGEDescriptor>) {
                    tree.put("type", "sfge");
                    tree.put("options.coefficient", options.coefficient);
                } else {
                    static_assert(sizeof(DescriptorType) == 0, "Unsupported regularisation descriptor supplied.");
                }
            }, descriptor);
        return tree;
    }

    inline Regularization::Descriptor deserialize_regularization(const PropertyTree& tree, const std::string& context)
    {
        const auto type = Detail::to_lower(Detail::get_string(tree, "type", context));
        if (type == "l1") {
            Regularization::Details::L1Options options;
            options.coefficient = Detail::get_numeric<double>(tree, "options.coefficient", context);
            return Regularization::Descriptor{Regularization::Details::L1Descriptor{options}};
        }
        if (type == "elastic_net") {
            Regularization::Details::ElasticNetOptions options;
            options.l1_coefficient = Detail::get_numeric<double>(tree, "options.l1_coefficient", context);
            options.l2_coefficient = Detail::get_numeric<double>(tree, "options.l2_coefficient", context);
            return Regularization::Descriptor{Regularization::Details::ElasticNetDescriptor{options}};
        }
        if (type == "group_lasso") {
            Regularization::Details::GroupLassoOptions options;
            options.coefficient = Detail::get_numeric<double>(tree, "options.coefficient", context);
            options.group_dim = Detail::get_numeric<std::int64_t>(tree, "options.group_dim", context);
            options.epsilon = Detail::get_numeric<double>(tree, "options.epsilon", context);
            return Regularization::Descriptor{Regularization::Details::GroupLassoDescriptor{options}};
        }
        if (type == "structured_l2") {
            Regularization::Details::StructuredL2Options options;
            options.coefficient = Detail::get_numeric<double>(tree, "options.coefficient", context);
            options.group_dim = Detail::get_numeric<std::int64_t>(tree, "options.group_dim", context);
            return Regularization::Descriptor{Regularization::Details::StructuredL2Descriptor{options}};
        }
        if (type == "l0_hard_concrete") {
            Regularization::Details::L0HardConcreteOptions options;
            options.coefficient = Detail::get_numeric<double>(tree, "options.coefficient", context);
            options.beta = Detail::get_numeric<double>(tree, "options.beta", context);
            options.gamma = Detail::get_numeric<double>(tree, "options.gamma", context);
            options.zeta = Detail::get_numeric<double>(tree, "options.zeta", context);
            return Regularization::Descriptor{Regularization::Details::L0HardConcreteDescriptor{options}};
        }
        if (type == "orthogonality") {
            Regularization::Details::OrthogonalityOptions options;
            options.coefficient = Detail::get_numeric<double>(tree, "options.coefficient", context);
            return Regularization::Descriptor{Regularization::Details::OrthogonalityDescriptor{options}};
        }
        if (type == "spectral_norm") {
            Regularization::Details::SpectralNormOptions options;
            options.coefficient = Detail::get_numeric<double>(tree, "options.coefficient", context);
            options.target = Detail::get_numeric<double>(tree, "options.target", context);
            return Regularization::Descriptor{Regularization::Details::SpectralNormDescriptor{options}};
        }
        if (type == "max_norm") {
            Regularization::Details::MaxNormOptions options;
            options.coefficient = Detail::get_numeric<double>(tree, "options.coefficient", context);
            options.max_norm = Detail::get_numeric<double>(tree, "options.max_norm", context);
            options.dim = Detail::get_numeric<std::int64_t>(tree, "options.dim", context);
            return Regularization::Descriptor{Regularization::Details::MaxNormDescriptor{options}};
        }
        if (type == "kl_sparsity") {
            Regularization::Details::KLSparsityOptions options;
            options.coefficient = Detail::get_numeric<double>(tree, "options.coefficient", context);
            options.target = Detail::get_numeric<double>(tree, "options.target", context);
            options.epsilon = Detail::get_numeric<double>(tree, "options.epsilon", context);
            return Regularization::Descriptor{Regularization::Details::KLSparsityDescriptor{options}};
        }
        if (type == "decov") {
            Regularization::Details::DeCovOptions options;
            options.coefficient = Detail::get_numeric<double>(tree, "options.coefficient", context);
            options.epsilon = Detail::get_numeric<double>(tree, "options.epsilon", context);
            return Regularization::Descriptor{Regularization::Details::DeCovDescriptor{options}};
        }
        if (type == "centering_variance") {
            Regularization::Details::CenteringVarianceOptions options;
            options.coefficient = Detail::get_numeric<double>(tree, "options.coefficient", context);
            options.target_std = Detail::get_numeric<double>(tree, "options.target_std", context);
            return Regularization::Descriptor{Regularization::Details::CenteringVarianceDescriptor{options}};
        }
        if (type == "jacobian_norm") {
            Regularization::Details::JacobianNormOptions options;
            options.coefficient = Detail::get_numeric<double>(tree, "options.coefficient", context);
            return Regularization::Descriptor{Regularization::Details::JacobianNormDescriptor{options}};
        }
        if (type == "wgan_gp") {
            Regularization::Details::WGANGPOptions options;
            options.coefficient = Detail::get_numeric<double>(tree, "options.coefficient", context);
            options.target = Detail::get_numeric<double>(tree, "options.target", context);
            return Regularization::Descriptor{Regularization::Details::WGANGPDescriptor{options}};
        }
        if (type == "r1") {
            Regularization::Details::R1Options options;
            options.coefficient = Detail::get_numeric<double>(tree, "options.coefficient", context);
            return Regularization::Descriptor{Regularization::Details::R1Descriptor{options}};
        }
        if (type == "r2") {
            Regularization::Details::R2Options options;
            options.coefficient = Detail::get_numeric<double>(tree, "options.coefficient", context);
            return Regularization::Descriptor{Regularization::Details::R2Descriptor{options}};
        }
        if (type == "trades") {
            Regularization::Details::TRADESOptions options;
            options.coefficient = Detail::get_numeric<double>(tree, "options.coefficient", context);
            return Regularization::Descriptor{Regularization::Details::TRADESDescriptor{options}};
        }
        if (type == "vat") {
            Regularization::Details::VATOptions options;
            options.coefficient = Detail::get_numeric<double>(tree, "options.coefficient", context);
            return Regularization::Descriptor{Regularization::Details::VATDescriptor{options}};
        }
        if (type == "l2") {
            Regularization::Details::L2Options options;
            options.coefficient = Detail::get_numeric<double>(tree, "options.coefficient", context);
            return Regularization::Descriptor{Regularization::Details::L2Descriptor{options}};
        }
        if (type == "ewc") {
            Regularization::Details::EWCOptions options;
            options.strength = Detail::get_numeric<double>(tree, "options.strength", context);
            return Regularization::Descriptor{Regularization::Details::EWCDescriptor{options}};
        }
        if (type == "mas") {
            Regularization::Details::MASOptions options;
            options.strength = Detail::get_numeric<double>(tree, "options.strength", context);
            return Regularization::Descriptor{Regularization::Details::MASDescriptor{options}};
        }
        if (type == "si") {
            Regularization::Details::SIOptions options;
            options.strength = Detail::get_numeric<double>(tree, "options.strength", context);
            options.damping = Detail::get_numeric<double>(tree, "options.damping", context);
            return Regularization::Descriptor{Regularization::Details::SIDescriptor{options}};
        }
        if (type == "nuclear_norm") {
            Regularization::Details::NuclearNormOptions options;
            options.strength = Detail::get_numeric<double>(tree, "options.strength", context);
            return Regularization::Descriptor{Regularization::Details::NuclearNormDescriptor{options}};
        }
        if (type == "swa") {
            Regularization::Details::SWAOptions options;
            options.coefficient = Detail::get_numeric<double>(tree, "options.coefficient", context);
            return Regularization::Descriptor{Regularization::Details::SWADescriptor{options}};
        }
        if (type == "swag") {
            Regularization::Details::SWAGOptions options;
            options.coefficient = Detail::get_numeric<double>(tree, "options.coefficient", context);
            options.variance_epsilon = Detail::get_numeric<double>(tree, "options.variance_epsilon", context);
            options.start_step = static_cast<std::size_t>(Detail::get_numeric<std::uint64_t>(tree, "options.start_step", context));
            options.accumulation_stride = static_cast<std::size_t>(
                Detail::get_numeric<std::uint64_t>(tree, "options.accumulation_stride", context));
            options.max_snapshots = static_cast<std::size_t>(
                Detail::get_numeric<std::uint64_t>(tree, "options.max_snapshots", context));
            return Regularization::Descriptor{Regularization::Details::SWAGDescriptor{options}};
        }
        if (type == "fge") {
            Regularization::Details::FGEOptions options;
            options.coefficient = Detail::get_numeric<double>(tree, "options.coefficient", context);
            return Regularization::Descriptor{Regularization::Details::FGEDescriptor{options}};
        }
        if (type == "sfge") {
            Regularization::Details::SFGEOptions options;
            options.coefficient = Detail::get_numeric<double>(tree, "options.coefficient", context);
            return Regularization::Descriptor{Regularization::Details::SFGEDescriptor{options}};
        }
        std::ostringstream message;
        message << "Unknown regularisation descriptor '" << type << "' in " << context;
        throw std::runtime_error(message.str());
    }
    inline PropertyTree serialize_loss(const Loss::Descriptor& descriptor)
    {
        PropertyTree tree;
        std::visit(
            [&](const auto& concrete) {
                using DescriptorType = std::decay_t<decltype(concrete)>;
                const auto& options = concrete.options;
                if constexpr (std::is_same_v<DescriptorType, Loss::Details::MSEDescriptor>) {
                    tree.put("type", "mse");
                    tree.put("options.reduction", Detail::loss_reduction_to_string(options.reduction));
                    tree.put_child("options.weight", Detail::write_array(options.weight));
                } else if constexpr (std::is_same_v<DescriptorType, Loss::Details::CrossEntropyDescriptor>) {
                    tree.put("type", "cross_entropy");
                    tree.put("options.reduction", Detail::loss_reduction_to_string(options.reduction));
                    tree.put_child("options.weight", Detail::write_array(options.weight));
                    tree.put("options.label_smoothing", options.label_smoothing);
                } else if constexpr (std::is_same_v<DescriptorType, Loss::Details::BCEWithLogitsDescriptor>) {
                    tree.put("type", "bce_with_logits");
                    tree.put("options.reduction", Detail::loss_reduction_to_string(options.reduction));
                    tree.put_child("options.weight", Detail::write_array(options.weight));
                    tree.put_child("options.pos_weight", Detail::write_array(options.pos_weight));
                } else if constexpr (std::is_same_v<DescriptorType, Loss::Details::MAEDescriptor>) {
                    tree.put("type", "mae");
                    tree.put("options.reduction", Detail::loss_reduction_to_string(options.reduction));
                    tree.put_child("options.weight", Detail::write_array(options.weight));
                } else if constexpr (std::is_same_v<DescriptorType, Loss::Details::NegativeLogLikelihoodDescriptor>) {
                    tree.put("type", "nll");
                    tree.put("options.reduction", Detail::loss_reduction_to_string(options.reduction));
                    tree.put_child("options.weight", Detail::write_array(options.weight));
                    if (options.ignore_index.has_value())
                        tree.put("options.ignore_index", options.ignore_index.value());
                } else if constexpr (std::is_same_v<DescriptorType, Loss::Details::SmoothL1Descriptor>) {
                    tree.put("type", "smooth_l1");
                    tree.put("options.reduction", Detail::loss_reduction_to_string(options.reduction));
                    tree.put_child("options.weight", Detail::write_array(options.weight));
                    tree.put("options.beta", options.beta);
                } else if constexpr (std::is_same_v<DescriptorType, Loss::Details::KLDivDescriptor>) {
                    tree.put("type", "kl");
                    tree.put("options.reduction", Detail::loss_reduction_to_string(options.reduction));
                    tree.put("options.log_target", options.log_target);
                    tree.put("options.use_batch_mean", options.use_batch_mean);
                    tree.put("options.log_softmax_dim", options.log_softmax_dim);
                    tree.put("options.prediction_is_log", options.prediction_is_log);
                } else if constexpr (std::is_same_v<DescriptorType, Loss::Details::MarginRankingDescriptor>) {
                    tree.put("type", "margin_ranking");
                    tree.put("options.reduction", Detail::loss_reduction_to_string(options.reduction));
                    tree.put("options.margin", options.margin);
                } else if constexpr (std::is_same_v<DescriptorType, Loss::Details::CosineEmbeddingDescriptor>) {
                    tree.put("type", "cosine_embedding");
                    tree.put("options.reduction", Detail::loss_reduction_to_string(options.reduction));
                    tree.put("options.margin", options.margin);
                } else if constexpr (std::is_same_v<DescriptorType, Loss::Details::DiceDescriptor>) {
                    tree.put("type", "dice");
                    tree.put("options.reduction", Detail::loss_reduction_to_string(options.reduction));
                    tree.put("options.smooth", options.smooth);
                    tree.put("options.exponent", options.exponent);
                    tree.put("options.clamp_predictions", options.clamp_predictions);
                } else if constexpr (std::is_same_v<DescriptorType, Loss::Details::LovaszSoftmaxDescriptor>) {
                    tree.put("type", "lovasz_softmax");
                    tree.put("options.reduction", Detail::loss_reduction_to_string(options.reduction));
                    tree.put("options.per_image", options.per_image);
                    tree.put("options.ignore_index", options.ignore_index);
                    tree.put("options.apply_softmax", options.apply_softmax);
                    tree.put("options.include_background", options.include_background);
                    tree.put("options.only_present_classes", options.only_present_classes);
                } else if constexpr (std::is_same_v<DescriptorType, Loss::Details::TverskyDescriptor>) {
                    tree.put("type", "tversky");
                    tree.put("options.reduction", Detail::loss_reduction_to_string(options.reduction));
                    tree.put("options.alpha", options.alpha);
                    tree.put("options.beta", options.beta);
                    tree.put("options.smooth", options.smooth);
                } else {
                    static_assert(sizeof(DescriptorType) == 0, "Unsupported loss descriptor supplied.");
                }
            },
            descriptor);
        return tree;
    }

    inline Loss::Descriptor deserialize_loss(const PropertyTree& tree, const std::string& context)
    {
        const auto type = Detail::to_lower(Detail::get_string(tree, "type", context));
        if (type == "mse") {
            Loss::Details::MSEOptions options;
            options.reduction = Detail::loss_reduction_from_string(Detail::get_string(tree, "options.reduction", context), context);
            if (const auto weight_tree = tree.get_child_optional("options.weight"))
                options.weight = Detail::read_array<double>(*weight_tree, context + ".options.weight");
            return Loss::Descriptor{Loss::Details::MSEDescriptor{options}};
        }
        if (type == "cross_entropy") {
            Loss::Details::CrossEntropyOptions options;
            options.reduction = Detail::loss_reduction_from_string(Detail::get_string(tree, "options.reduction", context), context);
            if (const auto weight_tree = tree.get_child_optional("options.weight"))
                options.weight = Detail::read_array<double>(*weight_tree, context + ".options.weight");
            options.label_smoothing = Detail::get_numeric<double>(tree, "options.label_smoothing", context);
            return Loss::Descriptor{Loss::Details::CrossEntropyDescriptor{options}};
        }
        if (type == "bce_with_logits") {
            Loss::Details::BCEWithLogitsOptions options;
            options.reduction = Detail::loss_reduction_from_string(Detail::get_string(tree, "options.reduction", context), context);
            if (const auto weight_tree = tree.get_child_optional("options.weight"))
                options.weight = Detail::read_array<double>(*weight_tree, context + ".options.weight");
            if (const auto pos_weight_tree = tree.get_child_optional("options.pos_weight"))
                options.pos_weight = Detail::read_array<double>(*pos_weight_tree, context + ".options.pos_weight");
            return Loss::Descriptor{Loss::Details::BCEWithLogitsDescriptor{options}};
        }
        if (type == "mae") {
            Loss::Details::MAEOptions options;
            options.reduction = Detail::loss_reduction_from_string(Detail::get_string(tree, "options.reduction", context), context);
            if (const auto weight_tree = tree.get_child_optional("options.weight"))
                options.weight = Detail::read_array<double>(*weight_tree, context + ".options.weight");
            return Loss::Descriptor{Loss::Details::MAEDescriptor{options}};
        }
        if (type == "nll") {
            Loss::Details::NegativeLogLikelihoodOptions options;
            options.reduction = Detail::loss_reduction_from_string(Detail::get_string(tree, "options.reduction", context), context);
            if (const auto weight_tree = tree.get_child_optional("options.weight"))
                options.weight = Detail::read_array<double>(*weight_tree, context + ".options.weight");
            if (const auto ignore_index = tree.get_optional<std::int64_t>("options.ignore_index"))
                options.ignore_index = *ignore_index;

            return Loss::Descriptor{Loss::Details::NegativeLogLikelihoodDescriptor{options}};
        }
        if (type == "smooth_l1") {
            Loss::Details::SmoothL1Options options;
            options.reduction = Detail::loss_reduction_from_string(Detail::get_string(tree, "options.reduction", context), context);
            if (const auto weight_tree = tree.get_child_optional("options.weight"))
                options.weight = Detail::read_array<double>(*weight_tree, context + ".options.weight");
            options.beta = Detail::get_numeric<double>(tree, "options.beta", context);
            return Loss::Descriptor{Loss::Details::SmoothL1Descriptor{options}};
        }
        if (type == "kl") {
            Loss::Details::KLDivOptions options;
            options.reduction = Detail::loss_reduction_from_string(
                Detail::get_string(tree, "options.reduction", context), context);
            options.log_target = Detail::get_boolean(tree, "options.log_target", context);
            options.use_batch_mean = Detail::get_boolean(tree, "options.use_batch_mean", context);
            options.log_softmax_dim = Detail::get_numeric<std::int64_t>(tree, "options.log_softmax_dim", context);
            options.prediction_is_log = Detail::get_boolean(tree, "options.prediction_is_log", context);
            return Loss::Descriptor{Loss::Details::KLDivDescriptor{options}};
        }
        if (type == "margin_ranking") {
            Loss::Details::MarginRankingOptions options;
            options.reduction = Detail::loss_reduction_from_string(
                Detail::get_string(tree, "options.reduction", context), context);
            options.margin = Detail::get_numeric<double>(tree, "options.margin", context);
            return Loss::Descriptor{Loss::Details::MarginRankingDescriptor{options}};
        }
        if (type == "cosine_embedding") {
            Loss::Details::CosineEmbeddingOptions options;
            options.reduction = Detail::loss_reduction_from_string(
                Detail::get_string(tree, "options.reduction", context), context);
            options.margin = Detail::get_numeric<double>(tree, "options.margin", context);
            return Loss::Descriptor{Loss::Details::CosineEmbeddingDescriptor{options}};
        }
        if (type == "dice") {
            Loss::Details::DiceOptions options;
            options.reduction = Detail::loss_reduction_from_string(
                Detail::get_string(tree, "options.reduction", context), context);
            options.smooth = Detail::get_numeric<double>(tree, "options.smooth", context);
            options.exponent = Detail::get_numeric<double>(tree, "options.exponent", context);
            options.clamp_predictions = Detail::get_boolean(tree, "options.clamp_predictions", context);
            return Loss::Descriptor{Loss::Details::DiceDescriptor{options}};
        }
        if (type == "lovasz_softmax") {
            Loss::Details::LovaszSoftmaxOptions options;
            options.reduction = Detail::loss_reduction_from_string(
                Detail::get_string(tree, "options.reduction", context), context);
            options.per_image = Detail::get_boolean(tree, "options.per_image", context);
            options.ignore_index = Detail::get_numeric<std::int64_t>(tree, "options.ignore_index", context);
            options.apply_softmax = Detail::get_boolean(tree, "options.apply_softmax", context);
            options.include_background = Detail::get_boolean(tree, "options.include_background", context);
            options.only_present_classes = Detail::get_boolean(tree, "options.only_present_classes", context);
            return Loss::Descriptor{Loss::Details::LovaszSoftmaxDescriptor{options}};
        }
        if (type == "tversky") {
            Loss::Details::TverskyOptions options;
            options.reduction = Detail::loss_reduction_from_string(
                Detail::get_string(tree, "options.reduction", context), context);
            options.alpha = Detail::get_numeric<double>(tree, "options.alpha", context);
            options.beta = Detail::get_numeric<double>(tree, "options.beta", context);
            options.smooth = Detail::get_numeric<double>(tree, "options.smooth", context);
            return Loss::Descriptor{Loss::Details::TverskyDescriptor{options}};
        }
        std::ostringstream message;
        message << "Unknown loss descriptor '" << type << "' in " << context;
        throw std::runtime_error(message.str());
    }


    inline PropertyTree serialize_local_config(const LocalConfig& config)
    {
        PropertyTree tree;
        if (config.optimizer) {
            tree.add_child("optimizer", serialize_optimizer(*config.optimizer));
        }
        if (config.loss) {
            tree.add_child("loss", serialize_loss(*config.loss));
        }
        PropertyTree regularization;
        for (const auto& descriptor : config.regularization) {
            regularization.push_back({"", serialize_regularization(descriptor)});
        }
        tree.add_child("regularization", regularization);
        return tree;
    }

    inline LocalConfig deserialize_local_config(const PropertyTree& tree, const std::string& context)
    {
        LocalConfig config;
        if (const auto optimizer = tree.get_child_optional("optimizer")) {
            config.optimizer = deserialize_optimizer(*optimizer, context + " optimizer");
        }
        if (const auto loss = tree.get_child_optional("loss")) {
            config.loss = deserialize_loss(*loss, context + " loss");
        }
        if (const auto regularization = tree.get_child_optional("regularization")) {
            for (const auto& node : *regularization) {
                config.regularization.push_back(deserialize_regularization(node.second, context + " regularization"));
            }
        }
        return config;
    }

    inline PropertyTree serialize_attention(const Attention::Descriptor& descriptor)
    {
        PropertyTree tree;
        std::visit(
            [&](const auto& concrete) {
                using DescriptorType = std::decay_t<decltype(concrete)>;
                if constexpr (std::is_same_v<DescriptorType, Attention::MultiHeadDescriptor>) {
                    const auto& options = concrete.options;
                    tree.put("type", "multi_head");
                    tree.put("options.embed_dim", options.embed_dim);
                    tree.put("options.num_heads", options.num_heads);
                    tree.put("options.dropout", options.dropout);
                    tree.put("options.bias", options.bias);
                    tree.put("options.add_bias_kv", options.add_bias_kv);
                    tree.put("options.add_zero_attn", options.add_zero_attn);
                    tree.put("options.batch_first", options.batch_first);
                    tree.put("options.variant", Detail::attention_variant_to_string(options.variant));
                } else if constexpr (std::is_same_v<DescriptorType, Attention::MultiHeadLatentDescriptor>) {
                    const auto& options = concrete.options;
                    tree.put("type", "multi_head_latent");
                    tree.put("options.embed_dim", options.embed_dim);
                    tree.put("options.num_heads", options.num_heads);
                    tree.put("options.latent_dim", options.latent_dim);
                    tree.put("options.dropout", options.dropout);
                    tree.put("options.bias", options.bias);
                    tree.put("options.batch_first", options.batch_first);
                    tree.put("options.variant", Detail::attention_variant_to_string(options.variant));
                } else {
                    static_assert(sizeof(DescriptorType) == 0, "Unsupported attention descriptor supplied.");
                }
            },
            descriptor);
        return tree;
    }

    inline Attention::Descriptor deserialize_attention(const PropertyTree& tree, const std::string& context) {
        const auto type = Detail::to_lower(Detail::get_string(tree, "type", context));
        if (type == "multi_head") {
            Attention::MultiHeadOptions options;
            options.embed_dim = Detail::get_numeric<std::int64_t>(tree, "options.embed_dim", context);
            options.num_heads = Detail::get_numeric<std::int64_t>(tree, "options.num_heads", context);
            options.dropout = Detail::get_numeric<double>(tree, "options.dropout", context);
            options.bias = Detail::get_boolean(tree, "options.bias", context);
            options.add_bias_kv = Detail::get_boolean(tree, "options.add_bias_kv", context);
            options.add_zero_attn = Detail::get_boolean(tree, "options.add_zero_attn", context);
            options.batch_first = Detail::get_boolean(tree, "options.batch_first", context);
            options.variant = Detail::attention_variant_from_string(Detail::get_string(tree, "options.variant", context));
            return Attention::Descriptor{Attention::MultiHeadDescriptor{options}};
        }
        if (type == "multi_head_latent") {
            Attention::MultiHeadLatentOptions options;
            options.embed_dim = Detail::get_numeric<std::int64_t>(tree, "options.embed_dim", context);
            options.num_heads = Detail::get_numeric<std::int64_t>(tree, "options.num_heads", context);
            options.latent_dim = Detail::get_numeric<std::int64_t>(tree, "options.latent_dim", context);
            options.dropout = Detail::get_numeric<double>(tree, "options.dropout", context);
            options.bias = Detail::get_boolean(tree, "options.bias", context);
            options.batch_first = Detail::get_boolean(tree, "options.batch_first", context);
            options.variant = Detail::attention_variant_from_string(Detail::get_string(tree, "options.variant", context));
            return Attention::Descriptor{Attention::MultiHeadLatentDescriptor{options}};
        }
        std::ostringstream message;
        message << "Unsupported attention descriptor '" << type << "' in " << context;
        throw std::runtime_error(message.str());
    }

    inline PropertyTree serialize_layer_descriptor(const Layer::Descriptor& descriptor)
    {
        PropertyTree tree;
        std::visit(
            [&](const auto& concrete) {
                using DescriptorType = std::decay_t<decltype(concrete)>;
                if constexpr (std::is_same_v<DescriptorType, Layer::FCDescriptor>) {
                    tree.put("type", "fc");
                    tree.put("options.in_features", concrete.options.in_features);
                    tree.put("options.out_features", concrete.options.out_features);
                    tree.put("options.bias", concrete.options.bias);
                    tree.add_child("activation", Detail::serialize_activation_descriptor(concrete.activation));
                    tree.add_child("initialization", Detail::serialize_initialization_descriptor(concrete.initialization));
                    tree.add_child("local", serialize_local_config(concrete.local));
                } else if constexpr (std::is_same_v<DescriptorType, Layer::Conv1dDescriptor>) {
                    tree.put("type", "conv1d");
                    tree.put("options.in_channels", concrete.options.in_channels);
                    tree.put("options.out_channels", concrete.options.out_channels);
                    tree.add_child("options.kernel_size", Detail::write_array(concrete.options.kernel_size));
                    tree.add_child("options.stride", Detail::write_array(concrete.options.stride));
                    tree.add_child("options.padding", Detail::write_array(concrete.options.padding));
                    tree.add_child("options.dilation", Detail::write_array(concrete.options.dilation));
                    tree.put("options.groups", concrete.options.groups);
                    tree.put("options.bias", concrete.options.bias);
                    tree.put("options.padding_mode", concrete.options.padding_mode);
                    tree.add_child("activation", Detail::serialize_activation_descriptor(concrete.activation));
                    tree.add_child("initialization", Detail::serialize_initialization_descriptor(concrete.initialization));
                    tree.add_child("local", serialize_local_config(concrete.local));
                } else if constexpr (std::is_same_v<DescriptorType, Layer::Conv2dDescriptor>) {
                    tree.put("type", "conv2d");
                    tree.put("options.in_channels", concrete.options.in_channels);
                    tree.put("options.out_channels", concrete.options.out_channels);
                    tree.add_child("options.kernel_size", Detail::write_array(concrete.options.kernel_size));
                    tree.add_child("options.stride", Detail::write_array(concrete.options.stride));
                    tree.add_child("options.padding", Detail::write_array(concrete.options.padding));
                    tree.add_child("options.dilation", Detail::write_array(concrete.options.dilation));
                    tree.put("options.groups", concrete.options.groups);
                    tree.put("options.bias", concrete.options.bias);
                    tree.put("options.padding_mode", concrete.options.padding_mode);
                    tree.add_child("activation", Detail::serialize_activation_descriptor(concrete.activation));
                    tree.add_child("initialization", Detail::serialize_initialization_descriptor(concrete.initialization));
                    tree.add_child("local", serialize_local_config(concrete.local));
                } else if constexpr (std::is_same_v<DescriptorType, Layer::BatchNorm2dDescriptor>) {
                    tree.put("type", "batch_norm2d");
                    tree.put("options.num_features", concrete.options.num_features);
                    tree.put("options.eps", concrete.options.eps);
                    tree.put("options.momentum", concrete.options.momentum);
                    tree.put("options.affine", concrete.options.affine);
                    tree.put("options.track_running_stats", concrete.options.track_running_stats);
                    tree.add_child("activation", Detail::serialize_activation_descriptor(concrete.activation));
                    tree.add_child("initialization", Detail::serialize_initialization_descriptor(concrete.initialization));
                    tree.add_child("local", serialize_local_config(concrete.local));
                } else if constexpr (std::is_same_v<DescriptorType, Layer::PoolingDescriptor>) {
                    tree.put("type", "pooling");
                    tree.put("options.variant", Detail::pooling_variant_to_string(concrete.options));
                    std::visit(
                        [&](const auto& options) {
                            using OptionType = std::decay_t<decltype(options)>;
                            if constexpr (std::is_same_v<OptionType, Layer::Details::MaxPool1dOptions>) {
                                tree.add_child("options.kernel_size", Detail::write_array(options.kernel_size));
                                tree.add_child("options.stride", Detail::write_array(options.stride));
                                tree.add_child("options.padding", Detail::write_array(options.padding));
                                tree.add_child("options.dilation", Detail::write_array(options.dilation));
                                tree.put("options.ceil_mode", options.ceil_mode);
                            } else if constexpr (std::is_same_v<OptionType, Layer::Details::AvgPool1dOptions>) {
                                tree.add_child("options.kernel_size", Detail::write_array(options.kernel_size));
                                tree.add_child("options.stride", Detail::write_array(options.stride));
                                tree.add_child("options.padding", Detail::write_array(options.padding));
                                tree.put("options.ceil_mode", options.ceil_mode);
                                tree.put("options.count_include_pad", options.count_include_pad);
                            } else if constexpr (std::is_same_v<OptionType, Layer::Details::AdaptiveAvgPool1dOptions>) {
                                tree.add_child("options.output_size", Detail::write_array(options.output_size));
                            } else if constexpr (std::is_same_v<OptionType, Layer::Details::AdaptiveMaxPool1dOptions>) {
                                tree.add_child("options.output_size", Detail::write_array(options.output_size));
                            } else if constexpr (std::is_same_v<OptionType, Layer::Details::MaxPool2dOptions>) {
                                tree.add_child("options.kernel_size", Detail::write_array(options.kernel_size));
                                tree.add_child("options.stride", Detail::write_array(options.stride));
                                tree.add_child("options.padding", Detail::write_array(options.padding));
                                tree.add_child("options.dilation", Detail::write_array(options.dilation));
                                tree.put("options.ceil_mode", options.ceil_mode);
                            } else if constexpr (std::is_same_v<OptionType, Layer::Details::AvgPool2dOptions>) {
                                tree.add_child("options.kernel_size", Detail::write_array(options.kernel_size));
                                tree.add_child("options.stride", Detail::write_array(options.stride));
                                tree.add_child("options.padding", Detail::write_array(options.padding));
                                tree.put("options.ceil_mode", options.ceil_mode);
                                tree.put("options.count_include_pad", options.count_include_pad);
                            } else if constexpr (std::is_same_v<OptionType, Layer::Details::AdaptiveAvgPool2dOptions>) {
                                tree.add_child("options.output_size", Detail::write_array(options.output_size));
                            } else if constexpr (std::is_same_v<OptionType, Layer::Details::AdaptiveMaxPool2dOptions>) {
                                tree.add_child("options.output_size", Detail::write_array(options.output_size));
                            }
                        },
                        concrete.options);
                    tree.add_child("activation", Detail::serialize_activation_descriptor(concrete.activation));
                    tree.add_child("local", serialize_local_config(concrete.local));
                } else if constexpr (std::is_same_v<DescriptorType, Layer::HardDropoutDescriptor>) {
                    tree.put("type", "harddropout");
                    tree.put("options.probability", concrete.options.probability);
                    tree.put("options.inplace", concrete.options.inplace);
                    tree.add_child("activation", Detail::serialize_activation_descriptor(concrete.activation));
                    tree.add_child("local", serialize_local_config(concrete.local));
                } else if constexpr (std::is_same_v<DescriptorType, Layer::SoftDropoutDescriptor>) {
                    tree.put("type", "softdropout");
                    tree.put("options.probability", concrete.options.probability);
                    tree.put("options.inplace", concrete.options.inplace);
                    tree.put("options.noise_mean", concrete.options.noise_mean);
                    tree.put("options.noise_std", concrete.options.noise_std);
                    tree.put("options.noise_type", Detail::soft_dropout_noise_type_to_string(concrete.options.noise_type));
                    tree.add_child("activation", Detail::serialize_activation_descriptor(concrete.activation));
                    tree.add_child("local", serialize_local_config(concrete.local));
                } else if constexpr (std::is_same_v<DescriptorType, Layer::FlattenDescriptor>) {
                    tree.put("type", "flatten");
                    tree.put("options.start_dim", concrete.options.start_dim);
                    tree.put("options.end_dim", concrete.options.end_dim);
                    tree.add_child("activation", Detail::serialize_activation_descriptor(concrete.activation));
                    tree.add_child("local", serialize_local_config(concrete.local));
                } else if constexpr (std::is_same_v<DescriptorType, Layer::UpsampleDescriptor>) {
                    tree.put("type", "upsample");
                    tree.add_child("options.scale", Detail::write_array(concrete.options.scale));
                    tree.put("options.mode", Detail::upsample_mode_to_string(concrete.options.mode));
                    tree.put("options.align_corners", concrete.options.align_corners);
                    tree.put("options.recompute_scale_factor", concrete.options.recompute_scale_factor);
                    tree.add_child("activation", Detail::serialize_activation_descriptor(concrete.activation));
                    tree.add_child("local", serialize_local_config(concrete.local));
                } else if constexpr (std::is_same_v<DescriptorType, Layer::DownsampleDescriptor>) {
                    tree.put("type", "downsample");
                    tree.add_child("options.scale", Detail::write_array(concrete.options.scale));
                    tree.put("options.mode", Detail::downsample_mode_to_string(concrete.options.mode));
                    tree.put("options.align_corners", concrete.options.align_corners);
                    tree.put("options.recompute_scale_factor", concrete.options.recompute_scale_factor);
                    tree.add_child("activation", Detail::serialize_activation_descriptor(concrete.activation));
                    tree.add_child("local", serialize_local_config(concrete.local));
                } else if constexpr (std::is_same_v<DescriptorType, Layer::ReduceDescriptor>) {
                    tree.put("type", "reduce");
                    tree.put("options.op", Detail::reduce_op_to_string(concrete.options.op));
                    tree.add_child("options.dims", Detail::write_array(concrete.options.dims));
                    tree.put("options.keep_dim", concrete.options.keep_dim);
                    tree.add_child("activation", Detail::serialize_activation_descriptor(concrete.activation));
                    tree.add_child("local", serialize_local_config(concrete.local));
                } else if constexpr (std::is_same_v<DescriptorType, Layer::RNNDescriptor>) {
                    tree.put("type", "rnn");
                    tree.put("options.input_size", concrete.options.input_size);
                    tree.put("options.hidden_size", concrete.options.hidden_size);
                    tree.put("options.num_layers", concrete.options.num_layers);
                    tree.put("options.dropout", concrete.options.dropout);
                    tree.put("options.batch_first", concrete.options.batch_first);
                    tree.put("options.bidirectional", concrete.options.bidirectional);
                    tree.put("options.nonlinearity", concrete.options.nonlinearity);
                    tree.add_child("activation", Detail::serialize_activation_descriptor(concrete.activation));
                    tree.add_child("initialization", Detail::serialize_initialization_descriptor(concrete.initialization));
                    tree.add_child("local", serialize_local_config(concrete.local));
                } else if constexpr (std::is_same_v<DescriptorType, Layer::LSTMDescriptor>) {
                    tree.put("type", "lstm");
                    tree.put("options.input_size", concrete.options.input_size);
                    tree.put("options.hidden_size", concrete.options.hidden_size);
                    tree.put("options.num_layers", concrete.options.num_layers);
                    tree.put("options.dropout", concrete.options.dropout);
                    tree.put("options.batch_first", concrete.options.batch_first);
                    tree.put("options.bidirectional", concrete.options.bidirectional);
                    tree.add_child("activation", Detail::serialize_activation_descriptor(concrete.activation));
                    tree.add_child("initialization", Detail::serialize_initialization_descriptor(concrete.initialization));
                    tree.add_child("local", serialize_local_config(concrete.local));
                } else if constexpr (std::is_same_v<DescriptorType, Layer::xLSTMDescriptor>) {
                    tree.put("type", "xlstm");
                    tree.put("options.input_size", concrete.options.input_size);
                    tree.put("options.hidden_size", concrete.options.hidden_size);
                    tree.put("options.num_layers", concrete.options.num_layers);
                    tree.put("options.dropout", concrete.options.dropout);
                    tree.put("options.batch_first", concrete.options.batch_first);
                    tree.put("options.bidirectional", concrete.options.bidirectional);
                    tree.add_child("activation", Detail::serialize_activation_descriptor(concrete.activation));
                    tree.add_child("initialization", Detail::serialize_initialization_descriptor(concrete.initialization));
                    tree.add_child("local", serialize_local_config(concrete.local));
                } else if constexpr (std::is_same_v<DescriptorType, Layer::GRUDescriptor>) {
                    tree.put("type", "gru");
                    tree.put("options.input_size", concrete.options.input_size);
                    tree.put("options.hidden_size", concrete.options.hidden_size);
                    tree.put("options.num_layers", concrete.options.num_layers);
                    tree.put("options.dropout", concrete.options.dropout);
                    tree.put("options.batch_first", concrete.options.batch_first);
                    tree.put("options.bidirectional", concrete.options.bidirectional);
                    tree.add_child("activation", Detail::serialize_activation_descriptor(concrete.activation));
                    tree.add_child("initialization", Detail::serialize_initialization_descriptor(concrete.initialization));
                    tree.add_child("local", serialize_local_config(concrete.local));
                } else if constexpr (std::is_same_v<DescriptorType, Layer::S4Descriptor>) {
                    tree.put("type", "s4");
                    tree.put("options.input_size", concrete.options.input_size);
                    tree.put("options.state_size", concrete.options.state_size);
                    tree.put("options.rank", concrete.options.rank);
                    tree.put("options.output_size", concrete.options.output_size);
                    tree.put("options.batch_first", concrete.options.batch_first);
                    tree.put("options.bidirectional", concrete.options.bidirectional);
                    tree.put("options.dropout", concrete.options.dropout);
                    tree.put("options.initialization", Detail::s4_initialization_to_string(concrete.options.initialization));
                    tree.put("options.maximum_length", concrete.options.maximum_length);
                    tree.add_child("activation", Detail::serialize_activation_descriptor(concrete.activation));
                    tree.add_child("initialization", Detail::serialize_initialization_descriptor(concrete.initialization));
                    tree.add_child("local", serialize_local_config(concrete.local));
                } else if constexpr (std::is_same_v<DescriptorType, Layer::PatchUnembedDescriptor>) {
                    tree.put("type", "patchunembed");
                    tree.put("options.channels", concrete.options.channels);
                    tree.put("options.tokens_height", concrete.options.tokens_height);
                    tree.put("options.tokens_width", concrete.options.tokens_width);
                    tree.put("options.patch_size", concrete.options.patch_size);
                    tree.put("options.target_height", concrete.options.target_height);
                    tree.put("options.target_width", concrete.options.target_width);
                    tree.put("options.align_corners", concrete.options.align_corners);
                    tree.add_child("activation", Detail::serialize_activation_descriptor(concrete.activation));
                    tree.add_child("local", serialize_local_config(concrete.local));
                } else {
                    static_assert(sizeof(DescriptorType) == 0, "Unsupported layer descriptor supplied.");
                }
            },
            descriptor);
        return tree;
    }

    inline Layer::Descriptor deserialize_layer_descriptor(const PropertyTree& tree, const std::string& context)
    {
        const auto type = Detail::to_lower(Detail::get_string(tree, "type", context));
        if (type == "fc") {
            Layer::Details::FCDescriptor descriptor;
            descriptor.options.in_features = Detail::get_numeric<std::int64_t>(tree, "options.in_features", context);
            descriptor.options.out_features = Detail::get_numeric<std::int64_t>(tree, "options.out_features", context);
            descriptor.options.bias = Detail::get_boolean(tree, "options.bias", context);
            descriptor.activation = Detail::deserialize_activation_descriptor(tree.get_child("activation"), context);
            descriptor.initialization = Detail::deserialize_initialization_descriptor(tree.get_child("initialization"), context);
            descriptor.local = deserialize_local_config(tree.get_child("local"), context);
            return Layer::Descriptor{descriptor};
        }
        if (type == "conv1d") {
            Layer::Details::Conv1dDescriptor descriptor;
            descriptor.options.in_channels = Detail::get_numeric<std::int64_t>(tree, "options.in_channels", context);
            descriptor.options.out_channels = Detail::get_numeric<std::int64_t>(tree, "options.out_channels", context);
            descriptor.options.kernel_size = Detail::read_array<std::int64_t>(tree.get_child("options.kernel_size"), context);
            descriptor.options.stride = Detail::read_array<std::int64_t>(tree.get_child("options.stride"), context);
            descriptor.options.padding = Detail::read_array<std::int64_t>(tree.get_child("options.padding"), context);
            descriptor.options.dilation = Detail::read_array<std::int64_t>(tree.get_child("options.dilation"), context);
            descriptor.options.groups = Detail::get_numeric<std::int64_t>(tree, "options.groups", context);
            descriptor.options.bias = Detail::get_boolean(tree, "options.bias", context);
            descriptor.options.padding_mode = Detail::get_string(tree, "options.padding_mode", context);
            descriptor.activation = Detail::deserialize_activation_descriptor(tree.get_child("activation"), context);
            descriptor.initialization = Detail::deserialize_initialization_descriptor(tree.get_child("initialization"), context);
            descriptor.local = deserialize_local_config(tree.get_child("local"), context);
            return Layer::Descriptor{descriptor};
        }
        if (type == "conv2d") {
            Layer::Details::Conv2dDescriptor descriptor;
            descriptor.options.in_channels = Detail::get_numeric<std::int64_t>(tree, "options.in_channels", context);
            descriptor.options.out_channels = Detail::get_numeric<std::int64_t>(tree, "options.out_channels", context);
            descriptor.options.kernel_size = Detail::read_array<std::int64_t>(tree.get_child("options.kernel_size"), context);
            descriptor.options.stride = Detail::read_array<std::int64_t>(tree.get_child("options.stride"), context);
            descriptor.options.padding = Detail::read_array<std::int64_t>(tree.get_child("options.padding"), context);
            descriptor.options.dilation = Detail::read_array<std::int64_t>(tree.get_child("options.dilation"), context);
            descriptor.options.groups = Detail::get_numeric<std::int64_t>(tree, "options.groups", context);
            descriptor.options.bias = Detail::get_boolean(tree, "options.bias", context);
            descriptor.options.padding_mode = Detail::get_string(tree, "options.padding_mode", context);
            descriptor.activation = Detail::deserialize_activation_descriptor(tree.get_child("activation"), context);
            descriptor.initialization = Detail::deserialize_initialization_descriptor(tree.get_child("initialization"), context);
            descriptor.local = deserialize_local_config(tree.get_child("local"), context);
            return Layer::Descriptor{descriptor};
        }
        if (type == "batch_norm2d") {
            Layer::Details::BatchNorm2dDescriptor descriptor;
            descriptor.options.num_features = Detail::get_numeric<std::int64_t>(tree, "options.num_features", context);
            descriptor.options.eps = Detail::get_numeric<double>(tree, "options.eps", context);
            descriptor.options.momentum = Detail::get_numeric<double>(tree, "options.momentum", context);
            descriptor.options.affine = Detail::get_boolean(tree, "options.affine", context);
            descriptor.options.track_running_stats = Detail::get_boolean(tree, "options.track_running_stats", context);
            descriptor.activation = Detail::deserialize_activation_descriptor(tree.get_child("activation"), context);
            descriptor.initialization = Detail::deserialize_initialization_descriptor(tree.get_child("initialization"), context);
            descriptor.local = deserialize_local_config(tree.get_child("local"), context);
            return Layer::Descriptor{descriptor};
        }
        if (type == "pooling") {
            Layer::Details::PoolingDescriptor descriptor;
            descriptor.options = Detail::pooling_variant_from_string(Detail::get_string(tree, "options.variant", context));
            descriptor.activation = Detail::deserialize_activation_descriptor(tree.get_child("activation"), context);
            descriptor.local = deserialize_local_config(tree.get_child("local"), context);
            std::visit(
                [&](auto& options) {
                    using OptionType = std::decay_t<decltype(options)>;
                    if constexpr (std::is_same_v<OptionType, Layer::Details::MaxPool1dOptions>) {
                        options.kernel_size = Detail::read_array<std::int64_t>(tree.get_child("options.kernel_size"), context);
                        options.stride = Detail::read_array<std::int64_t>(tree.get_child("options.stride"), context);
                        options.padding = Detail::read_array<std::int64_t>(tree.get_child("options.padding"), context);
                        options.dilation = Detail::read_array<std::int64_t>(tree.get_child("options.dilation"), context);
                        options.ceil_mode = Detail::get_boolean(tree, "options.ceil_mode", context);
                    } else if constexpr (std::is_same_v<OptionType, Layer::Details::AvgPool1dOptions>) {
                        options.kernel_size = Detail::read_array<std::int64_t>(tree.get_child("options.kernel_size"), context);
                        options.stride = Detail::read_array<std::int64_t>(tree.get_child("options.stride"), context);
                        options.padding = Detail::read_array<std::int64_t>(tree.get_child("options.padding"), context);
                        options.ceil_mode = Detail::get_boolean(tree, "options.ceil_mode", context);
                        options.count_include_pad = Detail::get_boolean(tree, "options.count_include_pad", context);
                    } else if constexpr (std::is_same_v<OptionType, Layer::Details::AdaptiveAvgPool1dOptions>) {
                        options.output_size = Detail::read_array<std::int64_t>(tree.get_child("options.output_size"), context);
                    } else if constexpr (std::is_same_v<OptionType, Layer::Details::AdaptiveMaxPool1dOptions>) {
                        options.output_size = Detail::read_array<std::int64_t>(tree.get_child("options.output_size"), context);
                    } else if constexpr (std::is_same_v<OptionType, Layer::Details::MaxPool2dOptions>) {
                        options.kernel_size = Detail::read_array<std::int64_t>(tree.get_child("options.kernel_size"), context);
                        options.stride = Detail::read_array<std::int64_t>(tree.get_child("options.stride"), context);
                        options.padding = Detail::read_array<std::int64_t>(tree.get_child("options.padding"), context);
                        options.dilation = Detail::read_array<std::int64_t>(tree.get_child("options.dilation"), context);
                        options.ceil_mode = Detail::get_boolean(tree, "options.ceil_mode", context);
                    } else if constexpr (std::is_same_v<OptionType, Layer::Details::AvgPool2dOptions>) {
                        options.kernel_size = Detail::read_array<std::int64_t>(tree.get_child("options.kernel_size"), context);
                        options.stride = Detail::read_array<std::int64_t>(tree.get_child("options.stride"), context);
                        options.padding = Detail::read_array<std::int64_t>(tree.get_child("options.padding"), context);
                        options.ceil_mode = Detail::get_boolean(tree, "options.ceil_mode", context);
                        options.count_include_pad = Detail::get_boolean(tree, "options.count_include_pad", context);
                    } else if constexpr (std::is_same_v<OptionType, Layer::Details::AdaptiveAvgPool2dOptions>) {
                        options.output_size = Detail::read_array<std::int64_t>(tree.get_child("options.output_size"), context);
                    } else if constexpr (std::is_same_v<OptionType, Layer::Details::AdaptiveMaxPool2dOptions>) {
                        options.output_size = Detail::read_array<std::int64_t>(tree.get_child("options.output_size"), context);
                    }
                },
                descriptor.options);
            return Layer::Descriptor{descriptor};
        }
        if (type == "harddropout") {
            Layer::Details::HardDropoutDescriptor descriptor;
            descriptor.options.probability = Detail::get_numeric<double>(tree, "options.probability", context);
            descriptor.options.inplace = Detail::get_boolean(tree, "options.inplace", context);
            descriptor.activation = Detail::deserialize_activation_descriptor(tree.get_child("activation"), context);
            descriptor.local = deserialize_local_config(tree.get_child("local"), context);
            return Layer::Descriptor{descriptor};
        }
        if (type == "softdropout") {
            Layer::Details::SoftDropoutDescriptor descriptor;
            descriptor.options.probability = Detail::get_numeric<double>(tree, "options.probability", context);
            descriptor.options.inplace = Detail::get_boolean(tree, "options.inplace", context);
            descriptor.options.noise_mean = Detail::get_numeric<double>(tree, "options.noise_mean", context);
            descriptor.options.noise_std = Detail::get_numeric<double>(tree, "options.noise_std", context);
            if (const auto noise_type = tree.get_optional<std::string>("options.noise_type")) {
                descriptor.options.noise_type = Detail::soft_dropout_noise_type_from_string(*noise_type);
            }
            descriptor.activation = Detail::deserialize_activation_descriptor(tree.get_child("activation"), context);
            descriptor.local = deserialize_local_config(tree.get_child("local"), context);
            return Layer::Descriptor{descriptor};
        }
        if (type == "flatten") {
            Layer::Details::FlattenDescriptor descriptor;
            descriptor.options.start_dim = Detail::get_numeric<std::int64_t>(tree, "options.start_dim", context);
            descriptor.options.end_dim = Detail::get_numeric<std::int64_t>(tree, "options.end_dim", context);
            descriptor.activation = Detail::deserialize_activation_descriptor(tree.get_child("activation"), context);
            descriptor.local = deserialize_local_config(tree.get_child("local"), context);
            return Layer::Descriptor{descriptor};
        }
        if (type == "upsample") {
            Layer::Details::UpsampleDescriptor descriptor;
            descriptor.options.scale = Detail::read_array<double>(tree.get_child("options.scale"), context);
            descriptor.options.mode = Detail::upsample_mode_from_string(Detail::get_string(tree, "options.mode", context), context);
            descriptor.options.align_corners = Detail::get_boolean(tree, "options.align_corners", context);
            descriptor.options.recompute_scale_factor = Detail::get_boolean(tree, "options.recompute_scale_factor", context);
            descriptor.activation = Detail::deserialize_activation_descriptor(tree.get_child("activation"), context);
            descriptor.local = deserialize_local_config(tree.get_child("local"), context);
            return Layer::Descriptor{descriptor};
        }
        if (type == "downsample") {
            Layer::Details::DownsampleDescriptor descriptor;
            descriptor.options.scale = Detail::read_array<double>(tree.get_child("options.scale"), context);
            descriptor.options.mode = Detail::downsample_mode_from_string(Detail::get_string(tree, "options.mode", context), context);
            descriptor.options.align_corners = Detail::get_boolean(tree, "options.align_corners", context);
            descriptor.options.recompute_scale_factor = Detail::get_boolean(tree, "options.recompute_scale_factor", context);
            descriptor.activation = Detail::deserialize_activation_descriptor(tree.get_child("activation"), context);
            descriptor.local = deserialize_local_config(tree.get_child("local"), context);
            return Layer::Descriptor{descriptor};
        }
        if (type == "reduce") {
            Layer::Details::ReduceDescriptor descriptor;
            descriptor.options.op = Detail::reduce_op_from_string(Detail::get_string(tree, "options.op", context));
            if (auto dims = tree.get_child_optional("options.dims")) {
                descriptor.options.dims = Detail::read_array<std::int64_t>(*dims, context);
            }
            descriptor.options.keep_dim = Detail::get_boolean(tree, "options.keep_dim", context);
            descriptor.activation = Detail::deserialize_activation_descriptor(tree.get_child("activation"), context);
            descriptor.local = deserialize_local_config(tree.get_child("local"), context);
            return Layer::Descriptor{descriptor};
        }
        if (type == "rnn") {
            Layer::Details::RNNDescriptor descriptor;
            descriptor.options.input_size = Detail::get_numeric<std::int64_t>(tree, "options.input_size", context);
            descriptor.options.hidden_size = Detail::get_numeric<std::int64_t>(tree, "options.hidden_size", context);
            descriptor.options.num_layers = Detail::get_numeric<std::int64_t>(tree, "options.num_layers", context);
            descriptor.options.dropout = Detail::get_numeric<double>(tree, "options.dropout", context);
            descriptor.options.batch_first = Detail::get_boolean(tree, "options.batch_first", context);
            descriptor.options.bidirectional = Detail::get_boolean(tree, "options.bidirectional", context);
            descriptor.options.nonlinearity = Detail::get_string(tree, "options.nonlinearity", context);
            descriptor.activation = Detail::deserialize_activation_descriptor(tree.get_child("activation"), context);
            descriptor.initialization = Detail::deserialize_initialization_descriptor(tree.get_child("initialization"), context);
            descriptor.local = deserialize_local_config(tree.get_child("local"), context);
            return Layer::Descriptor{descriptor};
        }
        if (type == "lstm") {
            Layer::Details::LSTMDescriptor descriptor;
            descriptor.options.input_size = Detail::get_numeric<std::int64_t>(tree, "options.input_size", context);
            descriptor.options.hidden_size = Detail::get_numeric<std::int64_t>(tree, "options.hidden_size", context);
            descriptor.options.num_layers = Detail::get_numeric<std::int64_t>(tree, "options.num_layers", context);
            descriptor.options.dropout = Detail::get_numeric<double>(tree, "options.dropout", context);
            descriptor.options.batch_first = Detail::get_boolean(tree, "options.batch_first", context);
            descriptor.options.bidirectional = Detail::get_boolean(tree, "options.bidirectional", context);
            descriptor.activation = Detail::deserialize_activation_descriptor(tree.get_child("activation"), context);
            descriptor.initialization = Detail::deserialize_initialization_descriptor(tree.get_child("initialization"), context);
            descriptor.local = deserialize_local_config(tree.get_child("local"), context);
            return Layer::Descriptor{descriptor};
        }
        if (type == "xlstm") {
            Layer::Details::xLSTMDescriptor descriptor;
            descriptor.options.input_size = Detail::get_numeric<std::int64_t>(tree, "options.input_size", context);
            descriptor.options.hidden_size = Detail::get_numeric<std::int64_t>(tree, "options.hidden_size", context);
            descriptor.options.num_layers = Detail::get_numeric<std::int64_t>(tree, "options.num_layers", context);
            descriptor.options.dropout = Detail::get_numeric<double>(tree, "options.dropout", context);
            descriptor.options.batch_first = Detail::get_boolean(tree, "options.batch_first", context);
            descriptor.options.bidirectional = Detail::get_boolean(tree, "options.bidirectional", context);
            descriptor.activation = Detail::deserialize_activation_descriptor(tree.get_child("activation"), context);
            descriptor.initialization = Detail::deserialize_initialization_descriptor(tree.get_child("initialization"), context);
            descriptor.local = deserialize_local_config(tree.get_child("local"), context);
            return Layer::Descriptor{descriptor};
        }
        if (type == "gru") {
            Layer::Details::GRUDescriptor descriptor;
            descriptor.options.input_size = Detail::get_numeric<std::int64_t>(tree, "options.input_size", context);
            descriptor.options.hidden_size = Detail::get_numeric<std::int64_t>(tree, "options.hidden_size", context);
            descriptor.options.num_layers = Detail::get_numeric<std::int64_t>(tree, "options.num_layers", context);
            descriptor.options.dropout = Detail::get_numeric<double>(tree, "options.dropout", context);
            descriptor.options.batch_first = Detail::get_boolean(tree, "options.batch_first", context);
            descriptor.options.bidirectional = Detail::get_boolean(tree, "options.bidirectional", context);
            descriptor.activation = Detail::deserialize_activation_descriptor(tree.get_child("activation"), context);
            descriptor.initialization = Detail::deserialize_initialization_descriptor(tree.get_child("initialization"), context);
            descriptor.local = deserialize_local_config(tree.get_child("local"), context);
            return Layer::Descriptor{descriptor};
        }
        if (type == "s4") {
            Layer::Details::S4Descriptor descriptor;
            descriptor.options.input_size = Detail::get_numeric<std::int64_t>(tree, "options.input_size", context);
            descriptor.options.state_size = Detail::get_numeric<std::int64_t>(tree, "options.state_size", context);
            descriptor.options.rank = Detail::get_numeric<std::int64_t>(tree, "options.rank", context);
            descriptor.options.output_size = Detail::get_numeric<std::int64_t>(tree, "options.output_size", context);
            descriptor.options.batch_first = Detail::get_boolean(tree, "options.batch_first", context);
            descriptor.options.bidirectional = Detail::get_boolean(tree, "options.bidirectional", context);
            descriptor.options.dropout = Detail::get_numeric<double>(tree, "options.dropout", context);
            descriptor.options.initialization =
                Detail::s4_initialization_from_string(Detail::get_string(tree, "options.initialization", context));
            descriptor.options.maximum_length = Detail::get_numeric<std::int64_t>(tree, "options.maximum_length", context);
            descriptor.activation = Detail::deserialize_activation_descriptor(tree.get_child("activation"), context);
            descriptor.initialization = Detail::deserialize_initialization_descriptor(tree.get_child("initialization"), context);
            descriptor.local = deserialize_local_config(tree.get_child("local"), context);
            return Layer::Descriptor{descriptor};
        }
        if (type == "patchunembed") {
            Layer::Details::PatchUnembedDescriptor descriptor;
            descriptor.options.channels = Detail::get_numeric<std::int64_t>(tree, "options.channels", context);
            descriptor.options.tokens_height = Detail::get_numeric<std::int64_t>(tree, "options.tokens_height", context);
            descriptor.options.tokens_width = Detail::get_numeric<std::int64_t>(tree, "options.tokens_width", context);
            descriptor.options.patch_size = Detail::get_numeric<std::int64_t>(tree, "options.patch_size", context);
            descriptor.options.target_height = Detail::get_numeric<std::int64_t>(tree, "options.target_height", context);
            descriptor.options.target_width = Detail::get_numeric<std::int64_t>(tree, "options.target_width", context);
            descriptor.options.align_corners = Detail::get_boolean(tree, "options.align_corners", context);
            descriptor.activation = Detail::deserialize_activation_descriptor(tree.get_child("activation"), context);
            descriptor.local = deserialize_local_config(tree.get_child("local"), context);
            return Layer::Descriptor{descriptor};
        }
        std::ostringstream message;
        message << "Unknown layer descriptor '" << type << "' in " << context;
        throw std::runtime_error(message.str());
    }

    inline PropertyTree serialize_block_descriptor(const Block::Descriptor& descriptor) {
        PropertyTree tree;
        std::visit(
            [&](const auto& concrete) {
                using DescriptorType = std::decay_t<decltype(concrete)>;
                if constexpr (std::is_same_v<DescriptorType, Block::Details::SequentialDescriptor>) {

                    tree.put("type", "sequential");
                    PropertyTree layers;
                    for (const auto& layer : concrete.layers)
                        layers.push_back({"", serialize_layer_descriptor(layer)});

                    tree.add_child("layers", layers);
                    tree.add_child("local", serialize_local_config(concrete.local));

                } else if constexpr (std::is_same_v<DescriptorType, Block::Details::ResidualDescriptor>) {
                    tree.put("type", "residual");
                    PropertyTree layers;
                    for (const auto& layer : concrete.layers)
                        layers.push_back({"", serialize_layer_descriptor(layer)});

                    tree.add_child("layers", layers);
                    tree.put("repeats", static_cast<std::uint64_t>(concrete.repeats));

                    if (concrete.skip.projection)
                        tree.add_child("skip.projection", serialize_layer_descriptor(*concrete.skip.projection));

                    tree.add_child("output.final_activation",Detail::serialize_activation_descriptor(concrete.output.final_activation));
                    tree.put("output.dropout", concrete.output.dropout);
                    tree.add_child("local", serialize_local_config(concrete.local));

                } else if constexpr (std::is_same_v<DescriptorType, Detail::Classic::EncoderDescriptor>) {
                    tree.put("type", "transformer_encoder");
                    tree.add_child("options.attention", Detail::serialize_classic_attention_options(concrete.options.attention));
                    tree.add_child("options.feed_forward",Detail::serialize_classic_feed_forward_options(concrete.options.feed_forward));
                    tree.add_child("options.layer_norm",Detail::serialize_classic_layer_norm_options(concrete.options.layer_norm));
                    tree.add_child("options.positional_encoding", Detail::serialize_classic_positional_encoding_options(concrete.options.positional_encoding));
                    tree.put("options.layers", static_cast<std::uint64_t>(concrete.options.layers));
                    tree.put("options.embed_dim", concrete.options.embed_dim);
                    tree.put("options.dropout", concrete.options.dropout);
                    PropertyTree layers;
                    for (const auto& layer : concrete.layers) {
                        PropertyTree entry;
                        entry.add_child("attention", serialize_attention(layer.attention));
                        entry.add_child("attention_dropout", serialize_layer_descriptor(layer.attention_dropout));
                        PropertyTree feed_forward_layers;

                        for (const auto& ff : layer.feed_forward)
                            feed_forward_layers.push_back({"", serialize_layer_descriptor(ff)});

                        entry.add_child("feed_forward", feed_forward_layers);
                        entry.add_child("feed_forward_dropout", serialize_layer_descriptor(layer.feed_forward_dropout));
                        layers.push_back({"", entry});
                    }
                    tree.add_child("layers", layers);

                } else if constexpr (std::is_same_v<DescriptorType, Detail::Classic::DecoderDescriptor>) {
                    tree.put("type", "transformer_decoder");
                    tree.add_child("options.self_attention",Detail::serialize_classic_attention_options(concrete.options.self_attention));
                    tree.add_child("options.cross_attention",Detail::serialize_classic_attention_options(concrete.options.cross_attention));
                    tree.add_child("options.feed_forward",Detail::serialize_classic_feed_forward_options(concrete.options.feed_forward));
                    tree.add_child("options.layer_norm",Detail::serialize_classic_layer_norm_options(concrete.options.layer_norm));
                    tree.add_child("options.positional_encoding", Detail::serialize_classic_positional_encoding_options(concrete.options.positional_encoding));
                    tree.put("options.layers", static_cast<std::uint64_t>(concrete.options.layers));
                    tree.put("options.embed_dim", concrete.options.embed_dim);
                    tree.put("options.dropout", concrete.options.dropout);

                    PropertyTree layers;

                    for (const auto& layer : concrete.layers) {
                        PropertyTree entry;
                        entry.add_child("self_attention", serialize_attention(layer.self_attention));
                        entry.add_child("self_attention_dropout", serialize_layer_descriptor(layer.self_attention_dropout));
                        entry.add_child("cross_attention", serialize_attention(layer.cross_attention));
                        entry.add_child("cross_attention_dropout", serialize_layer_descriptor(layer.cross_attention_dropout));
                        PropertyTree feed_forward_layers;

                        for (const auto& ff : layer.feed_forward)
                            feed_forward_layers.push_back({"", serialize_layer_descriptor(ff)});

                        entry.add_child("feed_forward", feed_forward_layers);
                        entry.add_child("feed_forward_dropout", serialize_layer_descriptor(layer.feed_forward_dropout));
                        layers.push_back({"", entry});
                    }
                    tree.add_child("layers", layers);

                } else if constexpr (std::is_same_v<DescriptorType, Detail::PlusPlus::EncoderDescriptor>) {
                    tree.put("type", "transformer_pp_encoder");
                    tree.put("options.layers", static_cast<std::uint64_t>(concrete.options.layers));
                    tree.put("options.embed_dim", concrete.options.embed_dim);
                    tree.add_child("options.hybrid_attention",Detail::serialize_transformer_pp_hybrid_attention_options(concrete.options.hybrid_attention));
                    tree.add_child("options.feed_forward",Detail::serialize_transformer_pp_feed_forward_options(concrete.options.feed_forward));
                    tree.add_child("options.layer_norm",Detail::serialize_transformer_pp_layer_norm_options(concrete.options.layer_norm));
                    tree.add_child("options.positional_encoding", Detail::serialize_classic_positional_encoding_options(concrete.options.positional_encoding));
                    tree.put("options.dropout", concrete.options.dropout);
                    tree.add_child("options.pos_head",Detail::serialize_transformer_pp_auxiliary_head_options(concrete.options.pos_head));
                    tree.add_child("options.ner_head",Detail::serialize_transformer_pp_auxiliary_head_options(concrete.options.ner_head));
                    PropertyTree layers;

                    for (const auto& layer : concrete.layers)
                        layers.push_back({"", Detail::serialize_transformer_pp_encoder_layer_descriptor(layer)});

                    tree.add_child("layers", layers);

                } else if constexpr (std::is_same_v<DescriptorType, Detail::PlusPlus::DecoderDescriptor>) {
                    tree.put("type", "transformer_pp_decoder");
                    tree.put("options.layers", static_cast<std::uint64_t>(concrete.options.layers));
                    tree.put("options.embed_dim", concrete.options.embed_dim);
                    tree.add_child("options.self_attention",Detail::serialize_transformer_pp_hybrid_attention_options(concrete.options.self_attention));
                    tree.add_child("options.cross_attention",Detail::serialize_transformer_pp_hybrid_attention_options(concrete.options.cross_attention));
                    tree.add_child("options.feed_forward",Detail::serialize_transformer_pp_feed_forward_options(concrete.options.feed_forward));
                    tree.add_child("options.layer_norm",Detail::serialize_transformer_pp_layer_norm_options(concrete.options.layer_norm));
                    tree.add_child("options.positional_encoding", Detail::serialize_classic_positional_encoding_options(concrete.options.positional_encoding));
                    tree.put("options.dropout", concrete.options.dropout);
                    tree.add_child("options.pos_head",Detail::serialize_transformer_pp_auxiliary_head_options(concrete.options.pos_head));
                    tree.add_child("options.ner_head",Detail::serialize_transformer_pp_auxiliary_head_options(concrete.options.ner_head));
                    PropertyTree layers;

                    for (const auto& layer : concrete.layers)
                        layers.push_back({"", Detail::serialize_transformer_pp_decoder_layer_descriptor(layer)});

                    tree.add_child("layers", layers);

                } else if constexpr (std::is_same_v<DescriptorType, Detail::EBT::EncoderDescriptor>) {
                    tree.put("type", "ebt_encoder");
                    tree.add_child("options.modality",Detail::serialize_ebt_modality_options(concrete.options.modality));
                    tree.add_child("options.energy", Detail::serialize_ebt_energy_options(concrete.options.energy));
                    tree.add_child("options.optimizer",Detail::serialize_ebt_optimizer_options(concrete.options.optimizer));
                    tree.add_child("options.refinement",Detail::serialize_ebt_refinement_options(concrete.options.refinement));

                } else if constexpr (std::is_same_v<DescriptorType, Detail::EBT::DecoderDescriptor>) {
                    tree.put("type", "ebt_decoder");
                    tree.add_child("options.target",Detail::serialize_ebt_modality_options(concrete.options.target));

                    if (concrete.options.context.has_value())
                        tree.add_child("options.context", Detail::serialize_ebt_modality_options(*concrete.options.context));

                    tree.add_child("options.energy", Detail::serialize_ebt_energy_options(concrete.options.energy));
                    tree.add_child("options.optimizer", Detail::serialize_ebt_optimizer_options(concrete.options.optimizer));
                    tree.add_child("options.refinement", Detail::serialize_ebt_refinement_options(concrete.options.refinement));

                } else if constexpr (std::is_same_v<DescriptorType, Detail::Mamba::EncoderDescriptor>) {
                    tree.put("type", "mamba_encoder");
                    tree.put("options.layers", static_cast<std::uint64_t>(concrete.options.layers));
                    tree.put("options.embed_dim", concrete.options.embed_dim);
                    tree.add_child("options.rms_norm", Detail::serialize_mamba_rms_norm_options(concrete.options.rms_norm));
                    tree.put("options.normalization", Detail::serialize_mamba_normalization_order(concrete.options.normalization));
                    tree.put("options.residual_dropout", concrete.options.residual_dropout);
                    tree.put("options.feed_forward_dropout", concrete.options.feed_forward_dropout);
                    tree.put("options.residual_gating", concrete.options.residual_gating);
                    tree.put("options.feed_forward_gating", concrete.options.feed_forward_gating);
                    tree.put("options.batch_first", concrete.options.batch_first);
                    tree.put("options.final_layer_norm", concrete.options.final_layer_norm);
                    tree.add_child("options.selective_state", Detail::serialize_mamba_selective_state_options(concrete.options.selective_state));
                    tree.add_child("options.feed_forward", Detail::serialize_mamba_feed_forward_options(concrete.options.feed_forward));

                    PropertyTree layers;
                    for (const auto& layer : concrete.layers) {
                        PropertyTree entry;
                        entry.add_child("selective_state", Detail::serialize_mamba_selective_state_options(layer.selective_state));
                        entry.add_child("feed_forward", Detail::serialize_mamba_feed_forward_options(layer.feed_forward));
                        layers.push_back({"", entry});
                    }

                    tree.add_child("layers", layers);
                } else if constexpr (std::is_same_v<
                    DescriptorType,
                    ::Nott::Block::Details::Transformer::Perceiver::EncoderDescriptor>) {

                    tree.put("type", "perceiver_encoder");

                    tree.put("options.layers", static_cast<std::uint64_t>(concrete.options.layers));
                    tree.put("options.self_layers", static_cast<std::uint64_t>(concrete.options.self_layers));
                    tree.put("options.repeats", static_cast<std::uint64_t>(concrete.options.repeats));
                    tree.put("options.latent_dim", concrete.options.latent_dim);
                    tree.put("options.input_dim", concrete.options.input_dim);
                    tree.put("options.latent_slots", static_cast<std::uint64_t>(concrete.options.latent_slots));
                    tree.add_child("options.cross_attention", Detail::serialize_perceiver_attention_options(concrete.options.cross_attention));
                    tree.add_child("options.self_attention", Detail::serialize_perceiver_attention_options(concrete.options.self_attention));
                    tree.add_child("options.feed_forward", Detail::serialize_perceiver_feed_forward_options(concrete.options.feed_forward));
                    tree.put("options.residual_dropout", concrete.options.residual_dropout);
                    tree.put("options.attention_dropout", concrete.options.attention_dropout);

                    PropertyTree layers;
                    for (const auto &layer : concrete.layers) {
                        PropertyTree entry;
                        entry.add_child("feed_forward", Detail::serialize_perceiver_feed_forward_options(layer.feed_forward));
                        layers.push_back({"", entry});
                    }
                    tree.add_child("layers", layers);

                } else if constexpr (std::is_same_v<DescriptorType, ::Nott::Block::Details::Transformer::Bert::EncoderDescriptor>) {
                    tree.put("type", "bert_encoder");
                    tree.put("options.layers", static_cast<std::uint64_t>(concrete.options.layers));
                    tree.put("options.embed_dim", concrete.options.embed_dim);

                    tree.add_child("options.attention", Detail::serialize_bert_attention_options(concrete.options.attention));
                    tree.add_child("options.feed_forward", Detail::serialize_bert_feed_forward_options(concrete.options.feed_forward));
                    tree.add_child("options.layer_norm", Detail::serialize_bert_layer_norm_options(concrete.options.layer_norm));
                    tree.add_child("options.embedding", Detail::serialize_bert_embedding_options(concrete.options.embedding));
                    tree.put("options.residual_dropout", concrete.options.residual_dropout);
                    tree.put("options.attention_dropout", concrete.options.attention_dropout);
                    tree.put("options.feed_forward_dropout", concrete.options.feed_forward_dropout);
                    tree.put("options.pre_norm", concrete.options.pre_norm);
                    tree.put("options.final_layer_norm", concrete.options.final_layer_norm);

                    /// layers vector
                    PropertyTree layers;
                    for (const auto &layer : concrete.layers) {
                        PropertyTree entry;
                        entry.add_child("attention", Detail::serialize_bert_attention_options(layer.attention));
                        entry.add_child("feed_forward", Detail::serialize_bert_feed_forward_options(layer.feed_forward));
                        layers.push_back({"", entry});
                    }
                    tree.add_child("layers", layers);

                    } else if constexpr (std::is_same_v<DescriptorType, ::Nott::Block::Details::Transformer::Vision::EncoderDescriptor>) {

                        tree.put("type", "vision_encoder");

                        /// top-level options
                        tree.put("options.layers", static_cast<std::uint64_t>(concrete.options.layers));
                        tree.put("options.embed_dim", concrete.options.embed_dim);
                        tree.put("options.variant", Detail::serialize_vision_variant(concrete.options.variant));
                        tree.add_child("options.attention", Detail::serialize_vision_attention_options(concrete.options.attention));
                        tree.add_child("options.feed_forward", Detail::serialize_vision_feed_forward_options(concrete.options.feed_forward));
                        tree.add_child("options.layer_norm", Detail::serialize_vision_layer_norm_options(concrete.options.layer_norm));
                        tree.add_child("options.patch_embedding", Detail::serialize_vision_patch_embedding_options(concrete.options.patch_embedding));
                        tree.add_child("options.window", Detail::serialize_vision_window_options(concrete.options.window));
                        tree.add_child("options.positional_encoding", Detail::serialize_positional_encoding_options(concrete.options.positional_encoding));
                        tree.put("options.residual_dropout", concrete.options.residual_dropout);
                        tree.put("options.attention_dropout", concrete.options.attention_dropout);
                        tree.put("options.feed_forward_dropout", concrete.options.feed_forward_dropout);
                        tree.put("options.pre_norm", concrete.options.pre_norm);
                        tree.put("options.final_layer_norm", concrete.options.final_layer_norm);

                        /// layers vector
                        PropertyTree layers;
                        for (const auto &layer : concrete.layers) {
                            PropertyTree entry;
                            entry.add_child("attention", Detail::serialize_vision_attention_options(layer.attention));
                            entry.add_child("feed_forward", Detail::serialize_vision_feed_forward_options(layer.feed_forward));
                            entry.add_child("window", Detail::serialize_vision_window_options(layer.window));
                            layers.push_back({"", entry});
                        }
                        tree.add_child("layers", layers);


                    } else if constexpr (std::is_same_v<
                        DescriptorType,
                        ::Nott::Block::Details::Transformer::LongformerXL::EncoderDescriptor>) {

                        tree.put("type", "longformer_xl_encoder");

                        /// top-level options
                        tree.put("options.layers", static_cast<std::uint64_t>(concrete.options.layers));
                        tree.put("options.embed_dim", concrete.options.embed_dim);
                        tree.add_child("options.attention", Detail::serialize_longformer_attention_options(concrete.options.attention));
                        tree.add_child("options.feed_forward", Detail::serialize_longformer_feed_forward_options(concrete.options.feed_forward));
                        tree.add_child("options.layer_norm", Detail::serialize_longformer_layer_norm_options(concrete.options.layer_norm));
                        tree.put("options.window_size", concrete.options.window_size);
                        tree.put("options.global_tokens", static_cast<std::uint64_t>(concrete.options.global_tokens));
                        tree.put("options.causal", concrete.options.causal);
                        tree.put("options.use_memory", concrete.options.use_memory);
                        tree.put("options.memory_size", static_cast<std::uint64_t>(concrete.options.memory_size));
                        tree.put("options.residual_dropout", concrete.options.residual_dropout);
                        tree.put("options.attention_dropout", concrete.options.attention_dropout);
                        tree.put("options.feed_forward_dropout", concrete.options.feed_forward_dropout);
                        tree.put("options.pre_norm", concrete.options.pre_norm);
                        tree.put("options.final_layer_norm", concrete.options.final_layer_norm);

                        /// per-layer descriptors
                        PropertyTree layers;
                        for (const auto &layer : concrete.layers) {
                            PropertyTree entry;
                            entry.add_child("attention", Detail::serialize_longformer_attention_options(layer.attention));
                            entry.add_child("feed_forward", Detail::serialize_longformer_feed_forward_options(layer.feed_forward));
                            layers.push_back({"", entry});
                        }
                        tree.add_child("layers", layers);

                } else {
                    static_assert(sizeof(DescriptorType) == 0, "Unsupported block descriptor supplied.");
                }
            },
            descriptor);
        return tree;
    }

    inline Block::Descriptor deserialize_block_descriptor(const PropertyTree& tree, const std::string& context)
    {
        const auto type = Detail::to_lower(Detail::get_string(tree, "type", context));
        if (type == "sequential") {
            Block::Details::SequentialDescriptor descriptor;
            for (const auto& node : tree.get_child("layers")) {
                descriptor.layers.push_back(deserialize_layer_descriptor(node.second, context + " sequential layer"));
            }
            descriptor.local = deserialize_local_config(tree.get_child("local"), context + " sequential local");
            return Block::Descriptor{descriptor};
        }
        if (type == "residual") {
            Block::Details::ResidualDescriptor descriptor;
            for (const auto& node : tree.get_child("layers")) {
                descriptor.layers.push_back(deserialize_layer_descriptor(node.second, context + " residual layer"));
            }
            descriptor.repeats = static_cast<std::size_t>(Detail::get_numeric<std::uint64_t>(tree, "repeats", context));
            if (const auto projection = tree.get_child_optional("skip.projection")) {
                descriptor.skip.projection = deserialize_layer_descriptor(*projection, context + " residual projection");
            }
            descriptor.output.final_activation =
                Detail::deserialize_activation_descriptor(tree.get_child("output.final_activation"), context);
            descriptor.output.dropout = Detail::get_numeric<double>(tree, "output.dropout", context);
            descriptor.local = deserialize_local_config(tree.get_child("local"), context + " residual local");
            return Block::Descriptor{descriptor};
        }
        if (type == "transformer_encoder") {
            Detail::Classic::EncoderDescriptor descriptor;
            descriptor.options.attention =
                Detail::deserialize_classic_attention_options(tree.get_child("options.attention"), context);
            descriptor.options.feed_forward =
                Detail::deserialize_classic_feed_forward_options(tree.get_child("options.feed_forward"), context);
            descriptor.options.layer_norm =
                Detail::deserialize_classic_layer_norm_options(tree.get_child("options.layer_norm"), context);
            descriptor.options.positional_encoding = Detail::deserialize_classic_positional_encoding_options(
                tree.get_child("options.positional_encoding"), context);
            descriptor.options.layers =
                static_cast<std::size_t>(Detail::get_numeric<std::uint64_t>(tree, "options.layers", context));
            descriptor.options.embed_dim = Detail::get_numeric<std::int64_t>(tree, "options.embed_dim", context);
            descriptor.options.dropout = Detail::get_numeric<double>(tree, "options.dropout", context);
            for (const auto& node : tree.get_child("layers")) {
                Detail::Classic::EncoderLayerDescriptor layer;
                layer.attention = deserialize_attention(node.second.get_child("attention"), context + " encoder attention");
                layer.attention_dropout = deserialize_layer_descriptor(node.second.get_child("attention_dropout"),
                                                                       context + " encoder attention dropout");
                for (const auto& feed_forward : node.second.get_child("feed_forward")) {
                    layer.feed_forward.push_back(
                        deserialize_layer_descriptor(feed_forward.second, context + " encoder feed-forward"));
                }
                layer.feed_forward_dropout = deserialize_layer_descriptor(node.second.get_child("feed_forward_dropout"),
                                                                           context + " encoder feed-forward dropout");
                descriptor.layers.push_back(std::move(layer));
            }
            return Block::Descriptor{descriptor};
        }
        if (type == "transformer_decoder") {
            Detail::Classic::DecoderDescriptor descriptor;
            descriptor.options.self_attention =
                Detail::deserialize_classic_attention_options(tree.get_child("options.self_attention"), context);
            descriptor.options.cross_attention =
                Detail::deserialize_classic_attention_options(tree.get_child("options.cross_attention"), context);
            descriptor.options.feed_forward =
                Detail::deserialize_classic_feed_forward_options(tree.get_child("options.feed_forward"), context);
            descriptor.options.layer_norm =
                Detail::deserialize_classic_layer_norm_options(tree.get_child("options.layer_norm"), context);
            descriptor.options.positional_encoding = Detail::deserialize_classic_positional_encoding_options(
                tree.get_child("options.positional_encoding"), context);
            descriptor.options.layers =
                static_cast<std::size_t>(Detail::get_numeric<std::uint64_t>(tree, "options.layers", context));
            descriptor.options.embed_dim = Detail::get_numeric<std::int64_t>(tree, "options.embed_dim", context);
            descriptor.options.dropout = Detail::get_numeric<double>(tree, "options.dropout", context);
            for (const auto& node : tree.get_child("layers")) {
                Detail::Classic::DecoderLayerDescriptor layer;
                layer.self_attention = deserialize_attention(node.second.get_child("self_attention"),
                                                             context + " decoder self-attention");
                layer.self_attention_dropout = deserialize_layer_descriptor(node.second.get_child("self_attention_dropout"),
                                                                            context + " decoder self-attention dropout");
                layer.cross_attention = deserialize_attention(node.second.get_child("cross_attention"),
                                                              context + " decoder cross-attention");
                layer.cross_attention_dropout = deserialize_layer_descriptor(node.second.get_child("cross_attention_dropout"),
                                                                             context + " decoder cross-attention dropout");
                for (const auto& feed_forward : node.second.get_child("feed_forward")) {
                    layer.feed_forward.push_back(deserialize_layer_descriptor(feed_forward.second, context + " decoder feed-forward"));
                }
                layer.feed_forward_dropout = deserialize_layer_descriptor(node.second.get_child("feed_forward_dropout"),
                                                                           context + " decoder feed-forward dropout");
                descriptor.layers.push_back(std::move(layer));
            }
            return Block::Descriptor{descriptor};
        }
        if (type == "transformer_pp_encoder") {
            Detail::PlusPlus::EncoderDescriptor descriptor;
            descriptor.options.layers = static_cast<std::size_t>(
                Detail::get_numeric<std::uint64_t>(tree, "options.layers", context));
            descriptor.options.embed_dim = Detail::get_numeric<std::int64_t>(tree, "options.embed_dim", context);
            descriptor.options.hybrid_attention = Detail::deserialize_transformer_pp_hybrid_attention_options(
                tree.get_child("options.hybrid_attention"), context);
            descriptor.options.feed_forward = Detail::deserialize_transformer_pp_feed_forward_options(
                tree.get_child("options.feed_forward"), context);
            descriptor.options.layer_norm = Detail::deserialize_transformer_pp_layer_norm_options(
                tree.get_child("options.layer_norm"), context);
            descriptor.options.positional_encoding = Detail::deserialize_classic_positional_encoding_options(
                tree.get_child("options.positional_encoding"), context);
            descriptor.options.dropout = Detail::get_numeric<double>(tree, "options.dropout", context);
            descriptor.options.pos_head = Detail::deserialize_transformer_pp_auxiliary_head_options(
                tree.get_child("options.pos_head"), context);
            descriptor.options.ner_head = Detail::deserialize_transformer_pp_auxiliary_head_options(
                tree.get_child("options.ner_head"), context);
            for (const auto& node : tree.get_child("layers")) {
                descriptor.layers.push_back(Detail::deserialize_transformer_pp_encoder_layer_descriptor(
                    node.second, context + " transformer++ encoder layer"));
            }
            return Block::Descriptor{descriptor};
        }
        if (type == "transformer_pp_decoder") {
            Detail::PlusPlus::DecoderDescriptor descriptor;
            descriptor.options.layers = static_cast<std::size_t>(
                Detail::get_numeric<std::uint64_t>(tree, "options.layers", context));
            descriptor.options.embed_dim = Detail::get_numeric<std::int64_t>(tree, "options.embed_dim", context);
            descriptor.options.self_attention = Detail::deserialize_transformer_pp_hybrid_attention_options(
                tree.get_child("options.self_attention"), context);
            descriptor.options.cross_attention = Detail::deserialize_transformer_pp_hybrid_attention_options(
                tree.get_child("options.cross_attention"), context);
            descriptor.options.feed_forward = Detail::deserialize_transformer_pp_feed_forward_options(
                tree.get_child("options.feed_forward"), context);
            descriptor.options.layer_norm = Detail::deserialize_transformer_pp_layer_norm_options(
                tree.get_child("options.layer_norm"), context);
            descriptor.options.positional_encoding = Detail::deserialize_classic_positional_encoding_options(
                tree.get_child("options.positional_encoding"), context);
            descriptor.options.dropout = Detail::get_numeric<double>(tree, "options.dropout", context);
            descriptor.options.pos_head = Detail::deserialize_transformer_pp_auxiliary_head_options(
                tree.get_child("options.pos_head"), context);
            descriptor.options.ner_head = Detail::deserialize_transformer_pp_auxiliary_head_options(
                tree.get_child("options.ner_head"), context);
            for (const auto& node : tree.get_child("layers")) {
                descriptor.layers.push_back(Detail::deserialize_transformer_pp_decoder_layer_descriptor(
                    node.second, context + " transformer++ decoder layer"));
            }
            return Block::Descriptor{descriptor};
        }
        if (type == "mamba_encoder") {
            Detail::Mamba::EncoderDescriptor descriptor;
            descriptor.options.layers = static_cast<std::size_t>(Detail::get_numeric<std::uint64_t>(tree, "options.layers", context));
            descriptor.options.embed_dim = Detail::get_numeric<std::int64_t>(tree, "options.embed_dim", context);
            descriptor.options.rms_norm = Detail::deserialize_mamba_rms_norm_options(tree.get_child("options.rms_norm"), context);
            descriptor.options.normalization = Detail::deserialize_mamba_normalization_order(
                Detail::get_string(tree, "options.normalization", context), context);
            descriptor.options.residual_dropout = Detail::get_numeric<double>(tree, "options.residual_dropout", context);
            descriptor.options.feed_forward_dropout = Detail::get_numeric<double>(tree, "options.feed_forward_dropout", context);
            descriptor.options.residual_gating = Detail::get_boolean(tree, "options.residual_gating", context);
            descriptor.options.feed_forward_gating = Detail::get_boolean(tree, "options.feed_forward_gating", context);
            descriptor.options.batch_first = Detail::get_boolean(tree, "options.batch_first", context);
            descriptor.options.final_layer_norm = Detail::get_boolean(tree, "options.final_layer_norm", context);
            descriptor.options.selective_state =
                Detail::deserialize_mamba_selective_state_options(tree.get_child("options.selective_state"), context);
            descriptor.options.feed_forward =
                Detail::deserialize_mamba_feed_forward_options(tree.get_child("options.feed_forward"), context);
            for (const auto& node : tree.get_child("layers")) {
                Detail::Mamba::EncoderLayerDescriptor layer;
                layer.selective_state = Detail::deserialize_mamba_selective_state_options(node.second.get_child("selective_state"), context);
                layer.feed_forward = Detail::deserialize_mamba_feed_forward_options(node.second.get_child("feed_forward"), context);
                descriptor.layers.push_back(std::move(layer));
            }
            return Block::Descriptor{descriptor};
        }
        if (type == "bert_encoder") {
            ::Nott::Block::Details::Transformer::Bert::EncoderDescriptor descriptor;
            const auto& options_tree = tree.get_child("options");
            descriptor.options = Detail::deserialize_bert_encoder_options(options_tree, context + " bert encoder options");
            for (const auto& node : tree.get_child("layers")) {
                descriptor.layers.push_back(Detail::deserialize_bert_encoder_layer_descriptor(
                    node.second, context + " bert encoder layer"));
            }
            return Block::Descriptor{descriptor};
        }
        if (type == "ebt_encoder") {
            Detail::EBT::EncoderDescriptor descriptor;
            descriptor.options.modality =
                Detail::deserialize_ebt_modality_options(tree.get_child("options.modality"), context);
            descriptor.options.energy =
                Detail::deserialize_ebt_energy_options(tree.get_child("options.energy"), context);
            descriptor.options.optimizer =
                Detail::deserialize_ebt_optimizer_options(tree.get_child("options.optimizer"), context);
            descriptor.options.refinement =
                Detail::deserialize_ebt_refinement_options(tree.get_child("options.refinement"), context);
            return Block::Descriptor{descriptor};
        }
        if (type == "ebt_decoder") {
            Detail::EBT::DecoderDescriptor descriptor;
            descriptor.options.target =
                Detail::deserialize_ebt_modality_options(tree.get_child("options.target"), context);
            if (const auto context_node = tree.get_child_optional("options.context")) {
                descriptor.options.context = Detail::deserialize_ebt_modality_options(*context_node, context);
            }
            descriptor.options.energy =
                Detail::deserialize_ebt_energy_options(tree.get_child("options.energy"), context);
            descriptor.options.optimizer =
                Detail::deserialize_ebt_optimizer_options(tree.get_child("options.optimizer"), context);
            descriptor.options.refinement =
                Detail::deserialize_ebt_refinement_options(tree.get_child("options.refinement"), context);
            return Block::Descriptor{descriptor};
        }
        std::ostringstream message;
        message << "Unknown block descriptor '" << type << "' in " << context;
        throw std::runtime_error(message.str());
    }

    inline PropertyTree serialize_module_descriptor_payload(const ModuleDescriptor& descriptor)
    {
        PropertyTree tree;
        if (std::holds_alternative<Layer::Descriptor>(descriptor)) {
            tree.put("kind", "layer");
            tree.add_child("descriptor", serialize_layer_descriptor(std::get<Layer::Descriptor>(descriptor)));
        } else {
            tree.put("kind", "block");
            tree.add_child("descriptor", serialize_block_descriptor(std::get<Block::Descriptor>(descriptor)));
        }
        return tree;
    }

    inline PropertyTree serialize_module_descriptor(const NamedModuleDescriptor& descriptor)
    {
        auto tree = serialize_module_descriptor_payload(descriptor.descriptor);
        if (!descriptor.name.empty()) {
            tree.put("name", descriptor.name);
        }
        return tree;
    }


    inline ModuleDescriptor deserialize_module_descriptor(const PropertyTree& tree, const std::string& context)
    {
        const auto kind = Detail::to_lower(Detail::get_string(tree, "kind", context));
        if (kind == "layer") {
            return ModuleDescriptor{deserialize_layer_descriptor(tree.get_child("descriptor"), context + " layer")};
        }
        if (kind == "block") {
            return ModuleDescriptor{deserialize_block_descriptor(tree.get_child("descriptor"), context + " block")};
        }
        std::ostringstream message;
        message << "Unknown module kind '" << kind << "' in " << context;
        throw std::runtime_error(message.str());
    }

    inline NamedModuleDescriptor deserialize_named_module_descriptor(const PropertyTree& tree, const std::string& context)
    {
        NamedModuleDescriptor descriptor{};
        descriptor.descriptor = deserialize_module_descriptor(tree, context);
        if (const auto name_value = tree.get_optional<std::string>("name")) {
            descriptor.name = *name_value;
        }
        return descriptor;
    }

    inline PropertyTree serialize_module_list(const std::vector<NamedModuleDescriptor>& descriptors)
    {
        PropertyTree tree;
        for (const auto& descriptor : descriptors) {
            tree.push_back({"", serialize_module_descriptor(descriptor)});
        }
        return tree;
    }

    inline std::vector<NamedModuleDescriptor> deserialize_module_list(const PropertyTree& tree, const std::string& context)
    {
        std::vector<NamedModuleDescriptor> descriptors;
        descriptors.reserve(tree.size());
        for (const auto& node : tree) {
            descriptors.push_back(deserialize_named_module_descriptor(node.second, context));
        }
        return descriptors;
    }

    inline void write_json_file(const std::filesystem::path& path, const PropertyTree& tree)
    {
        std::ofstream stream(path);
        if (!stream) {
            std::ostringstream message;
            message << "Failed to open '" << path.string() << "' for writing.";
            throw std::runtime_error(message.str());
        }
        boost::property_tree::write_json(stream, tree, true);
    }

    inline PropertyTree read_json_file(const std::filesystem::path& path)
    {
        PropertyTree tree;
        boost::property_tree::read_json(path.string(), tree);
        return tree;
    }
}
#endif // Nott_COMMON_SAVE_LOAD_HPP
