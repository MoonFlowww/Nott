#ifndef Nott_ATTENTION_HPP
#define Nott_ATTENTION_HPP
/// This file is an factory, must exempt it from any logical-code. For functions look into "/details"

#include <cstdint>
#include <variant>

namespace Nott::Attention {
    enum class Variant {
        Full,
        Causal,
    };

    struct MultiHeadOptions {
        std::int64_t embed_dim{};
        std::int64_t num_heads{1};
        double dropout{0.0};
        bool bias{true};
        bool add_bias_kv{false};
        bool add_zero_attn{false};
        bool batch_first{true};
        Variant variant{Variant::Full};
    };

    struct MultiHeadDescriptor {
        MultiHeadOptions options{};
    };

    struct MultiHeadLatentOptions {
        std::int64_t embed_dim{};
        std::int64_t num_heads{1};
        std::int64_t latent_dim{128};
        double dropout{0.0};
        bool bias{true};
        bool batch_first{true};
        Variant variant{Variant::Full};
    };

    struct MultiHeadLatentDescriptor {
        MultiHeadLatentOptions options{};
    };

    using Descriptor = std::variant<MultiHeadDescriptor, MultiHeadLatentDescriptor>;

    [[nodiscard]] inline auto MultiHead(const MultiHeadOptions& options) -> Descriptor {
        return Descriptor{MultiHeadDescriptor{options}};
    }
    [[nodiscard]] inline auto MultiHeadLatent(const MultiHeadLatentOptions& options) -> Descriptor {
        return Descriptor{MultiHeadLatentDescriptor{options}};
    }
}

#endif //Nott_ATTENTION_HPP