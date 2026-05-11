#ifndef Nott_ACTIVATION_HPP
#define Nott_ACTIVATION_HPP
/// This file is a factory, must exempt it from any logical-code. For functions look into "/details"

namespace Nott::Activation {
    enum class Type {
        Identity,
        ReLU,
        Sigmoid,
        Tanh,
        LeakyReLU,
        Softmax,
        SiLU,
        GeLU,
        GLU,
        SwiGLU,
        dSiLU,
        PSiLU,
        Mish,
        Swish,
    };

    struct Descriptor {
        Type type{Type::Identity};
    };

    inline constexpr Descriptor Identity{Type::Identity};
    inline constexpr Descriptor ReLU{Type::ReLU};
    inline constexpr Descriptor Sigmoid{Type::Sigmoid};
    inline constexpr Descriptor Tanh{Type::Tanh};
    inline constexpr Descriptor LeakyReLU{Type::LeakyReLU};
    inline constexpr Descriptor Softmax{Type::Softmax};
    inline constexpr Descriptor SiLU{Type::SiLU};
    inline constexpr Descriptor GeLU{Type::GeLU};
    inline constexpr Descriptor GLU{Type::GLU};
    inline constexpr Descriptor SwiGLU{Type::SwiGLU};
    inline constexpr Descriptor dSiLU{Type::dSiLU};
    inline constexpr Descriptor PSiLU{Type::PSiLU};
    inline constexpr Descriptor Mish{Type::Mish};
    inline constexpr Descriptor Swish{Type::Swish};
}

#endif //Nott_ACTIVATION_HPP
