#ifndef Nott_INITIALIZATION_HPP
#define Nott_INITIALIZATION_HPP
/// This file is a factory, must exempt it from any logical-code. For functions look into "/details"

namespace Nott::Initialization {
    enum class Type {
        Default,
        XavierNormal,
        XavierUniform,
        HeNormal,
        HeUniform,
        ZeroBias,
        Dirac, // https://arxiv.org/pdf/1706.00388
        Lyapunov,
    };

    struct Descriptor {
        Type type{Type::Default};
    };

    inline constexpr Descriptor Default{Type::Default};
    inline constexpr Descriptor XavierNormal{Type::XavierNormal};
    inline constexpr Descriptor XavierUniform{Type::XavierUniform};
    inline constexpr Descriptor HeNormal{Type::HeNormal};
    inline constexpr Descriptor HeUniform{Type::HeUniform};
    inline constexpr Descriptor ZeroBias{Type::ZeroBias};
    inline constexpr Descriptor Dirac{Type::Dirac};
    inline constexpr Descriptor Lyapunov{Type::Lyapunov};
}

#endif //Nott_INITIALIZATION_HPP
