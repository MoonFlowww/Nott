#ifndef Nott_OPTIMIZER_HPP
#define Nott_OPTIMIZER_HPP
#include <variant>

#include "registry.hpp"


#include "details/adam.hpp"
#include "details/sgd.hpp"
#include "details/sophia.hpp"
#include "details/muon.hpp"
#include "details/adagrad.hpp"
#include "details/adafactor.hpp"
#include "details/lamb.hpp"
#include "details/lion.hpp"
#include "details/rmsprop.hpp"


namespace Nott::Optimizer {
    using SGDOptions = Details::SGDOptions;
    using SGDDescriptor = Details::SGDDescriptor;

    using RMSpropOptions = Details::RMSpropOptions;
    using RMSpropDescriptor = Details::RMSpropDescriptor;

    using AdagradOptions = Details::AdagradOptions;
    using AdagradDescriptor = Details::AdagradDescriptor;

    using AdamOptions = Details::AdamOptions;
    using AdamDescriptor = Details::AdamDescriptor;

    using AdamWOptions = Details::AdamWOptions;
    using AdamWDescriptor = Details::AdamWDescriptor;

    using LionOptions = Details::LionOptions;
    using LionDescriptor = Details::LionDescriptor;

    using LAMBOptions = Details::LAMBOptions;
    using LAMBDescriptor = Details::LAMBDescriptor;

    using AdafactorOptions = Details::AdafactorOptions;
    using AdafactorDescriptor = Details::AdafactorDescriptor;

    /// Sophia
    using SophiaGOptions = Details::SophiaGOptions;
    using SophiaGDescriptor = Details::SophiaGDescriptor;

    using SophiaHOptions = Details::SophiaHOptions;
    using SophiaHDescriptor = Details::SophiaHDescriptor;


    /// Muon
    using MuonOptions = Details::MuonOptions;
    using MuonDescriptor = Details::MuonDescriptor;

    using AdaMuonOptions = Details::AdaMuonOptions;
    using AdaMuonDescriptor = Details::AdaMuonDescriptor;

    using MuonManifoldOptions = Details::MuonManifoldOptions;
    using MuonManifoldDescriptor = Details::MuonManifoldDescriptor;


    using Descriptor = std::variant<SGDDescriptor,
                                    RMSpropDescriptor,
                                    AdamDescriptor,
                                    AdamWDescriptor,
                                    LionDescriptor,
                                    LAMBDescriptor,
                                    AdafactorDescriptor,
                                    SophiaGDescriptor,
                                    SophiaHDescriptor,
                                    MuonDescriptor,
                                    AdaMuonDescriptor,
                                    MuonManifoldDescriptor,
                                    AdagradDescriptor>;



    [[nodiscard]] inline constexpr auto SGD(const SGDOptions& options = {}) noexcept -> SGDDescriptor {
        return SGDDescriptor{.options = options};
    }


    [[nodiscard]] constexpr auto RMSprop(const RMSpropOptions& options = {}) noexcept -> RMSpropDescriptor {
        return RMSpropDescriptor{.options = options};
    }

    [[nodiscard]] constexpr auto Adagrad(const AdagradOptions& options = {}) noexcept -> AdagradDescriptor {
        return AdagradDescriptor{.options = options};
    }

    [[nodiscard]] inline constexpr auto AdamW(const AdamWOptions& options = {}) noexcept -> AdamWDescriptor {
        return AdamWDescriptor{.options = options};
    }

    [[nodiscard]] constexpr auto Adam(const AdamOptions& options = {}) noexcept -> AdamDescriptor {
        return AdamDescriptor{.options = options};
    }

    [[nodiscard]] inline auto Lion(const LionOptions& options = {}) noexcept -> LionDescriptor {
        return LionDescriptor{.options = options};
    }

    [[nodiscard]] inline auto LAMB(const LAMBOptions& options = {}) noexcept -> LAMBDescriptor {
        return LAMBDescriptor{.options = options};
    }

    [[nodiscard]] inline auto Adafactor(const AdafactorOptions& options = {}) noexcept -> AdafactorDescriptor {
        return AdafactorDescriptor{.options = options};
    }

    [[nodiscard]] inline auto SophiaG(const SophiaGOptions& options = {}) noexcept -> SophiaGDescriptor {
        return SophiaGDescriptor{.options = options};
    }

    [[nodiscard]] inline auto SophiaH(const SophiaHOptions& options = {}) noexcept -> SophiaHDescriptor {
        return SophiaHDescriptor{.options = options};
    }

    [[nodiscard]] inline auto Muon(const MuonOptions& options = {}) noexcept -> MuonDescriptor {
        return MuonDescriptor{.options=options};
    }

    [[nodiscard]] inline auto AdaMuon(const AdaMuonOptions& options = {}) noexcept -> AdaMuonDescriptor {
        return AdaMuonDescriptor{.options=options};
    }

    [[nodiscard]] inline auto MuonManifold(const MuonManifoldOptions& options = {}) noexcept -> MuonManifoldDescriptor {
        return MuonManifoldDescriptor{.options=options};
    }

}

#endif //Nott_OPTIMIZER_HPP