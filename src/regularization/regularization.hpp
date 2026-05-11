#ifndef Nott_REGULARIZATION_HPP
#define Nott_REGULARIZATION_HPP
/// This file is an factory, must exempt it from any logical-code. For functions look into "/details"
#include <variant>

#include "details/centeringvariance.hpp"
#include "details/decov.hpp"
#include "details/elasticnet.hpp"
#include "details/ewc.hpp"
#include "details/fge.hpp"
#include "details/grouplasso.hpp"
#include "details/jacobian.hpp"
#include "details/klsparsity.hpp"
#include "details/l0hardconcrete.hpp"
#include "details/l1.hpp"
#include "details/l2.hpp"
#include "details/mas.hpp"
#include "details/maxnorm.hpp"
#include "details/nuclearnorm.hpp"
#include "details/orthogonality.hpp"
#include "details/r1.hpp"
#include "details/r2.hpp"
#include "details/sfge.hpp"
#include "details/swa.hpp"
#include "details/swag.hpp"
#include "details/spectralnorm.hpp"
#include "details/si.hpp"
#include "details/structuredl2.hpp"
#include "details/trades.hpp"
#include "details/vat.hpp"
#include "details/wgangp.hpp"

namespace Nott::Regularization {

    using L1Options = Details::L1Options;
    using L1Descriptor = Details::L1Descriptor;

    using ElasticNetOptions = Details::ElasticNetOptions;
    using ElasticNetDescriptor = Details::ElasticNetDescriptor;

    using GroupLassoOptions = Details::GroupLassoOptions;
    using GroupLassoDescriptor = Details::GroupLassoDescriptor;

    using StructuredL2Options = Details::StructuredL2Options;
    using StructuredL2Descriptor = Details::StructuredL2Descriptor;

    using L0HardConcreteOptions = Details::L0HardConcreteOptions;
    using L0HardConcreteDescriptor = Details::L0HardConcreteDescriptor;

    using OrthogonalityOptions = Details::OrthogonalityOptions;
    using OrthogonalityDescriptor = Details::OrthogonalityDescriptor;

    using SpectralNormOptions = Details::SpectralNormOptions;
    using SpectralNormDescriptor = Details::SpectralNormDescriptor;

    using MaxNormOptions = Details::MaxNormOptions;
    using MaxNormDescriptor = Details::MaxNormDescriptor;

    using KLSparsityOptions = Details::KLSparsityOptions;
    using KLSparsityDescriptor = Details::KLSparsityDescriptor;

    using DeCovOptions = Details::DeCovOptions;
    using DeCovDescriptor = Details::DeCovDescriptor;

    using CenteringVarianceOptions = Details::CenteringVarianceOptions;
    using CenteringVarianceDescriptor = Details::CenteringVarianceDescriptor;

    using JacobianNormOptions = Details::JacobianNormOptions;
    using JacobianNormDescriptor = Details::JacobianNormDescriptor;

    using WGANGPOptions = Details::WGANGPOptions;
    using WGANGPDescriptor = Details::WGANGPDescriptor;

    using R1Options = Details::R1Options;
    using R1Descriptor = Details::R1Descriptor;

    using R2Options = Details::R2Options;
    using R2Descriptor = Details::R2Descriptor;

    using TRADESOptions = Details::TRADESOptions;
    using TRADESDescriptor = Details::TRADESDescriptor;

    using VATOptions = Details::VATOptions;
    using VATDescriptor = Details::VATDescriptor;


    using L2Options = Details::L2Options;
    using L2Descriptor = Details::L2Descriptor;

    using EWCOptions = Details::EWCOptions;
    using EWCDescriptor = Details::EWCDescriptor;

    using MASOptions = Details::MASOptions;
    using MASDescriptor = Details::MASDescriptor;

    using SIOptions = Details::SIOptions;
    using SIDescriptor = Details::SIDescriptor;

    using NuclearNormOptions = Details::NuclearNormOptions;
    using NuclearNormDescriptor = Details::NuclearNormDescriptor;

    using SWAOptions = Details::SWAOptions;
    using SWADescriptor = Details::SWADescriptor;

    using SWAGOptions = Details::SWAGOptions;
    using SWAGDescriptor = Details::SWAGDescriptor;

    using FGEOptions = Details::FGEOptions;
    using FGEDescriptor = Details::FGEDescriptor;

    using SFGEOptions = Details::SFGEOptions;
    using SFGEDescriptor = Details::SFGEDescriptor;

    using Descriptor = std::variant<
        L1Descriptor,
        ElasticNetDescriptor,
        GroupLassoDescriptor,
        StructuredL2Descriptor,
        L0HardConcreteDescriptor,
        OrthogonalityDescriptor,
        SpectralNormDescriptor,
        MaxNormDescriptor,
        KLSparsityDescriptor,
        DeCovDescriptor,
        CenteringVarianceDescriptor,
        JacobianNormDescriptor,
        WGANGPDescriptor,
        R1Descriptor,
        R2Descriptor,
        TRADESDescriptor,
        VATDescriptor,
        L2Descriptor,
        EWCDescriptor,
        MASDescriptor,
        SIDescriptor,
        NuclearNormDescriptor,
        SWADescriptor,
        SWAGDescriptor,
        FGEDescriptor,
        SFGEDescriptor>;

    [[nodiscard]] constexpr auto L1(const L1Options& options = {}) noexcept -> L1Descriptor {
        return {options};
    }

    [[nodiscard]] constexpr auto ElasticNet(const ElasticNetOptions& options = {}) noexcept -> ElasticNetDescriptor {
        return {options};
    }

    [[nodiscard]] constexpr auto GroupLasso(const GroupLassoOptions& options = {}) noexcept -> GroupLassoDescriptor {
        return {options};
    }

    [[nodiscard]] constexpr auto StructuredL2(const StructuredL2Options& options = {}) noexcept -> StructuredL2Descriptor {
        return {options};
    }

    [[nodiscard]] constexpr auto L0HardConcrete(const L0HardConcreteOptions& options = {}) noexcept
        -> L0HardConcreteDescriptor
    {
        return {options};
    }

    [[nodiscard]] constexpr auto Orthogonality(const OrthogonalityOptions& options = {}) noexcept
        -> OrthogonalityDescriptor
    {
        return {options};
    }

    [[nodiscard]] constexpr auto SpectralNorm(const SpectralNormOptions& options = {}) noexcept
        -> SpectralNormDescriptor
    {
        return {options};
    }

    [[nodiscard]] constexpr auto MaxNorm(const MaxNormOptions& options = {}) noexcept -> MaxNormDescriptor
    {
        return {options};
    }

    [[nodiscard]] constexpr auto KLSparsity(const KLSparsityOptions& options = {}) noexcept -> KLSparsityDescriptor
    {
        return {options};
    }

    [[nodiscard]] constexpr auto DeCov(const DeCovOptions& options = {}) noexcept -> DeCovDescriptor
    {
        return {options};
    }

    [[nodiscard]] constexpr auto CenteringVariance(const CenteringVarianceOptions& options = {}) noexcept
        -> CenteringVarianceDescriptor
    {
        return {options};
    }


    [[nodiscard]] constexpr auto JacobianNorm(const JacobianNormOptions& options = {}) noexcept
        -> JacobianNormDescriptor
    {
        return {options};
    }

    [[nodiscard]] constexpr auto WGANGP(const WGANGPOptions& options = {}) noexcept -> WGANGPDescriptor
    {
        return {options};
    }

    [[nodiscard]] constexpr auto R1(const R1Options& options = {}) noexcept -> R1Descriptor { return {options}; }

    [[nodiscard]] constexpr auto R2(const R2Options& options = {}) noexcept -> R2Descriptor { return {options}; }

    [[nodiscard]] constexpr auto TRADES(const TRADESOptions& options = {}) noexcept -> TRADESDescriptor
    {
        return {options};
    }

    [[nodiscard]] constexpr auto VAT(const VATOptions& options = {}) noexcept -> VATDescriptor { return {options}; }

    [[nodiscard]] constexpr auto L2(const L2Options& options = {}) noexcept -> L2Descriptor {
        return {options};
    }

    [[nodiscard]] constexpr auto EWC(const EWCOptions& options = {}) noexcept -> EWCDescriptor {
        return {options};
    }

    [[nodiscard]] constexpr auto MAS(const MASOptions& options = {}) noexcept -> MASDescriptor {
        return {options};
    }

    [[nodiscard]] constexpr auto SI(const SIOptions& options = {}) noexcept -> SIDescriptor {
        return {options};
    }

    [[nodiscard]] constexpr auto NuclearNorm(const NuclearNormOptions& options = {}) noexcept
        -> NuclearNormDescriptor {
        return {options};
    }

    [[nodiscard]] constexpr auto SWA(const SWAOptions& options = {}) noexcept -> SWADescriptor {
        return {options};
    }

    [[nodiscard]] constexpr auto SWAG(const SWAGOptions& options = {}) noexcept -> SWAGDescriptor {
        return {options};
    }

    [[nodiscard]] constexpr auto FGE(const FGEOptions& options = {}) noexcept -> FGEDescriptor {
        return {options};
    }

    [[nodiscard]] constexpr auto SFGE(const SFGEOptions& options = {}) noexcept -> SFGEDescriptor {
        return {options};
    }

}

#endif //Nott_REGULARIZATION_HPP