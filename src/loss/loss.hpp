#ifndef Nott_LOSS_HPP
#define Nott_LOSS_HPP
/// This file is a factory, must exempt it from any logical-code. For functions look into "/details"
#include <variant>



#include "details/reduction.hpp"
#include "details/ce.hpp"
#include "details/mse.hpp"
#include "details/bce.hpp"
#include "details/nll.hpp"
#include "details/mae.hpp"
#include "details/smooth_l1.hpp"
#include "details/kl.hpp"
#include "details/cosine_embedding.hpp"
#include "details/margin_ranking.hpp"
#include "details/dice.hpp"
#include "details/tversky.hpp"
#include "details/lovasz_softmax.hpp"

namespace Nott::Loss {
    using Reduction = Details::Reduction;

    using Descriptor = std::variant<
        Details::MSEDescriptor,
        Details::CrossEntropyDescriptor,
        Details::BCEWithLogitsDescriptor,
        Details::CosineEmbeddingDescriptor,
        Details::KLDivDescriptor,
        Details::MAEDescriptor,
        Details::MarginRankingDescriptor,
        Details::NegativeLogLikelihoodDescriptor,
        Details::SmoothL1Descriptor,
        Details::DiceDescriptor,
        Details::TverskyDescriptor,
        Details::LovaszSoftmaxDescriptor>;


    [[nodiscard]] constexpr auto MSE(const Details::MSEOptions& options = {}) noexcept -> Details::MSEDescriptor {
        return {options};
    }

    [[nodiscard]] constexpr auto CrossEntropy(const Details::CrossEntropyOptions& options = {}) noexcept -> Details::CrossEntropyDescriptor {
        return {options};
    }


    [[nodiscard]] constexpr auto BCEWithLogits(const Details::BCEWithLogitsOptions& options = {}) noexcept -> Details::BCEWithLogitsDescriptor {
        return {options};
    }

    [[nodiscard]] constexpr auto NegativeLogLikelihood(const Details::NegativeLogLikelihoodOptions& options = {}) noexcept -> Details::NegativeLogLikelihoodDescriptor {
        return {options};
    }

    [[nodiscard]] constexpr auto MAE(const Details::MAEOptions& options = {}) noexcept -> Details::MAEDescriptor {
        return {options};
    }

    [[nodiscard]] constexpr auto SmoothL1(const Details::SmoothL1Options& options = {}) noexcept -> Details::SmoothL1Descriptor {
        return {options};
    }

    [[nodiscard]] constexpr auto KLDiv(const Details::KLDivOptions& options = {}) noexcept -> Details::KLDivDescriptor {
        return {options};
    }

    [[nodiscard]] constexpr auto CosineEmbedding(const Details::CosineEmbeddingOptions& options = {}) noexcept -> Details::CosineEmbeddingDescriptor {
        return {options};
    }

    [[nodiscard]] constexpr auto MarginRanking(const Details::MarginRankingOptions& options = {}) noexcept -> Details::MarginRankingDescriptor {
        return {options};
    }
    [[nodiscard]] constexpr auto Dice(const Details::DiceOptions& options = {}) noexcept -> Details::DiceDescriptor {
        return {options};
    }

    [[nodiscard]] constexpr auto Tversky(const Details::TverskyOptions& options = {}) noexcept -> Details::TverskyDescriptor {
        return {options};
    }

    [[nodiscard]] constexpr auto LovaszSoftmax(const Details::LovaszSoftmaxOptions& options = {}) noexcept -> Details::LovaszSoftmaxDescriptor {
        return {options};
    }
}

#endif //Nott_LOSS_HPP
