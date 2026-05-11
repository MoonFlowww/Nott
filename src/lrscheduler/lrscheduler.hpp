#ifndef Nott_LRSCHEDULER_HPP
#define Nott_LRSCHEDULER_HPP
/// This file is an factory, must exempt it from any logical-code. For functions look into "/details"
#include <variant>

#include "details/cosineannealing.hpp"
#include "registry.hpp"

namespace Nott::LrScheduler {
    using CosineAnnealingOptions = Details::CosineAnnealingOptions;
    using CosineAnnealingDescriptor = Details::CosineAnnealingDescriptor;

    using Descriptor = std::variant<CosineAnnealingDescriptor>;

    [[nodiscard]] constexpr auto CosineAnnealing(const CosineAnnealingOptions& options = {}) noexcept
        -> CosineAnnealingDescriptor {
        return {options};
    }
}


#endif //Nott_LRSCHEDULER_HPP