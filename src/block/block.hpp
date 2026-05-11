#ifndef Nott_BLOCK_HPP
#define Nott_BLOCK_HPP
/// This file is an factory, must exempt it from any logical-code. For functions look into "/details"
#include <cstddef>
#include <initializer_list>
#include <utility>
#include <variant>
#include <vector>

#include "../common/local.hpp"
#include "details/blocks/residual.hpp"
#include "details/blocks/sequential.hpp"
#include "details/transformers/classic.hpp"
#include "details/transformers/mamba.hpp"
#include "details/transformers/ebt.hpp"
#include "details/transformers/plusplus.hpp"
#include "details/transformers/atlas.hpp"
#include "details/transformers/titan.hpp"
#include "details/transformers/bert.hpp"
#include "details/transformers/vision.hpp"
#include "details/transformers/perceiver.hpp"
#include "details/transformers/longformer_xl.hpp"

namespace Nott::Block {
    namespace Details::Transformer::Bert {
        struct EncoderDescriptor;
    }

    using SequentialDescriptor = Details::SequentialDescriptor;
    using ResidualDescriptor = Details::ResidualDescriptor;

    namespace Transformer = Details::Transformer;

    using Descriptor = std::variant<SequentialDescriptor,
                                    ResidualDescriptor,
                                    Transformer::Classic::EncoderDescriptor,
                                    Transformer::Classic::DecoderDescriptor,
                                    Transformer::Mamba::EncoderDescriptor,
                                    Transformer::EBT::EncoderDescriptor,
                                    Transformer::EBT::DecoderDescriptor,
                                    Transformer::PlusPlus::EncoderDescriptor,
                                    Transformer::PlusPlus::DecoderDescriptor,
                                    Transformer::Bert::EncoderDescriptor,
                                    Transformer::Vision::EncoderDescriptor,
                                    Transformer::Perceiver::EncoderDescriptor,
                                    Transformer::LongformerXL::EncoderDescriptor>;

    [[nodiscard]] inline auto Sequential(std::initializer_list<::Nott::Layer::Descriptor> layers, ::Nott::LocalConfig local = {}) -> SequentialDescriptor {
        SequentialDescriptor descriptor{};
        descriptor.layers.assign(layers.begin(), layers.end());
        descriptor.local = std::move(local);
        return descriptor;
    }

    [[nodiscard]] inline auto Sequential(std::vector<::Nott::Layer::Descriptor> layers, ::Nott::LocalConfig local = {}) -> SequentialDescriptor {
        SequentialDescriptor descriptor{};
        descriptor.layers = std::move(layers);
        descriptor.local = std::move(local);
        return descriptor;
    }

    [[nodiscard]] inline auto Residual(std::initializer_list<::Nott::Layer::Descriptor> layers,
            std::size_t repeats = 1, Details::ResidualSkipOptions skip = {}, Details::ResidualOutputOptions output = {}, ::Nott::LocalConfig local = {}) -> ResidualDescriptor {
        ResidualDescriptor descriptor{};
        descriptor.layers.assign(layers.begin(), layers.end());
        descriptor.repeats = repeats;
        descriptor.skip = std::move(skip);
        descriptor.output = std::move(output);
        descriptor.local = std::move(local);
        return descriptor;
    }

    [[nodiscard]] inline auto Residual(std::vector<::Nott::Layer::Descriptor> layers,
            std::size_t repeats = 1, Details::ResidualSkipOptions skip = {}, Details::ResidualOutputOptions output = {}, ::Nott::LocalConfig local = {}) -> ResidualDescriptor {
        ResidualDescriptor descriptor{};
        descriptor.layers = std::move(layers);
        descriptor.repeats = repeats;
        descriptor.skip = std::move(skip);
        descriptor.output = std::move(output);
        descriptor.local = std::move(local);
        return descriptor;
    }
}

#endif //Nott_BLOCK_HPP