#ifndef Nott_LAYER_HPP
#define Nott_LAYER_HPP
/// This file is a factory, must exempt it from any logical-code. For functions look into "/details"
#include <variant>
#include <vector>
#include <utility>

#include "../common/local.hpp"
#include "details/batchnorm.hpp"
#include "details/conv.hpp"
#include "details/dropout.hpp"
#include "details/s4.hpp"
#include "details/fc.hpp"
#include "details/reduce.hpp"
#include "details/flatten.hpp"
#include "details/pooling.hpp"
#include "details/recurrent.hpp"
#include "details/patch_unembed.hpp"
#include "details/resizing.hpp"

#include "registry.hpp"

namespace Nott::Layer {
    using FCOptions = Details::FCOptions;
    using FCDescriptor = Details::FCDescriptor;


    using Conv1dOptions = Details::Conv1dOptions;
    using Conv1dDescriptor = Details::Conv1dDescriptor;

    using Conv2dOptions = Details::Conv2dOptions;
    using Conv2dDescriptor = Details::Conv2dDescriptor;

    using BatchNorm2dOptions = Details::BatchNorm2dOptions;
    using BatchNorm2dDescriptor = Details::BatchNorm2dDescriptor;


    using MaxPool1dOptions = Details::MaxPool1dOptions;
    using AvgPool1dOptions = Details::AvgPool1dOptions;
    using AdaptiveAvgPool1dOptions = Details::AdaptiveAvgPool1dOptions;
    using AdaptiveMaxPool1dOptions = Details::AdaptiveMaxPool1dOptions;
    using MaxPool2dOptions = Details::MaxPool2dOptions;
    using AvgPool2dOptions = Details::AvgPool2dOptions;
    using AdaptiveAvgPool2dOptions = Details::AdaptiveAvgPool2dOptions;
    using AdaptiveMaxPool2dOptions = Details::AdaptiveMaxPool2dOptions;
    using PoolingDescriptor = Details::PoolingDescriptor;

    using HardDropoutOptions = Details::HardDropoutOptions;
    using HardDropoutDescriptor = Details::HardDropoutDescriptor;
    using SoftDropoutOptions = Details::SoftDropoutOptions;
    using SoftDropoutDescriptor = Details::SoftDropoutDescriptor;

    using FlattenOptions = Details::FlattenOptions;
    using FlattenDescriptor = Details::FlattenDescriptor;

    using UpsampleOptions = Details::UpsampleOptions;
    using UpsampleDescriptor = Details::UpsampleDescriptor;

    using DownsampleOptions = Details::DownsampleOptions;
    using DownsampleDescriptor = Details::DownsampleDescriptor;

    using RNNOptions = Details::RNNOptions;
    using RNNDescriptor = Details::RNNDescriptor;

    using LSTMOptions = Details::LSTMOptions;
    using LSTMDescriptor = Details::LSTMDescriptor;

    using xLSTMOptions = Details::xLSTMOptions;
    using xLSTMDescriptor = Details::xLSTMDescriptor;

    using GRUOptions = Details::GRUOptions;
    using GRUDescriptor = Details::GRUDescriptor;



    using S4Options = Details::S4Options;
    using S4Descriptor = Details::S4Descriptor;

    /// NB: this is a layer
    using ReduceOp = Details::ReduceOp;
    using ReduceOptions = Details::ReduceOptions;
    using ReduceDescriptor = Details::ReduceDescriptor;


    using PatchUnembedOptions = Details::PatchUnembedOptions;
    using PatchUnembedDescriptor = Details::PatchUnembedDescriptor;


    using Descriptor = std::variant<FCDescriptor,
                                    Conv1dDescriptor,
                                    Conv2dDescriptor,
                                    BatchNorm2dDescriptor,
                                    PoolingDescriptor,
                                    HardDropoutDescriptor,
                                    SoftDropoutDescriptor,
                                    FlattenDescriptor,
                                    RNNDescriptor,
                                    LSTMDescriptor,
                                    xLSTMDescriptor,
                                    GRUDescriptor,
                                    S4Descriptor,
                                    ReduceDescriptor,
                                    PatchUnembedDescriptor,
                                    UpsampleDescriptor,
                                    DownsampleDescriptor>;

    [[nodiscard]] inline auto FC(const FCOptions& options,
                                 ::Nott::Activation::Descriptor activation = ::Nott::Activation::Identity,
                                 ::Nott::Initialization::Descriptor initialization = ::Nott::Initialization::Default, ::Nott::LocalConfig local = {}) -> FCDescriptor {
        return {options, activation, initialization, std::move(local)};
    }

    [[nodiscard]] inline auto Conv1d(const Conv1dOptions& options,
                                 ::Nott::Activation::Descriptor activation = ::Nott::Activation::Identity,
                                 ::Nott::Initialization::Descriptor initialization = ::Nott::Initialization::Default,
                                 ::Nott::LocalConfig local = {}) -> Conv1dDescriptor{
        return {options, activation, initialization, std::move(local)};
    }

    [[nodiscard]] inline auto Conv2d(const Conv2dOptions& options, ::Nott::Activation::Descriptor activation = ::Nott::Activation::Identity,
                                     ::Nott::Initialization::Descriptor initialization = ::Nott::Initialization::Default, ::Nott::LocalConfig local = {}) -> Conv2dDescriptor {
        return {options, activation, initialization, std::move(local)};
    }

    [[nodiscard]] inline auto BatchNorm2d(const BatchNorm2dOptions& options,
                                              ::Nott::Activation::Descriptor activation = ::Nott::Activation::Identity,
                                              ::Nott::Initialization::Descriptor initialization = ::Nott::Initialization::Default, ::Nott::LocalConfig local = {}) -> BatchNorm2dDescriptor {
        return {options, activation, initialization, std::move(local)};
    }

    [[nodiscard]] inline auto MaxPool1d(const MaxPool1dOptions& options,
                                        ::Nott::Activation::Descriptor activation = ::Nott::Activation::Identity,
                                        ::Nott::LocalConfig local = {}) -> PoolingDescriptor {
        PoolingDescriptor descriptor{};
        descriptor.options = options;
        descriptor.activation = activation;
        descriptor.local = std::move(local);
        return descriptor;
    }

    [[nodiscard]] inline auto AvgPool1d(const AvgPool1dOptions& options,
                                        ::Nott::Activation::Descriptor activation = ::Nott::Activation::Identity,
                                        ::Nott::LocalConfig local = {}) -> PoolingDescriptor {
        PoolingDescriptor descriptor{};
        descriptor.options = options;
        descriptor.activation = activation;
        descriptor.local = std::move(local);
        return descriptor;
    }

    [[nodiscard]] inline auto AdaptiveAvgPool1d(const AdaptiveAvgPool1dOptions& options,
                                                ::Nott::Activation::Descriptor activation = ::Nott::Activation::Identity,
                                                ::Nott::LocalConfig local = {}) -> PoolingDescriptor {
        PoolingDescriptor descriptor{};
        descriptor.options = options;
        descriptor.activation = activation;
        descriptor.local = std::move(local);
        return descriptor;
    }

    [[nodiscard]] inline auto AdaptiveMaxPool1d(const AdaptiveMaxPool1dOptions& options,
                                                ::Nott::Activation::Descriptor activation = ::Nott::Activation::Identity,
                                                ::Nott::LocalConfig local = {}) -> PoolingDescriptor {
        PoolingDescriptor descriptor{};
        descriptor.options = options;
        descriptor.activation = activation;
        descriptor.local = std::move(local);
        return descriptor;
    }

    [[nodiscard]] inline auto MaxPool2d(const MaxPool2dOptions& options,
                                        ::Nott::Activation::Descriptor activation = ::Nott::Activation::Identity,  ::Nott::LocalConfig local = {}) -> PoolingDescriptor {
        PoolingDescriptor descriptor{};
        descriptor.options = options;
        descriptor.activation = activation;
        descriptor.local = std::move(local);
        return descriptor;
    }

    [[nodiscard]] inline auto AvgPool2d(const AvgPool2dOptions& options,
                                        ::Nott::Activation::Descriptor activation = ::Nott::Activation::Identity,  ::Nott::LocalConfig local = {}) -> PoolingDescriptor {
        PoolingDescriptor descriptor{};
        descriptor.options = options;
        descriptor.activation = activation;
        descriptor.local = std::move(local);
        return descriptor;
    }

    [[nodiscard]] inline auto AdaptiveAvgPool2d(const AdaptiveAvgPool2dOptions& options,
                                                ::Nott::Activation::Descriptor activation = ::Nott::Activation::Identity,  ::Nott::LocalConfig local = {}) -> PoolingDescriptor {
        PoolingDescriptor descriptor{};
        descriptor.options = options;
        descriptor.activation = activation;
        descriptor.local = std::move(local);
        return descriptor;
    }

    [[nodiscard]] inline auto AdaptiveMaxPool2d(const AdaptiveMaxPool2dOptions& options,
                                                ::Nott::Activation::Descriptor activation = ::Nott::Activation::Identity,  ::Nott::LocalConfig local = {}) -> PoolingDescriptor {
        PoolingDescriptor descriptor{};
        descriptor.options = options;
        descriptor.activation = activation;
        descriptor.local = std::move(local);
        return descriptor;
    }

    [[nodiscard]] inline auto HardDropout(const HardDropoutOptions& options, ::Nott::Activation::Descriptor activation = ::Nott::Activation::Identity,  ::Nott::LocalConfig local = {}) -> HardDropoutDescriptor {
        return {options, activation, std::move(local)};
    }

    [[nodiscard]] inline auto SoftDropout(const SoftDropoutOptions& options, ::Nott::Activation::Descriptor activation = ::Nott::Activation::Identity, ::Nott::LocalConfig local = {}) -> SoftDropoutDescriptor {
        return {options, activation, std::move(local)};
    }

    [[nodiscard]] inline auto Flatten(const FlattenOptions& options = {},
                                      ::Nott::Activation::Descriptor activation = ::Nott::Activation::Identity,  ::Nott::LocalConfig local = {}) -> FlattenDescriptor {
        return {options, activation, std::move(local)};
    }

    [[nodiscard]] inline auto Upsample(const UpsampleOptions& options = {}, ::Nott::Activation::Descriptor activation = ::Nott::Activation::Identity, ::Nott::LocalConfig local = {}) -> UpsampleDescriptor {
        return {options, activation, std::move(local)};
    }

    [[nodiscard]] inline auto Downsample(const DownsampleOptions& options = {}, ::Nott::Activation::Descriptor activation = ::Nott::Activation::Identity, ::Nott::LocalConfig local = {}) -> DownsampleDescriptor {
        return {options, activation, std::move(local)};
    }

    [[nodiscard]] inline auto RNN(const RNNOptions& options,
                                   ::Nott::Activation::Descriptor activation = ::Nott::Activation::Identity,
                                   ::Nott::Initialization::Descriptor initialization = ::Nott::Initialization::Default,
                                   ::Nott::LocalConfig local = {}) -> RNNDescriptor {
        return {options, activation, initialization, std::move(local)};
    }

    [[nodiscard]] inline auto LSTM(const LSTMOptions& options,
                                    ::Nott::Activation::Descriptor activation = ::Nott::Activation::Identity,
                                    ::Nott::Initialization::Descriptor initialization = ::Nott::Initialization::Default,
                                    ::Nott::LocalConfig local = {}) -> LSTMDescriptor {
        return {options, activation, initialization, std::move(local)};
    }
    [[nodiscard]] inline auto xLSTM(const xLSTMOptions& options,
                                    ::Nott::Activation::Descriptor activation = ::Nott::Activation::Identity,
                                    ::Nott::Initialization::Descriptor initialization = ::Nott::Initialization::Default,
                                    ::Nott::LocalConfig local = {}) -> xLSTMDescriptor {
        return {options, activation, initialization, std::move(local)};
    }


    [[nodiscard]] inline auto GRU(const GRUOptions& options,
                                   ::Nott::Activation::Descriptor activation = ::Nott::Activation::Identity,
                                   ::Nott::Initialization::Descriptor initialization = ::Nott::Initialization::Default,
                                   ::Nott::LocalConfig local = {}) -> GRUDescriptor {
        return {options, activation, initialization, std::move(local)};
    }



    [[nodiscard]] inline auto S4(const S4Options& options,
                              ::Nott::Activation::Descriptor activation = ::Nott::Activation::Identity,
                              ::Nott::Initialization::Descriptor initialization = ::Nott::Initialization::Default,
                              ::Nott::LocalConfig local = {}) -> S4Descriptor {
        return {options, activation, initialization, std::move(local)};
    }


    [[nodiscard]] inline auto Reduce(const ReduceOptions& options = {},
                                 ::Nott::Activation::Descriptor activation = ::Nott::Activation::Identity,
                                 ::Nott::LocalConfig local = {}) -> ReduceDescriptor {
        return {options, activation, std::move(local)};
    }

    [[nodiscard]] inline auto PatchUnembed(const PatchUnembedOptions& options = {},
                                 ::Nott::Activation::Descriptor activation = ::Nott::Activation::Identity,
                                 ::Nott::LocalConfig local = {}) -> PatchUnembedDescriptor {
        return {options, activation, std::move(local)};
    }


}

#endif //Nott_LAYER_HPP
