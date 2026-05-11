#ifndef Nott_CONV_HPP
#define Nott_CONV_HPP
/// https://hal.science/hal-05083427/document
#include <cstdint>
#include <string>
#include <vector>
#include <algorithm>
#include <cctype>
#include <type_traits>

#include "../../activation/activation.hpp"
#include "../../initialization/initialization.hpp"
#include "../../common/local.hpp"
#include "../registry.hpp"

namespace Nott::Layer::Details {


    struct Conv1dOptions {
        std::int64_t in_channels{};
        std::int64_t out_channels{};
        std::vector<std::int64_t> kernel_size{3};
        std::vector<std::int64_t> stride{};
        std::vector<std::int64_t> padding{0};
        std::vector<std::int64_t> dilation{1};
        std::int64_t groups{1};
        bool bias{true};
        std::string padding_mode{"zeros"};
    };

    struct Conv1dDescriptor {
        Conv1dOptions options{};
        ::Nott::Activation::Descriptor activation{::Nott::Activation::Identity};
        ::Nott::Initialization::Descriptor initialization{::Nott::Initialization::Default};
        ::Nott::LocalConfig local{};
    };


    struct Conv2dOptions {
        std::int64_t in_channels{};
        std::int64_t out_channels{};
        std::vector<std::int64_t> kernel_size{3, 3};
        std::vector<std::int64_t> stride{1, 1};
        std::vector<std::int64_t> padding{0, 0};
        std::vector<std::int64_t> dilation{1, 1};
        std::int64_t groups{1};
        bool bias{true};
        std::string padding_mode{"zeros"};
    };

    struct Conv2dDescriptor {
        Conv2dOptions options{};
        ::Nott::Activation::Descriptor activation{::Nott::Activation::Identity};
        ::Nott::Initialization::Descriptor initialization{::Nott::Initialization::Default};
        ::Nott::LocalConfig local{};
    };

    template <typename T, typename = void>
    struct has_preferred_tensor_memory_format : std::false_type {};

    template <typename T>
    struct has_preferred_tensor_memory_format<T, std::void_t<decltype(std::declval<T&>().preferred_tensor_memory_format())>>
        : std::true_type {};

    template <typename T>
    inline constexpr bool has_preferred_tensor_memory_format_v = has_preferred_tensor_memory_format<T>::value;

    template <typename Module>
    void apply_memory_format(Module& module, torch::MemoryFormat format)
    {
        for (auto& parameter : module->parameters()) {
            auto contiguous = parameter.detach().contiguous(format);
            contiguous.set_requires_grad(parameter.requires_grad());
            parameter.set_data(contiguous);
        }

        for (auto& buffer : module->buffers()) {
            auto contiguous = buffer.detach().contiguous(format);
            buffer.set_data(contiguous);
        }
    }

    template <class Owner>
    RegisteredLayer build_registered_layer(Owner& owner, const Conv1dDescriptor& descriptor, std::size_t index)
    {
        if (descriptor.options.in_channels <= 0 || descriptor.options.out_channels <= 0) {
            throw std::invalid_argument("Conv1d layers require positive channel counts.");
        }

        auto options = torch::nn::Conv1dOptions(descriptor.options.in_channels,
                                                descriptor.options.out_channels,
                                                descriptor.options.kernel_size);
        if (!descriptor.options.stride.empty()) {
            options.stride(descriptor.options.stride);
        }
        if (!descriptor.options.padding.empty()) {
            options.padding(descriptor.options.padding);
        }
        if (!descriptor.options.dilation.empty()) {
            options.dilation(descriptor.options.dilation);
        }
        options.groups(descriptor.options.groups);
        options.bias(descriptor.options.bias);
        if (!descriptor.options.padding_mode.empty()) {
            auto padding_mode = descriptor.options.padding_mode;
            std::transform(padding_mode.begin(), padding_mode.end(), padding_mode.begin(),
                           [](unsigned char character) { return static_cast<char>(std::tolower(character)); });

            if (padding_mode == "zeros") {
                options.padding_mode(torch::kZeros);
            } else if (padding_mode == "reflect") {
                options.padding_mode(torch::kReflect);
            } else if (padding_mode == "replicate") {
                options.padding_mode(torch::kReplicate);
            } else if (padding_mode == "circular") {
                options.padding_mode(torch::kCircular);
            } else {
                throw std::invalid_argument("Unsupported padding mode provided to Conv1d descriptor: " +
                                            descriptor.options.padding_mode);
            }
        }

        auto module = owner.register_module("conv1d_" + std::to_string(index), torch::nn::Conv1d(options));
        if constexpr (has_preferred_tensor_memory_format_v<Owner>) {
            if (owner.preferred_tensor_memory_format() == torch::MemoryFormat::ChannelsLast) {
                apply_memory_format(module, torch::MemoryFormat::ChannelsLast);
            }
        }
        ::Nott::Initialization::Details::apply_module_initialization(module, descriptor);

        RegisteredLayer registered_layer{};
        registered_layer.activation = descriptor.activation.type;
        registered_layer.module = to_shared_module_ptr(module);
        registered_layer.local = descriptor.local;
        registered_layer.bind_module_forward(module.get());
        return registered_layer;
    }

    template <class Owner>
    RegisteredLayer build_registered_layer(Owner& owner, const Conv2dDescriptor& descriptor, std::size_t index)
    {
        if (descriptor.options.in_channels <= 0 || descriptor.options.out_channels <= 0) {
            throw std::invalid_argument("Conv2d layers require positive channel counts.");
        }

        auto options = torch::nn::Conv2dOptions(descriptor.options.in_channels,
                                                descriptor.options.out_channels,
                                                descriptor.options.kernel_size);
        if (!descriptor.options.stride.empty()) {
            options.stride(descriptor.options.stride);
        }
        if (!descriptor.options.padding.empty()) {
            options.padding(descriptor.options.padding);
        }
        if (!descriptor.options.dilation.empty()) {
            options.dilation(descriptor.options.dilation);
        }
        options.groups(descriptor.options.groups);
        options.bias(descriptor.options.bias);
        if (!descriptor.options.padding_mode.empty()) {
            auto padding_mode = descriptor.options.padding_mode;
            std::transform(padding_mode.begin(), padding_mode.end(), padding_mode.begin(), [](unsigned char character) { return static_cast<char>(std::tolower(character)); });

            if (padding_mode == "zeros") {
                options.padding_mode(torch::kZeros);
            } else if (padding_mode == "reflect") {
                options.padding_mode(torch::kReflect);
            } else if (padding_mode == "replicate") {
                options.padding_mode(torch::kReplicate);
            } else if (padding_mode == "circular") {
                options.padding_mode(torch::kCircular);
            } else {
                throw std::invalid_argument("Unsupported padding mode provided to Conv2d descriptor: " + descriptor.options.padding_mode);
            }
        }

        auto module = owner.register_module("conv2d_" + std::to_string(index), torch::nn::Conv2d(options));
        if constexpr (has_preferred_tensor_memory_format_v<Owner>) {
            if (owner.preferred_tensor_memory_format() == torch::MemoryFormat::ChannelsLast) {
                apply_memory_format(module, torch::MemoryFormat::ChannelsLast);
            }
        }
        ::Nott::Initialization::Details::apply_module_initialization(module, descriptor);

        RegisteredLayer registered_layer{};
        registered_layer.activation = descriptor.activation.type;
        registered_layer.module = to_shared_module_ptr(module);
        registered_layer.local = descriptor.local;
        registered_layer.bind_module_forward(module.get());
        return registered_layer;
    }

}

#endif //Nott_CONV_HPP