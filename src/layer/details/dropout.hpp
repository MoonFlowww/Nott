#ifndef Nott_DROPOUT_HPP
#define Nott_DROPOUT_HPP
/// "Dropout: A Simple Way to Prevent Neural Networks from Overfitting" https://arxiv.org/pdf/1207.0580
#include "../../activation/activation.hpp"
#include "../../common/local.hpp"
#include <torch/torch.h>
#include <string>
#include <utility>
#include "../registry.hpp"
namespace Nott::Layer::Details {

    struct HardDropoutOptions {
        double probability{0.5};
        bool inplace{false};
    };

    struct HardDropoutDescriptor {
        HardDropoutOptions options{};
        ::Nott::Activation::Descriptor activation{::Nott::Activation::Identity};
        ::Nott::LocalConfig local{};
    };

    class HardDropoutImpl : public torch::nn::Module {
    public:
        explicit HardDropoutImpl(HardDropoutOptions options = {})
            : options_(options)
        {
            TORCH_CHECK(options_.probability >= 0.0 && options_.probability < 1.0,
                        "HardDropout probability must be in the range [0, 1).");
        }

        torch::Tensor forward(torch::Tensor input)
        {
            if (!input.defined()) return input;
            TORCH_CHECK(input.is_floating_point(), "SoftDropout expects floating point tensors.");
            if (!is_training() || options_.probability == 0.0) return input;

            const double p = options_.probability;
            const double std = std::sqrt(p / (1.0 - p)); // matches inverted dropout variance

            /// mean=1 keeps E[output]=input
            auto eps = torch::empty_like(input).normal_(1.0, std);
            auto output = input * eps;

            if (options_.inplace && !input.requires_grad()) {
                input.mul_(eps);
                return input;
            }
            return output;
        }


        [[nodiscard]] const HardDropoutOptions& options() const noexcept { return options_; }

    private:
        HardDropoutOptions options_{};
    };

    TORCH_MODULE(HardDropout);

    template <class Owner>
    RegisteredLayer build_registered_layer(Owner& owner, const HardDropoutDescriptor& descriptor, std::size_t index)
    {
        auto module = owner.register_module("hard_dropout_" + std::to_string(index), HardDropout(descriptor.options));

        RegisteredLayer registered_layer{};
        registered_layer.activation = descriptor.activation.type;
        registered_layer.module = to_shared_module_ptr(module);
        registered_layer.local = descriptor.local;
        registered_layer.bind_module_forward(module.get());
        return registered_layer;
    }



    struct SoftDropoutOptions {
        enum class NoiseType {
            Gaussian,
            Poisson,
            Dithering,
            InterleavedGradientNoise,
            BlueNoise,
            Bayer
        };
        double probability{0.5};
        double noise_mean{0.0};
        double noise_std{1.0};
        /// Controls which distribution is sampled to generate noise.
        NoiseType noise_type{NoiseType::Gaussian};
        bool inplace{false};
    };

    struct SoftDropoutDescriptor {
        SoftDropoutOptions options{};
        ::Nott::Activation::Descriptor activation{::Nott::Activation::Identity};
        ::Nott::LocalConfig local{};
    };

    class SoftDropoutImpl : public torch::nn::Module {
    public:
        explicit SoftDropoutImpl(SoftDropoutOptions options = {})
            : options_(options)
        {
            TORCH_CHECK(options_.probability >= 0.0 && options_.probability < 1.0,
                        "SoftDropout probability must be in the range [0, 1).");
            TORCH_CHECK(options_.noise_std >= 0.0,
                        "SoftDropout noise standard deviation must be non-negative.");
            if (options_.noise_type == SoftDropoutOptions::NoiseType::Poisson) {
                TORCH_CHECK(options_.noise_mean >= 0.0,
                            "SoftDropout Poisson noise requires a non-negative rate (noise_mean).");
            }
        }

        torch::Tensor forward(torch::Tensor input)
        {
            if (!input.defined()) {
                return input;
            }

            TORCH_CHECK(input.is_floating_point(),
                        "SoftDropout expects floating point tensors.");

            if (!is_training() || options_.probability == 0.0) {
                return input;
            }

            const double keep_prob = 1.0 - options_.probability;
            TORCH_CHECK(keep_prob > 0.0,
                        "SoftDropout probability must be strictly less than 1.");

            auto mask = torch::bernoulli(torch::full_like(input, keep_prob)).to(torch::kBool);
            auto scaled_input = input / keep_prob;
            torch::Tensor noise;
            switch (options_.noise_type) {
                case SoftDropoutOptions::NoiseType::Gaussian:
                    noise = torch::empty_like(input).normal_(options_.noise_mean, options_.noise_std);
                    break;
                case SoftDropoutOptions::NoiseType::Poisson: {
                    const double rate_value = options_.noise_mean;
                    TORCH_CHECK(rate_value >= 0.0,
                                "SoftDropout Poisson noise requires a non-negative rate (noise_mean).");
                    auto rate = torch::full_like(input, rate_value);
                    auto samples = torch::poisson(rate);
                    if (rate_value > 0.0) {
                        const auto poisson_std = std::sqrt(rate_value);
                        if (options_.noise_std > 0.0 && poisson_std > 0.0) {
                            samples = (samples - rate) / poisson_std;
                            noise = samples * options_.noise_std + options_.noise_mean;
                        } else {
                            noise = (samples - rate) + options_.noise_mean;
                        }
                    } else {
                        noise = torch::zeros_like(input) + options_.noise_mean;
                    }
                    break;
                }
                case SoftDropoutOptions::NoiseType::Dithering: {
                    const double low = options_.noise_mean - options_.noise_std;
                    const double high = options_.noise_mean + options_.noise_std;
                    noise = torch::empty_like(input).uniform_(low, high);
                    break;
                }
                case SoftDropoutOptions::NoiseType::InterleavedGradientNoise: {
                    const auto total = input.numel();
                    auto device = input.device();
                    auto indices = torch::arange(total, torch::TensorOptions().dtype(torch::kLong).device(device));
                    auto positive = torch::full({total}, 1.0, input.options());
                    auto negative = torch::full({total}, -1.0, input.options());
                    auto pattern = torch::where((indices.remainder(2) == 0), positive, negative);
                    noise = pattern.view_as(input) * options_.noise_std + options_.noise_mean;
                    break;
                }
                case SoftDropoutOptions::NoiseType::BlueNoise: {
                    auto raw_noise = torch::rand_like(input) - 0.5;
                    auto centered = raw_noise - raw_noise.mean();
                    auto std_tensor = centered.std();
                    const auto std_value = std_tensor.to(torch::kCPU).item<double>();
                    if (std_value > 0.0) {
                        centered = centered / std_value;
                    }
                    noise = centered * options_.noise_std + options_.noise_mean;
                    break;
                }
                case SoftDropoutOptions::NoiseType::Bayer: {
                    const auto total = input.numel();
                    auto device = input.device();
                    auto indices = torch::arange(total, torch::TensorOptions().dtype(torch::kLong).device(device));
                    auto base_pattern = torch::tensor({0.0, 2.0, 3.0, 1.0},
                                                       torch::TensorOptions().dtype(input.scalar_type()));
                    base_pattern = base_pattern.to(device);
                    auto tiled = base_pattern.index_select(0, indices.remainder(4)).view_as(input);
                    noise = (tiled / 3.0) * options_.noise_std + options_.noise_mean;
                    break;
                }
                default:
                    TORCH_CHECK(false, "Unsupported SoftDropout noise type encountered.");
            }
            auto noisy_input = scaled_input + noise;
            auto output = torch::where(mask, scaled_input, noisy_input);

            if (options_.inplace) {
                input.copy_(output);
                return input;
            }

            return output;
        }


        [[nodiscard]] const SoftDropoutOptions& options() const noexcept { return options_; }

    private:
        SoftDropoutOptions options_{};
    };

    TORCH_MODULE(SoftDropout);

    template <class Owner>
    RegisteredLayer build_registered_layer(Owner& owner, const SoftDropoutDescriptor& descriptor, std::size_t index)
    {
        auto module = owner.register_module("soft_dropout_" + std::to_string(index), SoftDropout(descriptor.options));

        RegisteredLayer registered_layer{};
        registered_layer.activation = descriptor.activation.type;
        registered_layer.module = to_shared_module_ptr(module);
        registered_layer.local = descriptor.local;
        registered_layer.bind_module_forward(module.get());
        return registered_layer;
    }

}

#endif //Nott_DROPOUT_HPP