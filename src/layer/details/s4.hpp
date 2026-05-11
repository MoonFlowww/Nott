#ifndef Nott_LAYER_S4_HPP
#define Nott_LAYER_S4_HPP
/// "Efficiently Modeling Long Sequences with Structured State Spaces" (S4) https://arxiv.org/pdf/2111.00396
#include <cmath>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <torch/torch.h>

#include "../../activation/activation.hpp"
#include "../../common/local.hpp"
#include "../../initialization/apply.hpp"
#include "../../initialization/initialization.hpp"
#include "../registry.hpp"

namespace Nott::Layer::Details {

    enum class S4Initialization {
        HiPPO,
        S4D,
    };

    struct S4Options {
        std::int64_t input_size{};
        std::int64_t state_size{};
        std::int64_t rank{1};
        std::int64_t output_size{0};
        bool batch_first{true};
        bool bidirectional{false};
        double dropout{0.0};
        S4Initialization initialization{S4Initialization::HiPPO};
        std::int64_t maximum_length{0};
    };

    struct S4Descriptor {
        S4Options options{};
        ::Nott::Activation::Descriptor activation{::Nott::Activation::Identity};
        ::Nott::Initialization::Descriptor initialization{::Nott::Initialization::Default};
        ::Nott::LocalConfig local{};
    };

    struct S4State {
        torch::Tensor value{}; // (directions, batch, output, state)
    };

    struct S4Output {
        torch::Tensor output{};
        S4State state{};
    };

    namespace detail {
        inline torch::Tensor hippo_legs_matrix(std::int64_t N, const torch::TensorOptions& options)
        {
            auto arange = torch::arange(N, options.dtype(torch::kFloat64));
            auto sqrt_values = torch::sqrt(2.0 * arange + 1.0);
            auto matrix = torch::zeros({N, N}, options.dtype(torch::kFloat64));

            for (std::int64_t n = 0; n < N; ++n) {
                for (std::int64_t k = 0; k < N; ++k) {
                    auto value = sqrt_values[n].item<double>() * sqrt_values[k].item<double>();
                    if (n > k) {
                        matrix.index_put_({n, k}, -value);
                    } else if (n == k) {
                        matrix.index_put_({n, k}, -(static_cast<double>(n) + 1.0));
                    } else {
                        matrix.index_put_({n, k}, value);
                    }
                }
            }

            return matrix.to(options.dtype(torch::kFloat64));
        }

        inline std::tuple<torch::Tensor, torch::Tensor> hippo_legs(std::int64_t N, const torch::TensorOptions& options)
        {
            auto A = hippo_legs_matrix(N, options);
            auto B = torch::sqrt(2.0 * torch::arange(N, options.dtype(torch::kFloat64)) + 1.0);
            return {A, B};
        }

        inline torch::Tensor default_s4d_eigenvalues(std::int64_t N, const torch::TensorOptions& options)
        {
            auto indices = torch::arange(N, options.dtype(torch::kFloat64));
            /// Mirror the S4D initialization: real negative spectrum spaced log
            auto base = torch::logspace(
                std::log10(0.1),
                std::log10(10.0),
                N,
                10.0,
                options.dtype(torch::kFloat64)
            );
            auto eigenvalues = -base;
            return eigenvalues.to(options.dtype(torch::kFloat64));
        }

        inline torch::Tensor complex_from_real_imag(const torch::Tensor& real, const torch::Tensor& imag)
        {
            return torch::complex(real, imag);
        }

        inline std::tuple<torch::Tensor, torch::Tensor> split_complex(const torch::Tensor& value)
        {
            return {torch::real(value), torch::imag(value)};
        }

        inline torch::Tensor ensure_complex(const torch::Tensor& tensor)
        {
            if (tensor.is_complex()) {
                return tensor;
            }
            return torch::complex(tensor, torch::zeros_like(tensor));
        }
    }

    class S4ModuleImpl : public torch::nn::Module {
    public:
        using Options = S4Options;

        explicit S4ModuleImpl(Options options)
            : options_(std::move(options))
        {
            if (options_.input_size <= 0) {
                throw std::invalid_argument("S4Module requires a positive input size.");
            }
            if (options_.state_size <= 0) {
                throw std::invalid_argument("S4Module requires a positive state size.");
            }
            if (options_.rank <= 0) {
                throw std::invalid_argument("S4Module requires a positive rank.");
            }
            if (options_.output_size <= 0) {
                options_.output_size = options_.input_size;
            }
            if (options_.dropout < 0.0 || options_.dropout >= 1.0) {
                throw std::invalid_argument("S4Module dropout must be in [0, 1).");
            }

            input_projection_ = register_module("input_projection", torch::nn::Linear(torch::nn::LinearOptions(options_.input_size, options_.rank).bias(false)));
            if (options_.dropout > 0.0) {
                dropout_ = register_module("dropout", torch::nn::Dropout(torch::nn::DropoutOptions(options_.dropout)));
            }

            initialize_parameters();
        }

        [[nodiscard]] const Options& options() const noexcept { return options_; }

        [[nodiscard]] torch::Tensor lambda() const
        {
            return detail::complex_from_real_imag(lambda_real_, lambda_imag_);
        }

        [[nodiscard]] torch::Tensor B_parameter() const
        {
            return detail::complex_from_real_imag(B_real_, B_imag_);
        }

        [[nodiscard]] torch::Tensor C_parameter() const
        {
            return detail::complex_from_real_imag(C_real_, C_imag_);
        }

        [[nodiscard]] torch::Tensor D_parameter() const
        {
            return D_;
        }

        [[nodiscard]] torch::Tensor log_dt_parameter() const
        {
            return log_dt_;
        }

        [[nodiscard]] torch::Tensor input_projection_weight() const
        {
            return input_projection_->weight;
        }

        [[nodiscard]] torch::Tensor discrete_lambda(torch::Tensor dt) const
        {
            auto lambda_complex = lambda();
            auto scaled = lambda_complex.unsqueeze(0) * dt.unsqueeze(-1);
            return torch::exp(scaled);
        }

        [[nodiscard]] torch::Tensor discretized_B(torch::Tensor dt) const
        {
            auto lambda_complex = lambda();
            auto scaled = lambda_complex.unsqueeze(0) * dt.unsqueeze(-1);
            auto exp_lambda = torch::exp(scaled);
            auto numerator = exp_lambda - 1.0;
            auto B = B_parameter();
            auto denom = lambda_complex.unsqueeze(0);
            auto discrete = numerator.unsqueeze(1) * B.unsqueeze(0) / denom.unsqueeze(0);
            return discrete;
        }

        [[nodiscard]] torch::Tensor compute_kernel(std::int64_t length)
        {
            if (length <= 0) {
                throw std::invalid_argument("S4Module kernel length must be positive.");
            }

            const bool reuse_cache = cached_kernel_.has_value() && cached_kernel_length_ == length && cached_kernel_device_ == lambda_real_.device() && !this->is_training();
            if (reuse_cache) {
                return cached_kernel_.value();
            }

            auto device = lambda_real_.device();
            auto dtype = lambda_real_.dtype();

            auto dt = torch::exp(log_dt_).to(device, dtype);
            auto discrete_lambda_values = discrete_lambda(dt).to(device);
            auto discrete_B = discretized_B(dt).to(device);
            auto C = C_parameter().to(device);

            auto log_discrete_lambda = torch::log(discrete_lambda_values);
            auto time = torch::arange(length, torch::TensorOptions().dtype(torch::kFloat32).device(device));
            auto time_broadcast = time.to(log_discrete_lambda.scalar_type());
            auto powers = torch::exp(log_discrete_lambda.unsqueeze(-1) * time_broadcast);

            auto kernel_complex = torch::einsum("oi,ori,oil->orl", std::vector<torch::Tensor>{C, discrete_B, powers});
            auto kernel = torch::real(kernel_complex);

            cached_kernel_ = kernel;
            cached_kernel_length_ = length;
            cached_kernel_device_ = device;

            return kernel;
        }

        [[nodiscard]] S4Output forward(torch::Tensor input, S4State state = {})
        {
            auto tensor = std::move(input);
            if (!tensor.defined()) {
                throw std::invalid_argument("S4Module requires a defined input tensor.");
            }
            if (tensor.dim() != 3) {
                throw std::invalid_argument("S4Module expects inputs shaped as (batch, seq, feature) or (seq, batch, feature).");
            }

            if (!options_.batch_first) {
                tensor = tensor.transpose(0, 1);
            }

            const auto batch = tensor.size(0);
            const auto sequence = tensor.size(1);

            auto projected = input_projection_->forward(tensor.reshape({batch * sequence, options_.input_size}));
            projected = projected.view({batch, sequence, options_.rank});

            auto kernel = compute_kernel(sequence);
            auto conv_output = apply_convolution(projected, kernel);

            auto residual = torch::matmul(tensor, D_.transpose(0, 1));
            auto output = conv_output + residual;

            if (dropout_ && this->is_training()) {
                output = dropout_->forward(output);
            }

            if (options_.bidirectional) {
                auto reversed = tensor.flip(1);
                auto projected_rev = input_projection_->forward(reversed.reshape({batch * sequence, options_.input_size}));
                projected_rev = projected_rev.view({batch, sequence, options_.rank});
                auto conv_rev = apply_convolution(projected_rev, kernel).flip(1);
                auto residual_rev = torch::matmul(reversed, D_.transpose(0, 1)).flip(1);
                auto combined = torch::cat({output, conv_rev + residual_rev}, -1);
                output = combined;
            }

            if (!options_.batch_first) {
                output = output.transpose(0, 1);
            }

            auto new_state = update_state(projected, state);

            return {output, std::move(new_state)};
        }

        torch::Tensor stream_step(const torch::Tensor& input, S4State& state, bool reverse = false)
        {
            auto tensor = input;
            if (tensor.dim() != 2) {
                throw std::invalid_argument("S4Module::stream_step expects inputs shaped as (batch, feature).");
            }

            auto dt = torch::exp(log_dt_).to(tensor.device(), tensor.dtype());
            auto discrete_lambda_values = discrete_lambda(dt).to(tensor.device());
            auto discrete_B = discretized_B(dt).to(tensor.device());
            auto C = C_parameter().to(tensor.device());

            auto projected = input_projection_->forward(tensor);
            auto updated_state = propagate_state(projected, state, discrete_lambda_values, discrete_B, reverse);

            state.value = updated_state.value;

            auto final_state = updated_state.value.index({0});
            auto complex_state = detail::complex_from_real_imag(final_state.select(-1, 0), final_state.select(-1, 1));
            auto emission = torch::einsum("oi,boi->bo", std::vector<torch::Tensor>{C, complex_state});
            auto residual = torch::matmul(tensor, D_.transpose(0, 1));
            auto output = torch::real(emission) + residual;

            if (reverse) {
                output = output.flip(1);
            }
            return output;
        }

        void apply_initialization(const ::Nott::Initialization::Descriptor& descriptor)
        {
            ::Nott::Initialization::Details::apply_module_initialization(input_projection_, descriptor);
        }

    private:
        void initialize_parameters()
        {
            auto options = torch::TensorOptions().dtype(torch::kFloat32);
            const auto N = options_.state_size;
            const auto rank = options_.rank;
            const auto output_size = options_.output_size;

            torch::Tensor lambda_values;
            torch::Tensor B_base;

            if (options_.initialization == S4Initialization::HiPPO) {
                auto [A, B] = detail::hippo_legs(N, options);
                auto eigen = torch::linalg_eig(A);
                auto eigenvalues = std::get<0>(eigen);
                auto eigenvectors = std::get<1>(eigen);
                auto V_inv = torch::linalg_inv(eigenvectors);
                auto B_transformed = torch::matmul(V_inv, B.unsqueeze(-1)).squeeze(-1);
                lambda_values = eigenvalues;
                B_base = B_transformed;
            } else {
                auto eigenvalues = detail::default_s4d_eigenvalues(N, options);
                lambda_values = detail::complex_from_real_imag(eigenvalues, torch::zeros_like(eigenvalues));
                B_base = torch::ones({N}, options.dtype(torch::kFloat32));
            }

            lambda_values = lambda_values.to(torch::kComplexFloat);
            auto [lambda_real, lambda_imag] = detail::split_complex(lambda_values);

            lambda_real_ = register_parameter("lambda_real", lambda_real);
            lambda_imag_ = register_parameter("lambda_imag", lambda_imag);

            auto B_expanded = B_base.to(torch::kComplexFloat).unsqueeze(0).repeat({rank, 1});
            auto [B_real, B_imag] = detail::split_complex(B_expanded);
            B_real_ = register_parameter("B_real", B_real);
            B_imag_ = register_parameter("B_imag", B_imag);

            auto C_real = torch::randn({output_size, N}, options.dtype(torch::kFloat32)) * std::sqrt(1.0 / static_cast<double>(N));
            auto C_imag = torch::zeros_like(C_real);
            C_real_ = register_parameter("C_real", C_real);
            C_imag_ = register_parameter("C_imag", C_imag);

            auto D = torch::eye(output_size, options_.input_size, options.dtype(torch::kFloat32));
            D_ = register_parameter("D", D);

            auto log_dt = torch::zeros({output_size}, options.dtype(torch::kFloat32));
            log_dt_ = register_parameter("log_dt", log_dt);
        }

        [[nodiscard]] torch::Tensor apply_convolution(const torch::Tensor& projected, const torch::Tensor& kernel)
        {
            auto batch = projected.size(0);
            auto length = projected.size(1);
            auto rank = projected.size(2);
            auto output_size = kernel.size(0);

            auto projected_permuted = projected.transpose(1, 2); // (batch, rank, length)
            auto flipped_kernel = kernel.flip(-1);
            auto conv_options = torch::nn::functional::Conv1dFuncOptions{}
                .stride(1)
                .padding(kernel.size(-1) - 1);
            auto convolved_full = torch::nn::functional::conv1d(
                projected_permuted,
                flipped_kernel,
                conv_options);
            auto start = kernel.size(-1) - 1;
            auto convolved = convolved_full.narrow(2, start, length);
            convolved = convolved.transpose(1, 2);
            return convolved;
        }

        [[nodiscard]] S4State update_state(const torch::Tensor& projected, const S4State& previous)
        {
            auto batch = projected.size(0);
            auto length = projected.size(1);
            auto rank = projected.size(2);
            auto directions = options_.bidirectional ? 2 : 1;
            auto device = projected.device();
            auto dtype = lambda_real_.dtype();

            auto dt = torch::exp(log_dt_).to(device, dtype);
            auto discrete_lambda_values = discrete_lambda(dt).to(device);
            auto discrete_B = discretized_B(dt).to(device);

            auto state_tensor = previous.value;
            if (!state_tensor.defined() || state_tensor.numel() == 0) {
                state_tensor = torch::zeros({directions, batch, options_.output_size, options_.state_size, 2}, projected.options().dtype(dtype));
            } else if (state_tensor.size(0) != directions || state_tensor.size(1) != batch) {
                state_tensor = torch::zeros({directions, batch, options_.output_size, options_.state_size, 2}, projected.options().dtype(dtype));
            }

            auto forward_state = state_tensor.index({0});
            auto forward_complex = detail::complex_from_real_imag(forward_state.select(-1, 0), forward_state.select(-1, 1));

            for (std::int64_t t = 0; t < length; ++t) {
                auto u_t = projected.index({torch::indexing::Slice(), t});
                auto contribution = torch::einsum("br,orn->bon", std::vector<torch::Tensor>{u_t, discrete_B});
                forward_complex = forward_complex * discrete_lambda_values.unsqueeze(0) + contribution;
            }

            auto [forward_real, forward_imag] = detail::split_complex(forward_complex);
            auto updated_state = torch::stack({forward_real, forward_imag}, -1);
            state_tensor.index_put_({0}, updated_state);

            if (options_.bidirectional) {
                auto backward_state = state_tensor.index({1});
                auto backward_complex = detail::complex_from_real_imag(backward_state.select(-1, 0), backward_state.select(-1, 1));

                for (std::int64_t t = length - 1; t >= 0; --t) {
                    auto u_t = projected.index({torch::indexing::Slice(), t});
                    auto contribution = torch::einsum("br,orn->bon", std::vector<torch::Tensor>{u_t, discrete_B});
                    backward_complex = backward_complex * discrete_lambda_values.unsqueeze(0) + contribution;
                }

                auto [back_real, back_imag] = detail::split_complex(backward_complex);
                state_tensor.index_put_({1}, torch::stack({back_real, back_imag}, -1));
            }

            return {state_tensor};
        }

        [[nodiscard]] S4State propagate_state(const torch::Tensor& projected,
                                              const S4State& previous,
                                              const torch::Tensor& discrete_lambda_values,
                                              const torch::Tensor& discrete_B,
                                              bool reverse)
        {
            auto batch = projected.size(0);
            auto directions = options_.bidirectional ? 2 : 1;
            auto dtype = lambda_real_.dtype();

            auto state_tensor = previous.value;
            if (!state_tensor.defined() || state_tensor.numel() == 0) {
                state_tensor = torch::zeros({directions, batch, options_.output_size, options_.state_size, 2}, projected.options().dtype(dtype));
            }

            auto direction_index = reverse && options_.bidirectional ? 1 : 0;
            auto direction_state = state_tensor.index({direction_index});
            auto direction_complex = detail::complex_from_real_imag(direction_state.select(-1, 0), direction_state.select(-1, 1));
            auto contribution = torch::einsum("br,orn->bon", std::vector<torch::Tensor>{projected, discrete_B});
            direction_complex = direction_complex * discrete_lambda_values.unsqueeze(0) + contribution;
            auto [real_part, imag_part] = detail::split_complex(direction_complex);
            state_tensor.index_put_({direction_index}, torch::stack({real_part, imag_part}, -1));

            if (options_.bidirectional && !reverse) {
                auto backward_state = state_tensor.index({1});
                auto backward_complex = detail::complex_from_real_imag(backward_state.select(-1, 0), backward_state.select(-1, 1));
                auto contribution_backward = torch::einsum("br,orn->bon", std::vector<torch::Tensor>{projected, discrete_B});
                backward_complex = backward_complex * discrete_lambda_values.unsqueeze(0) + contribution_backward;
                auto [back_real, back_imag] = detail::split_complex(backward_complex);
                state_tensor.index_put_({1}, torch::stack({back_real, back_imag}, -1));
            }

            return {state_tensor};
        }

    private:
        Options options_{};
        torch::nn::Linear input_projection_{nullptr};
        torch::nn::Dropout dropout_{nullptr};

        torch::Tensor lambda_real_{};
        torch::Tensor lambda_imag_{};
        torch::Tensor B_real_{};
        torch::Tensor B_imag_{};
        torch::Tensor C_real_{};
        torch::Tensor C_imag_{};
        torch::Tensor D_{};
        torch::Tensor log_dt_{};

        std::optional<torch::Tensor> cached_kernel_{};
        std::int64_t cached_kernel_length_{-1};
        torch::Device cached_kernel_device_{torch::kCPU};
    };

    TORCH_MODULE(S4Module);

    template <class Owner>
    RegisteredLayer build_registered_layer(Owner& owner, const S4Descriptor& descriptor, std::size_t index)
    {
        auto options = descriptor.options;
        if (options.input_size <= 0) {
            throw std::invalid_argument("S4 layers require a positive input size.");
        }
        if (options.state_size <= 0) {
            throw std::invalid_argument("S4 layers require a positive state size.");
        }
        if (options.rank <= 0) {
            throw std::invalid_argument("S4 layers require a positive rank.");
        }
        if (options.output_size <= 0) {
            options.output_size = options.input_size;
        }

        auto module = owner.register_module("s4_" + std::to_string(index), S4Module(options));
        module->apply_initialization(descriptor.initialization);

        RegisteredLayer registered_layer{};
        registered_layer.activation = descriptor.activation.type;
        registered_layer.module = to_shared_module_ptr(module);
        registered_layer.local = descriptor.local;
        struct S4Forward {
            decltype(module.get()) module_ptr;

            torch::Tensor operator()(torch::Tensor input) const
            {
                auto result = module_ptr->forward(std::move(input));
                return std::move(result.output);
            }
        };
        registered_layer.bind_inline_forward(S4Forward{module.get()});
        return registered_layer;
    }
}

#endif // Nott_LAYER_S4_HPP