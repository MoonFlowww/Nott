#ifndef Nott_RECURRENT_HPP
#define Nott_RECURRENT_HPP

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <tuple>
#include <type_traits>

#include <torch/torch.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/cuda.h>

#include "../../activation/activation.hpp"
#include "../../common/local.hpp"
#include "../../initialization/initialization.hpp"
#include "../registry.hpp"


namespace Nott::Layer::Details {

    // -------- Options --------
    struct RNNOptions {
        std::int64_t input_size{};
        std::int64_t hidden_size{};
        std::int64_t num_layers{1};
        double dropout{0.0};
        bool batch_first{false};
        bool bidirectional{false};
        std::string nonlinearity{"tanh"};
    };

    struct LSTMOptions {
        std::int64_t input_size{};
        std::int64_t hidden_size{};
        std::int64_t num_layers{1};
        double dropout{0.0};
        bool batch_first{false};
        bool bidirectional{false};
        bool bias{true};
        double forget_gate_bias{1.0}; // applied after init if bias==true
        c10::ScalarType param_dtype{at::kFloat};
        bool allow_tf32{true};
        bool benchmark_cudnn{true};
    };

    struct GRUOptions {
        std::int64_t input_size{};
        std::int64_t hidden_size{};
        std::int64_t num_layers{1};
        double dropout{0.0};
        bool batch_first{false};
        bool bidirectional{false};
        bool bias{true};
        c10::ScalarType param_dtype{at::kFloat};
        bool allow_tf32{true};
        bool benchmark_cudnn{true};
    };

    // xLSTMOptions mirrors LSTMOptions so existing descriptors compile.
    using xLSTMOptions = LSTMOptions;

    // -------- Descriptors (carry Activation/Initialization + Local flags) --------
    struct RNNDescriptor {
        RNNOptions options{};
        ::Nott::Activation::Descriptor activation{::Nott::Activation::Identity};
        ::Nott::Initialization::Descriptor initialization{::Nott::Initialization::Default};
        ::Nott::LocalConfig local{};
    };

    struct LSTMDescriptor {
        LSTMOptions options{};
        ::Nott::Activation::Descriptor activation{::Nott::Activation::Identity};
        ::Nott::Initialization::Descriptor initialization{::Nott::Initialization::Default};
        ::Nott::LocalConfig local{};
    };

    struct GRUDescriptor {
        GRUOptions options{};
        ::Nott::Activation::Descriptor activation{::Nott::Activation::Identity};
        ::Nott::Initialization::Descriptor initialization{::Nott::Initialization::Default};
        ::Nott::LocalConfig local{};
    };

    struct xLSTMDescriptor {
        xLSTMOptions options{};
        ::Nott::Activation::Descriptor activation{::Nott::Activation::Identity};
        ::Nott::Initialization::Descriptor initialization{::Nott::Initialization::Default};
        ::Nott::LocalConfig local{};
    };

    // -------- Internal detail helpers --------
    namespace Detail {

        // Adapt any RNN-like forward() result into the first tensor (sequence output).
        inline torch::Tensor take_recurrent_output(const torch::Tensor& t) {
            return t;
        }
        template<class A, class... Rest>
        inline torch::Tensor take_recurrent_output(const std::tuple<A, Rest...>& tup) {
            return std::get<0>(tup);
        }
        // Convert our options to LibTorch's RNN/GRU/LSTM options
        inline torch::nn::RNNOptions to_torch_rnn_options(const RNNOptions& o)
        {
            auto options = torch::nn::RNNOptions(o.input_size, o.hidden_size);
            options = options.num_layers(o.num_layers);
            options = options.dropout(o.dropout);
            options = options.batch_first(o.batch_first);
            options = options.bidirectional(o.bidirectional);

            std::string nonlinearity = o.nonlinearity;
            std::transform(
                nonlinearity.begin(),
                nonlinearity.end(),
                nonlinearity.begin(),
                [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });

            if (nonlinearity == "tanh") {
                options = options.nonlinearity(torch::kTanh);
            } else if (nonlinearity == "relu") {
                options = options.nonlinearity(torch::kReLU);
            } else {
                throw std::invalid_argument("Unsupported RNN nonlinearity: " + o.nonlinearity);
            }

            return options;
        }

        inline torch::nn::GRUOptions to_torch_gru_options(const GRUOptions& o)
        {
            auto options = torch::nn::GRUOptions(o.input_size, o.hidden_size);
            options = options.num_layers(o.num_layers);
            options = options.dropout(o.dropout);
            options = options.batch_first(o.batch_first);
            options = options.bidirectional(o.bidirectional);
            options = options.bias(o.bias);
            return options;
        }

        // Convert our options to LibTorch's LSTMOptions
        inline torch::nn::LSTMOptions to_torch_lstm_options(const LSTMOptions& o) {
            torch::nn::LSTMOptions opt(o.input_size, o.hidden_size);
            opt = opt.num_layers(o.num_layers);
            opt = opt.dropout(o.dropout);
            opt = opt.batch_first(o.batch_first);
            opt = opt.bidirectional(o.bidirectional);
            opt = opt.bias(o.bias);
            return opt;
        }

        // --- xLSTM: a thin wrapper around torch::nn::LSTM with extras (bias tweak, flags) ---
        class xLSTMImpl : public torch::nn::Module {
        public:
            explicit xLSTMImpl(const xLSTMOptions& options)
                : options_(options),
                  lstm_(torch::nn::LSTM(to_torch_lstm_options(options))) {

                // dtype for parameters (weights/bias of the LSTM)
                if (options_.param_dtype != at::kFloat) {
                    this->to(options_.param_dtype);
                }

                register_module("lstm", lstm_);

                // cuDNN setup (use new precision API instead of deprecated allowTF32)
                const char* tf32_setting = options_.allow_tf32 ? "tf32" : "none";
                at::globalContext().setAllowTF32CuDNN(options_.allow_tf32);
                at::globalContext().setAllowTF32CuBLAS(options_.allow_tf32);
                if (torch::cuda::is_available() && torch::cuda::cudnn_is_available()) {
                    at::globalContext().setBenchmarkCuDNN(options_.benchmark_cudnn);
                }

                // initialize weights/biases according to project's policy
                // (we let higher-level Initializer handle weight init. Here we only fix forget bias.)
                if (options_.bias && options_.forget_gate_bias != 0.0) {
                    set_forget_gate_bias_(options_.forget_gate_bias);
                }

                flatten_parameters_if_possible_();
            }

            // Forward returns the sequence output tensor (T,B,H*D or B,T,H*D) for consistency
            torch::Tensor forward(torch::Tensor x) {
                ensure_input_rank_(x);
                auto out = lstm_->forward(x); // tuple<Tensor output, (h, c)>
                return take_recurrent_output(out);
            }

        private:
            void ensure_input_rank_(const torch::Tensor& x) const {
                if (x.dim() != 3) {
                    throw std::invalid_argument("xLSTM expects a 3D tensor [T,B,F] or [B,T,F].");
                }
            }

            // Ensure (h0,c0) match expected sizes if we ever expose it (kept for parity with the old code).
            void ensure_hx_shapes_(const torch::Tensor& h0, const torch::Tensor& c0, const torch::Tensor& x) const {
                const int64_t num_dir = options_.bidirectional ? 2 : 1;
                const int64_t expected_h0_0 = options_.num_layers * num_dir;
                if (h0.size(0) != expected_h0_0 || c0.size(0) != expected_h0_0) {
                    throw std::invalid_argument("xLSTM (h0/c0) wrong number of layers or directions.");
                }
                const int64_t batch = options_.batch_first ? x.size(0) : x.size(1);
                if (h0.size(1) != batch || c0.size(1) != batch) {
                    throw std::invalid_argument("xLSTM (h0/c0) batch mismatch.");
                }
                if (h0.size(2) != options_.hidden_size || c0.size(2) != options_.hidden_size) {
                    throw std::invalid_argument("xLSTM (h0/c0) hidden_size mismatch.");
                }
            }

            void flatten_parameters_if_possible_() {
                if (torch::cuda::is_available() && torch::cuda::cudnn_is_available()) {
                    if (lstm_ && !lstm_->parameters().empty()) {
                        lstm_->flatten_parameters();
                    }
                }
            }

            // Set forget gate bias across all layers & directions.
            void set_forget_gate_bias_(double value) {
                torch::NoGradGuard no_grad;
                const int64_t num_dir = options_.bidirectional ? 2 : 1;
                const int64_t H = options_.hidden_size;
                const int64_t fourH = 4 * H;

                // Grab the parameter dictionary once so any pointers we take remain valid
                auto named_params = lstm_->named_parameters(/*recurse=*/false);

                for (int64_t layer = 0; layer < options_.num_layers; ++layer) {
                    for (int64_t dir = 0; dir < num_dir; ++dir) {
                        const std::string suffix = std::to_string(layer) + (dir == 0 ? "" : "_reverse");
                        const auto ih_name = "bias_ih_l" + suffix;
                        const auto hh_name = "bias_hh_l" + suffix;

                        auto* bias_ih = named_params.find(ih_name);
                        auto* bias_hh = named_params.find(hh_name);
                        if (!bias_ih || !bias_hh) {
                            continue;
                        }

                        if (bias_ih->numel() != fourH || bias_hh->numel() != fourH) {
                            continue;
                        }

                        bias_ih->narrow(0, H, H).fill_(value);
                        bias_hh->narrow(0, H, H).fill_(0.0); // keep hh forget bias neutral
                    }
                }
            }

        private:
            xLSTMOptions options_;
            torch::nn::LSTM lstm_{nullptr};
        };

    } // namespace Detail

    // Expose xLSTM in Nott::Layer::Details scope while keeping the impl in ::Detail
    using Detail::xLSTMImpl;
    TORCH_MODULE(xLSTM);

    // ---------------- Registry bindings ----------------
    // We only bind RNN/LSTM/GRU/xLSTM here. S4/etc. live elsewhere.

    template <class Owner>
    RegisteredLayer build_registered_layer(Owner& owner, const RNNDescriptor& descriptor, std::size_t index)
    {
        auto module = owner.register_module("recurrent_" + std::to_string(index), torch::nn::RNN(Detail::to_torch_rnn_options(descriptor.options)));

        if constexpr (requires(Owner& o, torch::nn::Module& m, ::Nott::Initialization::Descriptor d) {
                          o.apply_initialization(m, d);
                      }) {
            owner.apply_initialization(*module, descriptor.initialization);
                      }

        RegisteredLayer registered_layer;
        registered_layer.module = to_shared_module_ptr(module);
        registered_layer.local = descriptor.local;

        struct ForwardFunctor {
            decltype(module.get()) module_ptr;
            torch::Tensor operator()(torch::Tensor input) const
            {
                auto out = module_ptr->forward(std::move(input));
                return Detail::take_recurrent_output(out);
            }
        };
        registered_layer.bind_inline_forward(ForwardFunctor{module.get()});
        return registered_layer;
    }
    template <class Owner>
    RegisteredLayer build_registered_layer(Owner& owner, const LSTMDescriptor& descriptor, std::size_t index) {
        auto module = owner.register_module("recurrent_" + std::to_string(index), torch::nn::LSTM(Detail::to_torch_lstm_options(descriptor.options)));

        // Optional initialization hook at Model level
        if constexpr (requires(Owner& o, torch::nn::Module& m, ::Nott::Initialization::Descriptor d) {
            o.apply_initialization(m, d);
        }) {
            owner.apply_initialization(*module, descriptor.initialization);
        }

        RegisteredLayer registered_layer;
        registered_layer.module = to_shared_module_ptr(module);
        registered_layer.local = descriptor.local;

        struct ForwardFunctor {
            decltype(module.get()) module_ptr;
            torch::Tensor operator()(torch::Tensor input) const {
                auto out = module_ptr->forward(std::move(input));
                return Detail::take_recurrent_output(out);
            }
        };
        registered_layer.bind_inline_forward(ForwardFunctor{module.get()});
        return registered_layer;
    }

    template <class Owner>
    RegisteredLayer build_registered_layer(Owner& owner, const GRUDescriptor& descriptor, std::size_t index)
    {
        auto module = owner.register_module("recurrent_" + std::to_string(index), torch::nn::GRU(Detail::to_torch_gru_options(descriptor.options)));

        if (descriptor.options.param_dtype != at::kFloat) {
            module->to(descriptor.options.param_dtype);
        }

        const char* tf32_setting = descriptor.options.allow_tf32 ? "tf32" : "none";
        at::globalContext().setAllowTF32CuDNN(descriptor.options.allow_tf32);   // covers RNN + conv via cuDNN
        at::globalContext().setAllowTF32CuBLAS(descriptor.options.allow_tf32);

        if (torch::cuda::is_available() && torch::cuda::cudnn_is_available()) {
            at::globalContext().setBenchmarkCuDNN(descriptor.options.benchmark_cudnn);
            if (module && !module->parameters().empty()) {
                module->flatten_parameters();
            }
        }

        if constexpr (requires(Owner& o, torch::nn::Module& m, ::Nott::Initialization::Descriptor d) {
                          o.apply_initialization(m, d);
                      }) {
            owner.apply_initialization(*module, descriptor.initialization);
                      }

        RegisteredLayer registered_layer;
        registered_layer.module = to_shared_module_ptr(module);
        registered_layer.local = descriptor.local;

        struct ForwardFunctor {
            decltype(module.get()) module_ptr;
            torch::Tensor operator()(torch::Tensor input) const
            {
                auto out = module_ptr->forward(std::move(input));
                return Detail::take_recurrent_output(out);
            }
        };
        registered_layer.bind_inline_forward(ForwardFunctor{module.get()});
        return registered_layer;
    }


    template <class Owner>
    RegisteredLayer build_registered_layer(Owner& owner, const xLSTMDescriptor& descriptor, std::size_t index) {
        // Build our wrapper (still uses torch::nn::LSTM underneath)
        auto module = owner.register_module("recurrent_" + std::to_string(index), xLSTM(descriptor.options));

        if constexpr (requires(Owner& o, torch::nn::Module& m, ::Nott::Initialization::Descriptor d) {
            o.apply_initialization(m, d);
        }) {
            owner.apply_initialization(*module, descriptor.initialization);
        }

        RegisteredLayer registered_layer;
        registered_layer.module = to_shared_module_ptr(module);
        registered_layer.local = descriptor.local;

        struct ForwardFunctor {
            decltype(module.get()) module_ptr;
            torch::Tensor operator()(torch::Tensor input) const {
                // xLSTMImpl::forward already returns the sequence output
                auto out = module_ptr->forward(std::move(input));
                return Detail::take_recurrent_output(out);
            }
        };
        registered_layer.bind_inline_forward(ForwardFunctor{module.get()});
        return registered_layer;
    }

}

#endif // Nott_RECURRENT_HPP
