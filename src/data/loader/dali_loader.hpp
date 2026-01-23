#ifndef NOTT_DALI_LOADER_HPP
#define NOTT_DALI_LOADER_HPP

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>

#include <torch/torch.h>

#include "loader.hpp"
#include "../../common/streaming.hpp"

#if __has_include(<dali/pipeline/pipeline.h>)
#include <dali/pipeline/pipeline.h>
#include <dali/pipeline/workspace/workspace.h>
#define NOTT_HAS_DALI 1
#else
#define NOTT_HAS_DALI 0
#endif

namespace Nott::Data::Loader {
#if NOTT_HAS_DALI
    namespace Detail {
        template <typename Backend>
        inline torch::Tensor tensor_from_dali(dali::TensorList<Backend> &tensor_list)
        {
            auto *dlpack = tensor_list.ToDLTensor();
            if (dlpack == nullptr) {
                throw std::runtime_error("Failed to export DALI TensorList to DLPack.");
            }
            return torch::fromDLPack(dlpack);
        }
    }
#endif

    class DaliLoader final : public BaseLoader {
    public:
        struct Options {
            std::size_t queue_depth{2};
            bool drop_last{false};
            bool fixed_batch_size{false};
            GraphMode graph_mode{GraphMode::Disabled};
            bool use_shared_outputs{true};
            std::size_t seed{0};
        };

        DaliLoader(std::unique_ptr<dali::Pipeline> pipeline,
                   std::size_t batch_size,
                   Options options = {});
        ~DaliLoader() override;

        void start_epoch(std::size_t epoch) override;
        bool has_next() override;
        Batch next_batch() override;
        [[nodiscard]] std::size_t batch_size() const override;
        [[nodiscard]] bool drop_last() const override;
        void shutdown() override;

    private:
        struct Impl {
            std::unique_ptr<dali::Pipeline> pipeline{};
            std::size_t batch_size{0};
            std::size_t queue_depth{2};
            bool drop_last{false};
            bool fixed_batch_size{false};
            GraphMode graph_mode{GraphMode::Disabled};
            bool use_shared_outputs{true};
            std::size_t seed{0};

            std::atomic<bool> stop_requested{false};
            bool closed{false};
            std::mutex mutex{};
            std::condition_variable has_data{};
            std::condition_variable has_space{};
            std::deque<Batch> ready{};
            std::thread producer{};

            void request_stop()
            {
                stop_requested.store(true);
                {
                    std::lock_guard<std::mutex> lock(mutex);
                    closed = true;
                }
                has_data.notify_all();
                has_space.notify_all();
            }

            void join()
            {
                if (producer.joinable()) {
                    producer.join();
                }
            }

            void clear_queue()
            {
                std::lock_guard<std::mutex> lock(mutex);
                ready.clear();
            }

            bool wait_for_space()
            {
                std::unique_lock<std::mutex> lock(mutex);
                has_space.wait(lock, [&] {
                    return stop_requested.load() || ready.size() < queue_depth;
                });
                return !stop_requested.load();
            }

            bool push_batch(Batch batch)
            {
                std::unique_lock<std::mutex> lock(mutex);
                has_space.wait(lock, [&] {
                    return stop_requested.load() || ready.size() < queue_depth;
                });
                if (stop_requested.load()) {
                    return false;
                }
                ready.push_back(std::move(batch));
                has_data.notify_one();
                return true;
            }

            Batch pop_batch()
            {
                std::unique_lock<std::mutex> lock(mutex);
                has_data.wait(lock, [&] {
                    return stop_requested.load() || closed || !ready.empty();
                });
                if (ready.empty()) {
                    return {};
                }
                Batch batch = std::move(ready.front());
                ready.pop_front();
                has_space.notify_one();
                return batch;
            }

            bool has_next() const
            {
                std::lock_guard<std::mutex> lock(mutex);
                return !ready.empty() || !closed;
            }

            void reset_epoch(std::size_t epoch)
            {
                if (!pipeline) {
                    throw std::runtime_error("DaliLoader pipeline is null.");
                }
                pipeline->Reset();
                pipeline->SetSeed(static_cast<unsigned int>(seed + epoch));
            }

            Batch normalize_batch(Batch batch) const
            {
                if (!batch.first.defined() || !batch.second.defined()) {
                    return {};
                }

                const auto current_size = static_cast<std::size_t>(batch.first.size(0));
                if (current_size == batch_size) {
                    return batch;
                }

                const bool graph_mode_enabled = graph_mode != GraphMode::Disabled;
                if (graph_mode_enabled && !fixed_batch_size) {
                    return {};
                }

                if (drop_last && !fixed_batch_size) {
                    return {};
                }

                if (!fixed_batch_size) {
                    return batch;
                }

                if (current_size > batch_size) {
                    throw std::runtime_error("DaliLoader received a batch larger than the configured batch size.");
                }

                auto inputs_sizes = batch.first.sizes().vec();
                inputs_sizes[0] = static_cast<long>(batch_size);
                auto padded_inputs = torch::zeros(inputs_sizes, batch.first.options());
                padded_inputs.narrow(0, 0, batch.first.size(0)).copy_(batch.first);

                auto targets_sizes = batch.second.sizes().vec();
                targets_sizes[0] = static_cast<long>(batch_size);
                auto padded_targets = torch::zeros(targets_sizes, batch.second.options());
                padded_targets.narrow(0, 0, batch.second.size(0)).copy_(batch.second);

                return {std::move(padded_inputs), std::move(padded_targets)};
            }

            void run()
            {
#if NOTT_HAS_DALI
                try {
                    while (!stop_requested.load()) {
                        if (!wait_for_space()) {
                            return;
                        }

                        pipeline->Run();
                        dali::DeviceWorkspace workspace{};
                        if (use_shared_outputs) {
                            pipeline->ShareOutputs(&workspace);
                        } else {
                            pipeline->Outputs(&workspace);
                        }

                        auto &images = workspace.Output<dali::GPUBackend>(0);
                        auto &labels = workspace.Output<dali::GPUBackend>(1);

                        auto inputs = Detail::tensor_from_dali(images);
                        auto targets = Detail::tensor_from_dali(labels);

                        if (use_shared_outputs) {
                            pipeline->ReleaseOutputs();
                        }

                        auto batch = normalize_batch({std::move(inputs), std::move(targets)});
                        if (!batch.first.defined() || !batch.second.defined()) {
                            continue;
                        }

                        if (!push_batch(std::move(batch))) {
                            return;
                        }
                    }
                } catch (...) {
                    request_stop();
                }
#else
                request_stop();
#endif
            }
        };

        std::unique_ptr<Impl> impl_{};
    };

    inline DaliLoader::DaliLoader(std::unique_ptr<dali::Pipeline> pipeline,
                                  std::size_t batch_size,
                                  Options options)
        : impl_(std::make_unique<Impl>())
    {
#if NOTT_HAS_DALI
        if (!pipeline) {
            throw std::runtime_error("DaliLoader requires a valid DALI pipeline.");
        }
        impl_->pipeline = std::move(pipeline);
        impl_->batch_size = batch_size;
        impl_->queue_depth = options.queue_depth;
        impl_->drop_last = options.drop_last;
        impl_->fixed_batch_size = options.fixed_batch_size;
        impl_->graph_mode = options.graph_mode;
        impl_->use_shared_outputs = options.use_shared_outputs;
        impl_->seed = options.seed;
        impl_->producer = std::thread([impl = impl_.get()] { impl->run(); });
#else
        static_cast<void>(pipeline);
        static_cast<void>(batch_size);
        static_cast<void>(options);
        throw std::runtime_error("DaliLoader built without DALI headers.");
#endif
    }

    inline DaliLoader::~DaliLoader()
    {
        shutdown();
    }

    inline void DaliLoader::start_epoch(std::size_t epoch)
    {
#if NOTT_HAS_DALI
        shutdown();
        impl_->stop_requested.store(false);
        impl_->closed = false;
        impl_->clear_queue();
        impl_->reset_epoch(epoch);
        impl_->producer = std::thread([impl = impl_.get()] { impl->run(); });
#else
        static_cast<void>(epoch);
#endif
    }

    inline bool DaliLoader::has_next()
    {
#if NOTT_HAS_DALI
        return impl_->has_next();
#else
        return false;
#endif
    }

    inline Batch DaliLoader::next_batch()
    {
#if NOTT_HAS_DALI
        return impl_->pop_batch();
#else
        return {};
#endif
    }

    inline std::size_t DaliLoader::batch_size() const
    {
#if NOTT_HAS_DALI
        return impl_->batch_size;
#else
        return 0;
#endif
    }

    inline bool DaliLoader::drop_last() const
    {
#if NOTT_HAS_DALI
        return impl_->drop_last || (impl_->graph_mode != GraphMode::Disabled && !impl_->fixed_batch_size);
#else
        return true;
#endif
    }

    inline void DaliLoader::shutdown()
    {
#if NOTT_HAS_DALI
        if (!impl_) {
            return;
        }
        impl_->request_stop();
        impl_->join();
#endif
    }
}

#endif //NOTT_DALI_LOADER_HPP