#ifndef Nott_DATA_LOADER_HPP
#define Nott_DATA_LOADER_HPP

#include <condition_variable>
#include <cstddef>
#include <deque>
#include <mutex>
#include <utility>

#include <torch/torch.h>

namespace Nott::Data::Loader {
    using Batch = std::pair<torch::Tensor, torch::Tensor>;

    class BaseLoader {
    public:
        virtual ~BaseLoader() = default;

        virtual void start_epoch(std::size_t epoch) = 0;
        virtual bool has_next() = 0;
        virtual Batch next_batch() = 0;
        [[nodiscard]] virtual std::size_t batch_size() const = 0;
        [[nodiscard]] virtual bool drop_last() const = 0;
        virtual void shutdown() = 0;
    };

    template<class T>
    class ThreadSafeQueue {
    public:
        bool push(T value) {
            std::lock_guard<std::mutex> lock(mutex_);
            if (closed_) {
                return false;
            }
            queue_.push_back(std::move(value));
            condition_.notify_one();
            return true;
        }

        bool try_pop(T &value) {
            std::lock_guard<std::mutex> lock(mutex_);
            if (queue_.empty()) {
                return false;
            }
            value = std::move(queue_.front());
            queue_.pop_front();
            return true;
        }

        bool wait_pop(T &value) {
            std::unique_lock<std::mutex> lock(mutex_);
            condition_.wait(lock, [&] { return closed_ || !queue_.empty(); });
            if (queue_.empty()) {
                return false;
            }
            value = std::move(queue_.front());
            queue_.pop_front();
            return true;
        }

        void close() {
            std::lock_guard<std::mutex> lock(mutex_);
            closed_ = true;
            condition_.notify_all();
        }

        [[nodiscard]] bool closed() const {
            std::lock_guard<std::mutex> lock(mutex_);
            return closed_;
        }

    private:
        mutable std::mutex mutex_{};
        std::condition_variable condition_{};
        std::deque<T> queue_{};
        bool closed_{false};
    };
}

#endif //Nott_DATA_LOADER_HPP
