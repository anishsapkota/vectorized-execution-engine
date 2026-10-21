#include "dist/stream.h"

#include <algorithm>
#include <chrono>
#include <exception>
#include <thread>

namespace wb {

namespace {
using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
}  // namespace

double StreamTiming::slowest_slice_ms() const {
    return slice_ms.empty() ? 0.0 : *std::max_element(slice_ms.begin(), slice_ms.end());
}

double StreamTiming::fastest_slice_ms() const {
    return slice_ms.empty() ? 0.0 : *std::min_element(slice_ms.begin(), slice_ms.end());
}

StreamTiming run_stream(std::size_t num_slices, const std::function<void(std::size_t)>& work) {
    StreamTiming                    timing;
    std::vector<std::exception_ptr> errors(num_slices);
    timing.slice_ms.resize(num_slices);

    const auto               stream_start = Clock::now();
    std::vector<std::thread> threads;
    threads.reserve(num_slices);
    for (std::size_t slice = 0; slice < num_slices; ++slice) {
        threads.emplace_back([&, slice] {
            const auto slice_start = Clock::now();
            try {
                work(slice);
            } catch (...) {
                errors[slice] = std::current_exception();
            }
            timing.slice_ms[slice] = ms_since(slice_start);
        });
    }
    for (auto& thread : threads)
        thread.join();
    timing.wall_ms = ms_since(stream_start);

    for (const auto& error : errors) {
        if (error) std::rethrow_exception(error);
    }
    return timing;
}

}  // namespace wb
