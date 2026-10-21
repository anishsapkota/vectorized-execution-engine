#pragma once

#include <cstddef>
#include <functional>
#include <vector>

namespace wb {

struct StreamTiming {
    std::vector<double> slice_ms;  // index = slice
    double              wall_ms = 0.0;

    double slowest_slice_ms() const;
    double fastest_slice_ms() const;
};

// Runs `work(slice)` on every slice at once, one thread per slice, and returns after all
// finish. Redshift cuts a plan into streams at exchange boundaries for the same reason: a
// receiver cannot start until every sender is done, so each boundary is a barrier.
//
// One thread per slice with no work stealing is deliberate: a slice holding more rows simply
// finishes later, and the stream's wall time is its slowest slice. That is the skew behaviour
// the experiments measure, and the contrast with morsel-driven parallel_group_sum_count.
//
// If any slice throws, the first exception (by slice index) is rethrown after all threads join.
StreamTiming run_stream(std::size_t num_slices, const std::function<void(std::size_t)>& work);

}  // namespace wb
