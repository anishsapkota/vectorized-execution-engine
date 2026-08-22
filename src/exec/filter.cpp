#include "exec/filter.h"

#include <functional>
#include <span>

#include "exec/batch.h"
#include "storage/vector.h"

namespace wb {

namespace {

// Fill `out` from rows in sel_in that satisfy cmp(data[i], literal).
// out.clear() resets size but preserves capacity — no alloc on second+ calls.
template <typename T, typename Compare>
void filter_kernel(std::span<const T> data, std::span<const std::uint32_t> sel_in, const T& literal,
                   Compare cmp, std::vector<std::uint32_t>& out) {
    out.clear();
    out.reserve(sel_in.size());
    for (std::uint32_t i : sel_in) {
        if (cmp(data[i], literal)) out.push_back(i);
    }
}

// Same but over a contiguous [start, end) range (no incoming sel vector).
template <typename T, typename Compare>
void filter_kernel_range(std::span<const T> data, std::uint32_t start, std::uint32_t end,
                         const T& literal, Compare cmp, std::vector<std::uint32_t>& out) {
    out.clear();
    out.reserve(end - start);
    for (std::uint32_t i = start; i < end; ++i) {
        if (cmp(data[i], literal)) out.push_back(i);
    }
}

// Switch on CompareOp once per batch, fill out in-place.
template <typename T>
void dispatch_op(std::span<const T> data, const Batch& batch, CompareOp op, const T& lit,
                 std::vector<std::uint32_t>& out) {
    auto run = [&](auto cmp) {
        if (batch.sel()) {
            filter_kernel(data, std::span<const std::uint32_t>(*batch.sel()), lit, cmp, out);
        } else {
            filter_kernel_range(data, batch.offset(), batch.offset() + batch.size(), lit, cmp, out);
        }
    };

    switch (op) {
        case CompareOp::Eq:
            return run(std::equal_to<>{});
        case CompareOp::Ne:
            return run(std::not_equal_to<>{});
        case CompareOp::Lt:
            return run(std::less<>{});
        case CompareOp::Le:
            return run(std::less_equal<>{});
        case CompareOp::Gt:
            return run(std::greater<>{});
        case CompareOp::Ge:
            return run(std::greater_equal<>{});
    }
}

}  // anonymous namespace

FilterOperator::FilterOperator(std::unique_ptr<Operator> child, Predicate predicate)
    : child_(std::move(child)),
      predicate_(std::move(predicate)) {}

std::optional<Batch> FilterOperator::next() {
    while (true) {
        auto batch = child_->next();
        if (!batch) return std::nullopt;

        const Vector& col = batch->column(predicate_.col_idx);

        // Dispatch on (column type × op) once per batch.
        // survivors_buf_ is reused across calls — capacity grows to high-water mark
        // and stays there, eliminating per-batch heap allocation.
        switch (col.type()) {
            case DataType::Int64: {
                auto    data = static_cast<const Int64Vector&>(col).data();
                int64_t lit  = std::get<int64_t>(predicate_.value);
                dispatch_op(data, *batch, predicate_.op, lit, survivors_buf_);
                break;
            }
            case DataType::Float64: {
                auto   data = static_cast<const Float64Vector&>(col).data();
                double lit  = std::get<double>(predicate_.value);
                dispatch_op(data, *batch, predicate_.op, lit, survivors_buf_);
                break;
            }
            case DataType::String: {
                std::span<const std::string> data = static_cast<const StringVector&>(col).data();
                const std::string&           lit  = std::get<std::string>(predicate_.value);
                dispatch_op(data, *batch, predicate_.op, lit, survivors_buf_);
                break;
            }
        }

        if (survivors_buf_.empty()) continue;

        // Copy survivors into the batch. survivors_buf_ keeps its capacity for next call.
        batch->set_sel(survivors_buf_);
        return batch;
    }
}

}  // namespace wb
