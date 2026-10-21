#include "exec/filter.h"

#include <functional>
#include <span>

#include "exec/batch.h"
#include "storage/vector.h"

namespace wb {

namespace {

// Write the indices of live rows for which pred(i) holds into out, and return how many there are.
// Branchless: every row index is stored, and the write cursor only advances when pred holds, so
// mid-selectivity filters don't pay for branch mispredictions. out is grown to the batch size
// once and keeps its capacity across calls.
template <typename Pred>
std::size_t select_rows(const Batch& batch, Pred pred, std::vector<std::uint32_t>& out) {
    if (out.size() < batch.size()) out.resize(batch.size());
    std::uint32_t* dst = out.data();
    std::size_t    n   = 0;
    if (batch.has_sel()) {
        for (std::uint32_t i : batch.sel()) {
            dst[n] = i;
            n += pred(i);
        }
    } else {
        const std::uint32_t end = batch.offset() + batch.size();
        for (std::uint32_t i = batch.offset(); i < end; ++i) {
            dst[n] = i;
            n += pred(i);
        }
    }
    return n;
}

// Switch on CompareOp once per batch. get(i) reads row i of the filtered column.
template <typename Get, typename T>
std::size_t dispatch_op(const Batch& batch, Get get, CompareOp op, const T& lit,
                        std::vector<std::uint32_t>& out) {
    auto run = [&](auto cmp) {
        return select_rows(batch, [&](std::uint32_t i) { return cmp(get(i), lit); }, out);
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
    return 0;
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

        // Dispatch on (column type × encoding × op) once per batch.
        // survivors_buf_ is reused across calls — it grows to the largest batch size and stays
        // there, and the returned batch borrows it, so no per-batch heap allocation.
        std::size_t n = 0;
        switch (col.type()) {
            case DataType::Int64: {
                int64_t lit = std::get<int64_t>(predicate_.value);
                static_cast<const Int64Vector&>(col).visit([&](auto codes, int64_t base) {
                    auto get = [&](std::uint32_t i) { return Int64Vector::decode(base, codes[i]); };
                    n        = dispatch_op(*batch, get, predicate_.op, lit, survivors_buf_);
                });
                break;
            }
            case DataType::Float64: {
                double lit = std::get<double>(predicate_.value);
                static_cast<const Float64Vector&>(col).visit([&](auto values) {
                    auto get = [&](std::uint32_t i) { return static_cast<double>(values[i]); };
                    n        = dispatch_op(*batch, get, predicate_.op, lit, survivors_buf_);
                });
                break;
            }
            case DataType::String: {
                const auto&        values = static_cast<const StringVector&>(col).data();
                const std::string& lit    = std::get<std::string>(predicate_.value);
                auto get = [&](std::uint32_t i) -> const std::string& { return values[i]; };
                n        = dispatch_op(*batch, get, predicate_.op, lit, survivors_buf_);
                break;
            }
        }

        if (n == 0) continue;

        batch->set_sel(std::span<const std::uint32_t>(survivors_buf_.data(), n));
        return batch;
    }
}

}  // namespace wb
