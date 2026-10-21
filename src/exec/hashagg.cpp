#include "exec/hashagg.h"

#include <algorithm>
#include <stdexcept>
#include <span>
#include <string>

#include "exec/batch.h"
#include "storage/vector.h"

namespace wb {

namespace {

// The live rows of a batch, as an index sequence: a contiguous range (no selection vector) or the
// selection vector itself. Templating the loops on it avoids materializing row ids.
struct RangeRows {
    std::uint32_t start;
    std::uint32_t n;
    std::uint32_t operator[](std::size_t k) const { return start + static_cast<std::uint32_t>(k); }
    std::size_t   size() const { return n; }
};

struct SelRows {
    std::span<const std::uint32_t> sel;
    std::uint32_t operator[](std::size_t k) const { return sel[k]; }
    std::size_t   size() const { return sel.size(); }
};

// Call f(k, value) for each live row rows[k], with the column's encoding resolved once.
template <typename Rows, typename F>
void for_each_int(const Vector& col, const Rows& rows, F&& f) {
    static_cast<const Int64Vector&>(col).visit([&](auto codes, int64_t base) {
        for (std::size_t k = 0; k < rows.size(); ++k)
            f(k, Int64Vector::decode(base, codes[rows[k]]));
    });
}

template <typename Rows, typename F>
void for_each_double(const Vector& col, const Rows& rows, F&& f) {
    static_cast<const Float64Vector&>(col).visit([&](auto values) {
        for (std::size_t k = 0; k < rows.size(); ++k)
            f(k, static_cast<double>(values[rows[k]]));
    });
}

template <typename T>
void update_min(T& m, T v) {
    if (v < m || v != v) m = v;
}

template <typename T>
void update_max(T& m, T v) {
    if (v > m || v != v) m = v;
}

}  // namespace

HashAggOperator::HashAggOperator(std::unique_ptr<Operator> child, std::optional<std::size_t> group_col,
                                 std::vector<AggSpec> aggs)
    : child_(std::move(child)),
      group_col_(group_col) {
    for (const AggSpec& spec : aggs) {
        if (spec.kind == AggKind::Median) throw std::invalid_argument("aggregate median not supported yet");
        states_.push_back(State{spec, {}, {}, {}});
    }
}

HashAggOperator::HashAggOperator(std::unique_ptr<Operator> child, std::size_t group_col, std::size_t agg_col)
    : HashAggOperator(std::move(child), group_col,
                      {AggSpec{AggKind::Sum, agg_col, DataType::Float64}, AggSpec{AggKind::Count}}) {}

void HashAggOperator::add_group_(const Batch& batch, std::uint32_t row) {
    counts_.push_back(0);
    for (State& s : states_) {
        const AggSpec& spec = s.spec;
        if (spec.kind == AggKind::Count) continue;
        const bool is_int = spec.type == DataType::Int64;
        if (spec.kind == AggKind::Sum || spec.kind == AggKind::Avg) {
            if (is_int) s.i128.push_back(0);
            else s.f64.push_back(0.0);  // +0.0; the first row is added like every other
        } else if (is_int) {            // Min/Max start at the group's first value
            s.i64.push_back(static_cast<const Int64Vector&>(batch.column(spec.col)).value(row));
        } else {
            s.f64.push_back(static_cast<const Float64Vector&>(batch.column(spec.col)).value(row));
        }
    }
}

void HashAggOperator::consume_(const Batch& batch) {
    if (batch.has_sel()) consume_rows_(batch, SelRows{batch.sel()});
    else consume_rows_(batch, RangeRows{batch.offset(), batch.size()});
}

template <typename Rows>
void HashAggOperator::consume_rows_(const Batch& batch, const Rows& rows) {
    gids_.resize(rows.size());

    // Pass 1: group ids and counts (a global aggregate is group 0).
    if (group_col_) {
        for_each_int(batch.column(*group_col_), rows, [&](std::size_t k, int64_t key) {
            const std::uint32_t g = index_.find_or_insert(key);
            if (g == counts_.size()) add_group_(batch, rows[k]);
            ++counts_[g];
            gids_[k] = g;
        });
    } else {
        if (counts_.empty() && rows.size() > 0) add_group_(batch, rows[0]);
        std::fill(gids_.begin(), gids_.end(), 0u);
        if (rows.size() > 0) counts_[0] += static_cast<int64_t>(rows.size());
    }

    // Pass 2: one loop per aggregate; kind and type are resolved before the row loop.
    for (State& s : states_) {
        const AggSpec& spec = s.spec;
        if (spec.kind == AggKind::Count) continue;
        const Vector&  col    = batch.column(spec.col);
        const auto&    gids   = gids_;
        const bool     is_int = spec.type == DataType::Int64;
        switch (spec.kind) {
            case AggKind::Sum:
            case AggKind::Avg:
                if (is_int) for_each_int(col, rows, [&](std::size_t k, int64_t v) { s.i128[gids[k]] += v; });
                else for_each_double(col, rows, [&](std::size_t k, double v) { s.f64[gids[k]] += v; });
                break;
            case AggKind::Min:
                if (is_int) for_each_int(col, rows, [&](std::size_t k, int64_t v) { update_min(s.i64[gids[k]], v); });
                else for_each_double(col, rows, [&](std::size_t k, double v) { update_min(s.f64[gids[k]], v); });
                break;
            case AggKind::Max:
                if (is_int) for_each_int(col, rows, [&](std::size_t k, int64_t v) { update_max(s.i64[gids[k]], v); });
                else for_each_double(col, rows, [&](std::size_t k, double v) { update_max(s.f64[gids[k]], v); });
                break;
            case AggKind::Count:
            case AggKind::Median:
                break;
        }
    }
}

void HashAggOperator::make_output_() {
    const std::size_t groups = counts_.size();
    if (group_col_) {
        std::vector<int64_t> keys(groups);
        for (std::uint32_t g = 0; g < groups; ++g)
            keys[g] = index_.key_of(g);
        out_vecs_.push_back(std::make_unique<Int64Vector>(keys));
    }
    for (const State& s : states_) {
        const AggSpec& spec   = s.spec;
        const bool     is_int = spec.type == DataType::Int64;
        switch (spec.kind) {
            case AggKind::Count:
                out_vecs_.push_back(std::make_unique<Int64Vector>(counts_));
                break;
            case AggKind::Sum:
                if (is_int) {
                    std::vector<int64_t> sums(groups);
                    for (std::uint32_t g = 0; g < groups; ++g) {
                        if (s.i128[g] > INT64_MAX || s.i128[g] < INT64_MIN) {
                            throw std::overflow_error(
                                "SUM overflow in group " +
                                (group_col_ ? std::to_string(index_.key_of(g)) : std::string{"global"}));
                        }
                        sums[g] = static_cast<int64_t>(s.i128[g]);
                    }
                    out_vecs_.push_back(std::make_unique<Int64Vector>(sums));
                } else {
                    out_vecs_.push_back(std::make_unique<Float64Vector>(s.f64));
                }
                break;
            case AggKind::Avg: {
                std::vector<double> avg(groups);
                for (std::uint32_t g = 0; g < groups; ++g) {
                    const double n = static_cast<double>(counts_[g]);
                    avg[g]         = is_int ? static_cast<double>(s.i128[g]) / n : s.f64[g] / n;
                }
                out_vecs_.push_back(std::make_unique<Float64Vector>(std::move(avg)));
                break;
            }
            case AggKind::Min:
            case AggKind::Max:
                if (is_int) out_vecs_.push_back(std::make_unique<Int64Vector>(s.i64));
                else out_vecs_.push_back(std::make_unique<Float64Vector>(s.f64));
                break;
            case AggKind::Median:
                break;  // rejected in the constructor
        }
    }
    for (const auto& v : out_vecs_)
        out_cols_.push_back(v.get());
}

void HashAggOperator::build_() {
    while (auto batch = child_->next())
        consume_(*batch);
    make_output_();
    built_ = true;
}

std::optional<Batch> HashAggOperator::next() {
    if (!built_) build_();

    const std::size_t total = counts_.size();
    if (emit_cursor_ >= total) return std::nullopt;
    const std::size_t batch_size = std::min(static_cast<std::size_t>(emit_batch_size_), total - emit_cursor_);
    const auto        offset     = static_cast<std::uint32_t>(emit_cursor_);
    emit_cursor_ += batch_size;
    return Batch{out_cols_, offset, static_cast<std::uint32_t>(batch_size)};
}

}  // namespace wb
