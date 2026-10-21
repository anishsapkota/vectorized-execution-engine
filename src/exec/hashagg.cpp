#include "exec/hashagg.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <functional>
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
        states_.push_back(State{spec, {}, {}, {}, {}, {}, {}});
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
    if (group_col_) gids_.resize(rows.size());

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
        if (rows.size() > 0) counts_[0] += static_cast<int64_t>(rows.size());
    }

    if (!group_col_) return consume_global_(batch, rows);

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
            case AggKind::Median:
                s.med_gids.insert(s.med_gids.end(), gids.begin(), gids.end());
                if (is_int) for_each_int(col, rows, [&](std::size_t, int64_t v) { s.med_i64.push_back(v); });
                else for_each_double(col, rows, [&](std::size_t, double v) { s.med_f64.push_back(v); });
                break;
            case AggKind::Count:
                break;
        }
    }
}

// Global aggregate: no group ids needed. Each aggregate accumulates in a local for the batch (a
// register, not a memory read-modify-write per row), in row order like the grouped path.
template <typename Rows>
void HashAggOperator::consume_global_(const Batch& batch, const Rows& rows) {
    for (State& s : states_) {
        const AggSpec& spec = s.spec;
        if (spec.kind == AggKind::Count || rows.size() == 0) continue;
        const Vector& col    = batch.column(spec.col);
        const bool    is_int = spec.type == DataType::Int64;
        switch (spec.kind) {
            case AggKind::Sum:
            case AggKind::Avg:
                if (is_int) {
                    __int128 acc = s.i128[0];
                    for_each_int(col, rows, [&](std::size_t, int64_t v) { acc += v; });
                    s.i128[0] = acc;
                } else {
                    double acc = s.f64[0];
                    for_each_double(col, rows, [&](std::size_t, double v) { acc += v; });
                    s.f64[0] = acc;
                }
                break;
            case AggKind::Min:
            case AggKind::Max: {
                const bool is_min = spec.kind == AggKind::Min;
                if (is_int) {
                    int64_t m = s.i64[0];
                    for_each_int(col, rows, [&](std::size_t, int64_t v) { is_min ? update_min(m, v) : update_max(m, v); });
                    s.i64[0] = m;
                } else {
                    double m = s.f64[0];
                    for_each_double(col, rows, [&](std::size_t, double v) { is_min ? update_min(m, v) : update_max(m, v); });
                    s.f64[0] = m;
                }
                break;
            }
            case AggKind::Median:
                s.med_gids.resize(s.med_gids.size() + rows.size(), 0u);
                if (is_int) for_each_int(col, rows, [&](std::size_t, int64_t v) { s.med_i64.push_back(v); });
                else for_each_double(col, rows, [&](std::size_t, double v) { s.med_f64.push_back(v); });
                break;
            case AggKind::Count:
                break;
        }
    }
}

namespace {

// MEDIAN's total order (shared with the codegen runtime): numbers ascending with -0.0 before
// +0.0, then NaNs ordered by bit pattern. Values `<` cannot tell apart (signed zeros, NaN
// payloads) get a fixed position, so the median does not depend on the selection algorithm.
bool less_nan_last(double a, double b) {
    const bool a_nan = std::isnan(a), b_nan = std::isnan(b);
    if (a_nan || b_nan) return !a_nan || (b_nan && std::bit_cast<uint64_t>(a) < std::bit_cast<uint64_t>(b));
    return a < b || (a == b && std::signbit(a) && !std::signbit(b));
}

// Median of values[begin, end) (non-empty); reorders the range.
template <typename T, typename Less>
double median_of(T* begin, T* end, Less less) {
    const std::size_t n   = static_cast<std::size_t>(end - begin);
    T*                mid = begin + n / 2;
    std::nth_element(begin, mid, end, less);
    if (n % 2) return static_cast<double>(*mid);
    const T lo = *std::max_element(begin, mid, less);  // everything left of mid is <= *mid
    return (static_cast<double>(lo) + static_cast<double>(*mid)) / 2.0;
}

template <typename T, typename Less>
std::vector<double> group_medians(const std::vector<std::uint32_t>& gids, const std::vector<T>& values,
                                  const std::vector<int64_t>& counts, Less less) {
    // Counting sort by group: every row of a group contributed exactly one value.
    std::vector<std::size_t> start(counts.size() + 1, 0);
    for (std::size_t g = 0; g < counts.size(); ++g)
        start[g + 1] = start[g] + static_cast<std::size_t>(counts[g]);
    std::vector<T>           bucketed(values.size());
    std::vector<std::size_t> next(start.begin(), start.end() - 1);
    for (std::size_t i = 0; i < values.size(); ++i)
        bucketed[next[gids[i]]++] = values[i];

    std::vector<double> out(counts.size());
    for (std::size_t g = 0; g < counts.size(); ++g)
        out[g] = median_of(bucketed.data() + start[g], bucketed.data() + start[g + 1], less);
    return out;
}

}  // namespace

std::vector<double> HashAggOperator::medians_(State& s) const {
    std::vector<double> out = s.spec.type == DataType::Int64
                                  ? group_medians(s.med_gids, s.med_i64, counts_, std::less<int64_t>{})
                                  : group_medians(s.med_gids, s.med_f64, counts_, less_nan_last);
    s.med_gids = {};
    s.med_i64  = {};
    s.med_f64  = {};
    return out;
}

void HashAggOperator::make_output_() {
    const std::size_t groups = counts_.size();
    if (group_col_) {
        std::vector<int64_t> keys(groups);
        for (std::uint32_t g = 0; g < groups; ++g)
            keys[g] = index_.key_of(g);
        out_vecs_.push_back(std::make_unique<Int64Vector>(keys));
    }
    for (State& s : states_) {
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
                out_vecs_.push_back(std::make_unique<Float64Vector>(medians_(s)));
                break;
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
