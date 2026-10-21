#include "exec/hashagg.h"

#include <algorithm>
#include <cstring>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>

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

// Float-stored column values without widening (MEDIAN keeps the storage type).
template <typename Rows, typename F>
void for_each_float(const Vector& col, const Rows& rows, F&& f) {
    static_cast<const Float64Vector&>(col).visit([&](auto values) {
        if constexpr (std::is_same_v<typename decltype(values)::element_type, const float>) {
            for (std::size_t k = 0; k < rows.size(); ++k)
                f(k, values[rows[k]]);
        } else {
            throw std::logic_error("HashAggOperator: MEDIAN column changed storage between batches");
        }
    });
}

template <typename T>
T& at(std::byte* state, std::size_t offset) {
    return *reinterpret_cast<T*>(state + offset);
}

template <typename T>
void update_min(T& m, T v) {
    if (v < m || v != v) m = v;
}

template <typename T>
void update_max(T& m, T v) {
    if (v > m || v != v) m = v;
}

bool is_int128(const AggSpec& s) {
    return (s.kind == AggKind::Sum || s.kind == AggKind::Avg) && s.type == DataType::Int64;
}

}  // namespace

HashAggOperator::HashAggOperator(std::unique_ptr<Operator> child, std::optional<std::size_t> group_col,
                                 std::vector<AggSpec> aggs)
    : child_(std::move(child)),
      group_col_(group_col) {
    for (const AggSpec& spec : aggs)
        fields_.push_back(Field{spec, 0, false, {}});

    // State layout: __int128 fields (16-aligned), count, 8-byte fields, MEDIAN handles.
    std::size_t offset = 0;
    for (Field& f : fields_) {
        if (!is_int128(f.spec)) continue;
        f.offset = offset;
        offset += 16;
        state_align_ = 16;
    }
    count_offset_ = offset;
    offset += 8;
    for (Field& f : fields_) {
        if (f.spec.kind == AggKind::Count || f.spec.kind == AggKind::Median || is_int128(f.spec)) continue;
        f.offset = offset;
        offset += 8;
    }
    for (Field& f : fields_) {
        if (f.spec.kind != AggKind::Median) continue;
        f.offset = offset;
        offset += sizeof(MedianHandle);
    }
    state_bytes_ = offset;

    if (group_col_) {
        slots_.emplace(state_bytes_, state_align_);
    } else {
        global_state_raw_.resize(state_bytes_ + 16);
        const auto addr = reinterpret_cast<std::uintptr_t>(global_state_raw_.data());
        global_state_   = global_state_raw_.data() + ((16 - addr % 16) % 16);
    }
}

HashAggOperator::HashAggOperator(std::unique_ptr<Operator> child, std::size_t group_col, std::size_t agg_col)
    : HashAggOperator(std::move(child), group_col,
                      {AggSpec{AggKind::Sum, agg_col, DataType::Float64}, AggSpec{AggKind::Count}}) {}

void HashAggOperator::bind_median_stores_(const Batch& batch) {
    for (Field& f : fields_) {
        if (f.spec.kind != AggKind::Median) continue;
        if (f.spec.type == DataType::Int64) {
            f.store.emplace<MedianStore<int64_t>>();
        } else {
            f.narrow = static_cast<const Float64Vector&>(batch.column(f.spec.col)).narrow();
            if (f.narrow) f.store.emplace<MedianStore<float>>();
            else f.store.emplace<MedianStore<double>>();
        }
    }
    stores_bound_ = true;
}

void HashAggOperator::init_state_(std::byte* state, const Batch& batch, std::uint32_t row) {
    at<int64_t>(state, count_offset_) = 0;
    for (const Field& f : fields_) {
        const AggSpec& spec = f.spec;
        switch (spec.kind) {
            case AggKind::Count:
                break;
            case AggKind::Sum:
            case AggKind::Avg:
                if (is_int128(spec)) at<__int128>(state, f.offset) = 0;
                else at<double>(state, f.offset) = 0.0;  // +0.0; the first row is added like every other
                break;
            case AggKind::Min:
            case AggKind::Max:  // start at the group's first value
                if (spec.type == DataType::Int64) {
                    at<int64_t>(state, f.offset) = static_cast<const Int64Vector&>(batch.column(spec.col)).value(row);
                } else {
                    at<double>(state, f.offset) = static_cast<const Float64Vector&>(batch.column(spec.col)).value(row);
                }
                break;
            case AggKind::Median:
                new (state + f.offset) MedianHandle{};
                break;
        }
    }
}

void HashAggOperator::consume_(const Batch& batch) {
    if (!stores_bound_) bind_median_stores_(batch);
    if (batch.has_sel()) consume_rows_(batch, SelRows{batch.sel()});
    else consume_rows_(batch, RangeRows{batch.offset(), batch.size()});
}

template <typename Rows>
void HashAggOperator::consume_rows_(const Batch& batch, const Rows& rows) {
    if (!group_col_) return consume_global_(batch, rows);

    // Pass 1: each live row's state block (creating new groups) and counts. Reserving first means
    // the table cannot rehash during the batch, so every pointer stays valid until pass 2 ends.
    slots_->reserve(slots_->size() + rows.size());
    ptrs_.resize(rows.size());
    for_each_int(batch.column(*group_col_), rows, [&](std::size_t k, int64_t key) {
        bool       fresh = false;
        std::byte* state = slots_->find_or_insert(key, fresh);
        if (fresh) init_state_(state, batch, rows[k]);
        ++at<int64_t>(state, count_offset_);
        ptrs_[k] = state;
    });

    // Pass 2: one loop per aggregate, updating the state in place.
    const auto& ptrs = ptrs_;
    for (Field& f : fields_) {
        const AggSpec&    spec = f.spec;
        const std::size_t off  = f.offset;
        if (spec.kind == AggKind::Count) continue;
        const Vector& col    = batch.column(spec.col);
        const bool    is_int = spec.type == DataType::Int64;
        switch (spec.kind) {
            case AggKind::Sum:
            case AggKind::Avg:
                if (is_int) for_each_int(col, rows, [&](std::size_t k, int64_t v) { at<__int128>(ptrs[k], off) += v; });
                else for_each_double(col, rows, [&](std::size_t k, double v) { at<double>(ptrs[k], off) += v; });
                break;
            case AggKind::Min:
                if (is_int) for_each_int(col, rows, [&](std::size_t k, int64_t v) { update_min(at<int64_t>(ptrs[k], off), v); });
                else for_each_double(col, rows, [&](std::size_t k, double v) { update_min(at<double>(ptrs[k], off), v); });
                break;
            case AggKind::Max:
                if (is_int) for_each_int(col, rows, [&](std::size_t k, int64_t v) { update_max(at<int64_t>(ptrs[k], off), v); });
                else for_each_double(col, rows, [&](std::size_t k, double v) { update_max(at<double>(ptrs[k], off), v); });
                break;
            case AggKind::Median:
                if (auto* s = std::get_if<MedianStore<int64_t>>(&f.store)) {
                    for_each_int(col, rows, [&](std::size_t k, int64_t v) { s->append(at<MedianHandle>(ptrs[k], off), v); });
                } else if (auto* d = std::get_if<MedianStore<double>>(&f.store)) {
                    if (static_cast<const Float64Vector&>(col).narrow()) {
                        throw std::logic_error("HashAggOperator: MEDIAN column changed storage between batches");
                    }
                    for_each_double(col, rows, [&](std::size_t k, double v) { d->append(at<MedianHandle>(ptrs[k], off), v); });
                } else if (auto* fl = std::get_if<MedianStore<float>>(&f.store)) {
                    for_each_float(col, rows, [&](std::size_t k, float v) { fl->append(at<MedianHandle>(ptrs[k], off), v); });
                }
                break;
            case AggKind::Count:
                break;
        }
    }
}

// Global aggregate: one state block. Each aggregate accumulates in a local for the batch (a
// register, not a memory read-modify-write per row), in row order like the grouped path.
template <typename Rows>
void HashAggOperator::consume_global_(const Batch& batch, const Rows& rows) {
    if (rows.size() == 0) return;
    std::byte* state = global_state_;
    if (!global_started_) {
        init_state_(state, batch, rows[0]);
        global_started_ = true;
    }
    at<int64_t>(state, count_offset_) += static_cast<int64_t>(rows.size());

    for (Field& f : fields_) {
        const AggSpec& spec = f.spec;
        if (spec.kind == AggKind::Count) continue;
        const Vector& col    = batch.column(spec.col);
        const bool    is_int = spec.type == DataType::Int64;
        switch (spec.kind) {
            case AggKind::Sum:
            case AggKind::Avg:
                if (is_int) {
                    __int128 acc = at<__int128>(state, f.offset);
                    for_each_int(col, rows, [&](std::size_t, int64_t v) { acc += v; });
                    at<__int128>(state, f.offset) = acc;
                } else {
                    double acc = at<double>(state, f.offset);
                    for_each_double(col, rows, [&](std::size_t, double v) { acc += v; });
                    at<double>(state, f.offset) = acc;
                }
                break;
            case AggKind::Min:
            case AggKind::Max: {
                const bool is_min = spec.kind == AggKind::Min;
                if (is_int) {
                    int64_t m = at<int64_t>(state, f.offset);
                    for_each_int(col, rows, [&](std::size_t, int64_t v) { is_min ? update_min(m, v) : update_max(m, v); });
                    at<int64_t>(state, f.offset) = m;
                } else {
                    double m = at<double>(state, f.offset);
                    for_each_double(col, rows, [&](std::size_t, double v) { is_min ? update_min(m, v) : update_max(m, v); });
                    at<double>(state, f.offset) = m;
                }
                break;
            }
            case AggKind::Median: {
                MedianHandle& h = at<MedianHandle>(state, f.offset);
                if (auto* s = std::get_if<MedianStore<int64_t>>(&f.store)) {
                    for_each_int(col, rows, [&](std::size_t, int64_t v) { s->append(h, v); });
                } else if (auto* d = std::get_if<MedianStore<double>>(&f.store)) {
                    if (static_cast<const Float64Vector&>(col).narrow()) {
                        throw std::logic_error("HashAggOperator: MEDIAN column changed storage between batches");
                    }
                    for_each_double(col, rows, [&](std::size_t, double v) { d->append(h, v); });
                } else if (auto* fl = std::get_if<MedianStore<float>>(&f.store)) {
                    for_each_float(col, rows, [&](std::size_t, float v) { fl->append(h, v); });
                }
                break;
            }
            case AggKind::Count:
                break;
        }
    }
}

void HashAggOperator::make_output_() {
    const std::size_t groups = group_col_ ? slots_->size() : (global_started_ ? 1 : 0);
    auto              state  = [&](std::uint32_t g) { return group_col_ ? slots_->state_of(g) : global_state_; };
    auto              key    = [&](std::uint32_t g) {
        return group_col_ ? std::to_string(slots_->key_of(g)) : std::string{"global"};
    };
    out_rows_ = groups;

    if (group_col_) {
        std::vector<int64_t> keys(groups);
        for (std::uint32_t g = 0; g < groups; ++g)
            keys[g] = slots_->key_of(g);
        out_vecs_.push_back(std::make_unique<Int64Vector>(keys));
    }
    for (Field& f : fields_) {
        const AggSpec&    spec   = f.spec;
        const std::size_t off    = f.offset;
        const bool        is_int = spec.type == DataType::Int64;
        switch (spec.kind) {
            case AggKind::Count: {
                std::vector<int64_t> counts(groups);
                for (std::uint32_t g = 0; g < groups; ++g)
                    counts[g] = at<int64_t>(state(g), count_offset_);
                out_vecs_.push_back(std::make_unique<Int64Vector>(counts));
                break;
            }
            case AggKind::Sum:
                if (is_int) {
                    std::vector<int64_t> sums(groups);
                    for (std::uint32_t g = 0; g < groups; ++g) {
                        const __int128 s = at<__int128>(state(g), off);
                        if (s > INT64_MAX || s < INT64_MIN) throw std::overflow_error("SUM overflow in group " + key(g));
                        sums[g] = static_cast<int64_t>(s);
                    }
                    out_vecs_.push_back(std::make_unique<Int64Vector>(sums));
                } else {
                    std::vector<double> sums(groups);
                    for (std::uint32_t g = 0; g < groups; ++g)
                        sums[g] = at<double>(state(g), off);
                    out_vecs_.push_back(std::make_unique<Float64Vector>(std::move(sums)));
                }
                break;
            case AggKind::Avg: {
                std::vector<double> avg(groups);
                for (std::uint32_t g = 0; g < groups; ++g) {
                    const double n = static_cast<double>(at<int64_t>(state(g), count_offset_));
                    avg[g] = is_int ? static_cast<double>(at<__int128>(state(g), off)) / n : at<double>(state(g), off) / n;
                }
                out_vecs_.push_back(std::make_unique<Float64Vector>(std::move(avg)));
                break;
            }
            case AggKind::Min:
            case AggKind::Max:
                if (is_int) {
                    std::vector<int64_t> v(groups);
                    for (std::uint32_t g = 0; g < groups; ++g)
                        v[g] = at<int64_t>(state(g), off);
                    out_vecs_.push_back(std::make_unique<Int64Vector>(v));
                } else {
                    std::vector<double> v(groups);
                    for (std::uint32_t g = 0; g < groups; ++g)
                        v[g] = at<double>(state(g), off);
                    out_vecs_.push_back(std::make_unique<Float64Vector>(std::move(v)));
                }
                break;
            case AggKind::Median: {
                std::vector<double> med(groups);
                std::visit(
                    [&](auto& store) {
                        using S = std::decay_t<decltype(store)>;
                        if constexpr (!std::is_same_v<S, std::monostate>) {
                            if (!group_col_) {
                                if (groups) med[0] = store.finalize_single(at<MedianHandle>(state(0), off));
                                return;
                            }
                            std::vector<typename S::value_type> scratch;
                            for (std::uint32_t g = 0; g < groups; ++g)
                                med[g] = store.finalize_group(at<MedianHandle>(state(g), off), scratch);
                        }
                    },
                    f.store);
                out_vecs_.push_back(std::make_unique<Float64Vector>(std::move(med)));
                break;
            }
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
    if (emit_cursor_ >= out_rows_) return std::nullopt;
    const std::size_t batch_size = std::min(static_cast<std::size_t>(emit_batch_size_), out_rows_ - emit_cursor_);
    const auto        offset     = static_cast<std::uint32_t>(emit_cursor_);
    emit_cursor_ += batch_size;
    return Batch{out_cols_, offset, static_cast<std::uint32_t>(batch_size)};
}

}  // namespace wb
