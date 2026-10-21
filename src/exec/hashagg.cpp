#include "exec/hashagg.h"

#include <algorithm>

#include "exec/agg_kernel.h"
#include "exec/batch.h"
#include "storage/vector.h"

namespace wb {

HashAggOperator::HashAggOperator(std::unique_ptr<Operator> child, std::size_t group_col_idx,
                                 std::size_t agg_col_idx)
    : child_(std::move(child)),
      group_col_(group_col_idx),
      agg_col_(agg_col_idx) {}

void HashAggOperator::emit_init_() {}  // merged into build_()

void HashAggOperator::build_() {
    while (auto batch = child_->next()) {
        const auto& keys = static_cast<const Int64Vector&>(batch->column(group_col_));
        const auto& vals = static_cast<const Float64Vector&>(batch->column(agg_col_));

        if (batch->has_sel()) {
            accumulate_sum_count(hash_, keys, vals, [&](auto f) {
                for (std::uint32_t i : batch->sel())
                    f(i);
            });
        } else {
            accumulate_sum_count(hash_, keys, vals, [&](auto f) {
                std::uint32_t end = batch->offset() + batch->size();
                for (std::uint32_t i = batch->offset(); i < end; ++i)
                    f(i);
            });
        }
    }
    // Done absorbing input. Materialize the hash table into owned output columns.
    std::vector<int64_t> out_keys, out_counts;
    std::vector<double>  out_sums;
    out_keys.reserve(hash_.size());
    out_sums.reserve(hash_.size());
    out_counts.reserve(hash_.size());

    hash_.for_each([&](int64_t k, const AggHashTable::Value& v) {
        out_keys.push_back(k);
        out_sums.push_back(v.sum);
        out_counts.push_back(v.count);
    });

    out_key_vec_   = std::make_unique<Int64Vector>(out_keys);
    out_sum_vec_   = std::make_unique<Float64Vector>(std::move(out_sums));
    out_count_vec_ = std::make_unique<Int64Vector>(out_counts);
    out_cols_      = {out_key_vec_.get(), out_sum_vec_.get(), out_count_vec_.get()};

    built_ = true;
}

std::optional<Batch> HashAggOperator::next() {
    if (!built_) build_();

    // Emit a batch of up to emit_batch_size_ rows starting at emit_cursor_.

    std::size_t total = out_key_vec_->size();
    if (emit_cursor_ >= total) return std::nullopt;

    std::size_t batch_size =
        std::min(static_cast<std::size_t>(emit_batch_size_), total - emit_cursor_);

    auto offset = static_cast<std::uint32_t>(emit_cursor_);
    emit_cursor_ += batch_size;
    return Batch{out_cols_, offset, static_cast<std::uint32_t>(batch_size)};
}

}  // namespace wb