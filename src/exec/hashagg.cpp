#include "exec/hashagg.h"

#include <algorithm>

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
        const auto& key_vec = static_cast<const Int64Vector&>(batch->column(group_col_));
        const auto& agg_vec = static_cast<const Float64Vector&>(batch->column(agg_col_));
        auto        keys    = key_vec.data();
        auto        vals    = agg_vec.data();

        if (batch->sel()) {
            for (std::uint32_t i : *batch->sel()) {
                auto& it = hash_.find_or_insert(keys[i]);
                it.sum += vals[i];
                it.count += 1;
            }
        } else {
            std::uint32_t end = batch->offset() + batch->size();
            for (std::uint32_t i = batch->offset(); i < end; ++i) {
                auto& it = hash_.find_or_insert(keys[i]);
                it.sum += vals[i];
                it.count += 1;
            }
        }
    }
    // Done absorbing input. Materialize the hash table into owned output columns.
    out_keys_.reserve(hash_.size());
    out_sums_.reserve(hash_.size());
    out_counts_.reserve(hash_.size());

    hash_.for_each([&](int64_t k, const AggHashTable::Value& v) {
        out_keys_.push_back(k);
        out_sums_.push_back(v.sum);
        out_counts_.push_back(v.count);
    });

    out_key_vec_   = std::make_unique<Int64Vector>(std::move(out_keys_));
    out_sum_vec_   = std::make_unique<Float64Vector>(std::move(out_sums_));
    out_count_vec_ = std::make_unique<Int64Vector>(std::move(out_counts_));

    built_ = true;
}

std::optional<Batch> HashAggOperator::next() {
    if (!built_) build_();

    // Emit a batch of up to emit_batch_size_ rows starting at emit_cursor_.

    std::size_t total = out_key_vec_->size();
    if (emit_cursor_ >= total) return std::nullopt;

    std::size_t batch_size =
        std::min(static_cast<std::size_t>(emit_batch_size_), total - emit_cursor_);

    std::vector<const Vector*> cols = {out_key_vec_.get(), out_sum_vec_.get(),
                                       out_count_vec_.get()};

    auto offset = static_cast<std::uint32_t>(emit_cursor_);
    emit_cursor_ += batch_size;
    return Batch{std::move(cols), offset, static_cast<std::uint32_t>(batch_size)};
}

}  // namespace wb