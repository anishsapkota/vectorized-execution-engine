#include "exec/hash_join.h"

#include <algorithm>
#include <bit>
#include <numeric>
#include <stdexcept>

#include "storage/vector.h"

namespace wb {

namespace {

const Int64Vector& key_column(const Vector& col, const char* side) {
    if (col.type() != DataType::Int64) {
        throw std::invalid_argument(std::string{"HashJoinOperator: "} + side +
                                    " key column must be int64");
    }
    return static_cast<const Int64Vector&>(col);
}

}  // namespace

HashJoinOperator::HashJoinOperator(std::unique_ptr<Operator> probe, std::unique_ptr<Operator> build,
                                   std::size_t probe_key_col, std::size_t build_key_col)
    : probe_(std::move(probe)),
      build_(std::move(build)),
      probe_key_col_(probe_key_col),
      build_key_col_(build_key_col) {}

std::size_t HashJoinOperator::bucket_of_(int64_t key) const {
    // Same multiplicative hash as AggHashTable, but indexed by the top bits: the low bits of the
    // product depend only on the low bits of the key.
    const std::uint64_t h = static_cast<std::uint64_t>(key) * 0x9E3779B97F4A7C15ULL;
    return bucket_shift_ >= 64 ? 0 : static_cast<std::size_t>(h >> bucket_shift_);
}

const HashJoinOperator::Bucket* HashJoinOperator::find_(int64_t key) const {
    const std::size_t mask = buckets_.size() - 1;
    for (std::size_t i = bucket_of_(key);; i = (i + 1) & mask) {
        const Bucket& b = buckets_[i];
        if (b.count == 0) return nullptr;
        if (b.key == key) return &b;
    }
}

void HashJoinOperator::build_index_() {
    built_ = true;

    // 1. Copy every live build row into one owned chunk; batches die at the next next().
    std::optional<ChunkBuilder> gather;
    std::vector<std::uint32_t>  live;
    while (auto batch = build_->next()) {
        if (!gather) {
            key_column(batch->column(build_key_col_), "build");
            gather.emplace(column_types(*batch));
        }
        live.clear();
        for_each_live_row(*batch, [&](std::uint32_t row) { live.push_back(row); });
        gather->append_rows(*batch, live);
    }
    if (!gather || gather->num_rows() == 0) return;  // empty build side: no output

    const auto        unsorted = gather->seal();
    const Batch       all_rows = unsorted->as_batch();
    const std::size_t n        = unsorted->num_rows();

    // 2. Sort build rows by key so each key's rows are contiguous.
    std::vector<int64_t> keys(n);
    key_column(unsorted->column(build_key_col_), "build").visit([&](auto codes, int64_t base) {
        for (std::size_t i = 0; i < n; ++i)
            keys[i] = Int64Vector::decode(base, codes[i]);
    });
    std::vector<std::uint32_t> perm(n);
    std::iota(perm.begin(), perm.end(), 0u);
    std::stable_sort(perm.begin(), perm.end(),
                     [&](std::uint32_t a, std::uint32_t b) { return keys[a] < keys[b]; });

    const auto   types = column_types(all_rows);
    ChunkBuilder sorted{types};
    sorted.append_rows(all_rows, perm);
    build_rows_ = sorted.seal();
    out_build_builder_.emplace(types);

    // 3. Index each run of equal keys: load factor <= 0.5.
    std::size_t distinct = 0;
    for (std::size_t i = 0; i < n; ++i)
        distinct += (i == 0 || keys[perm[i]] != keys[perm[i - 1]]);
    const std::size_t capacity = std::bit_ceil(std::max<std::size_t>(2, distinct * 2));
    buckets_.assign(capacity, Bucket{});
    bucket_shift_ = 64 - static_cast<unsigned>(std::countr_zero(capacity));

    const std::size_t mask = capacity - 1;
    for (std::size_t start = 0; start < n;) {
        const int64_t key = keys[perm[start]];
        std::size_t   end = start + 1;
        while (end < n && keys[perm[end]] == key)
            ++end;
        std::size_t i = bucket_of_(key);
        while (buckets_[i].count != 0)
            i = (i + 1) & mask;
        buckets_[i] = {key, static_cast<std::uint32_t>(start), static_cast<std::uint32_t>(end - start)};
        start       = end;
    }
}

bool HashJoinOperator::load_probe_batch_() {
    if (probe_done_) return false;
    while ((probe_batch_ = probe_->next())) {
        const Int64Vector& keys = key_column(probe_batch_->column(probe_key_col_), "probe");
        if (!out_probe_builder_) out_probe_builder_.emplace(column_types(*probe_batch_));

        probe_rows_.clear();
        for_each_live_row(*probe_batch_, [&](std::uint32_t row) { probe_rows_.push_back(row); });
        probe_keys_.resize(probe_rows_.size());
        keys.visit([&](auto codes, int64_t base) {
            for (std::size_t i = 0; i < probe_rows_.size(); ++i)
                probe_keys_[i] = Int64Vector::decode(base, codes[probe_rows_[i]]);
        });
        probe_pos_ = 0;
        match_pos_ = 0;
        if (!probe_rows_.empty()) return true;
    }
    probe_done_ = true;
    probe_rows_.clear();
    probe_pos_ = 0;
    return false;
}

std::optional<Batch> HashJoinOperator::next() {
    if (!built_) build_index_();
    if (buckets_.empty()) return std::nullopt;

    out_probe_idx_.clear();
    out_build_idx_.clear();
    while (out_probe_idx_.size() < kOutputRows) {
        if (probe_pos_ == probe_rows_.size()) {
            // Emit what we have before pulling the next probe batch invalidates this one.
            if (!out_probe_idx_.empty() || !load_probe_batch_()) break;
        }
        const Bucket* match = find_(probe_keys_[probe_pos_]);
        if (!match) {
            ++probe_pos_;
            continue;
        }
        const auto room = static_cast<std::uint32_t>(kOutputRows - out_probe_idx_.size());
        const auto take = std::min(match->count - match_pos_, room);
        for (std::uint32_t k = 0; k < take; ++k) {
            out_probe_idx_.push_back(probe_rows_[probe_pos_]);
            out_build_idx_.push_back(match->first + match_pos_ + k);
        }
        match_pos_ += take;
        if (match_pos_ == match->count) {
            ++probe_pos_;
            match_pos_ = 0;
        }
    }
    if (out_probe_idx_.empty()) return std::nullopt;

    // Gather the matched rows into owned columns: probe columns, then build columns.
    out_probe_builder_->append_rows(*probe_batch_, out_probe_idx_);
    out_build_builder_->append_rows(build_rows_->as_batch(), out_build_idx_);
    out_probe_ = out_probe_builder_->seal();
    out_build_ = out_build_builder_->seal();

    out_cols_.clear();
    for (std::size_t c = 0; c < out_probe_->num_columns(); ++c)
        out_cols_.push_back(&out_probe_->column(c));
    for (std::size_t c = 0; c < out_build_->num_columns(); ++c)
        out_cols_.push_back(&out_build_->column(c));
    return Batch{out_cols_, 0, out_probe_->num_rows()};
}

}  // namespace wb
