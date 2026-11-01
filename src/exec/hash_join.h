#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "dist/chunk.h"
#include "exec/operator.h"

namespace wb {

// Inner equi-join on one int64 key.
//
//   SELECT probe.*, build.* FROM probe JOIN build ON probe[probe_key_col] = build[build_key_col]
//
// Contract:
//   - On the first next(), drain the build child completely and index it by key. Then stream
//     the probe child: each probe batch produces zero or more output batches.
//   - Output columns: every probe column in order, then every build column in order.
//   - Duplicate keys on either side are allowed; output is the full cross product per key.
//   - Probe batches may carry a selection vector (e.g. a FilterOperator below).
//   - Output batches follow the Operator contract: valid until the next next() call.
//
// Design:
//   - Build rows are copied into one owned Chunk, sorted by key, so all rows of a key are
//     contiguous. The index maps key -> (first row, count): 16 B per distinct key, open
//     addressing, linear probing.
//   - Output is gathered into owned chunks of at most kOutputRows rows. An output batch never
//     spans two probe batches, because pulling the next probe batch invalidates the current one.
class HashJoinOperator final : public Operator {
   public:
    HashJoinOperator(std::unique_ptr<Operator> probe, std::unique_ptr<Operator> build,
                     std::size_t probe_key_col, std::size_t build_key_col);

    std::optional<Batch> next() override;

    static constexpr std::uint32_t kOutputRows = 4096;

   private:
    struct Bucket {
        int64_t       key   = 0;
        std::uint32_t first = 0;
        std::uint32_t count = 0;  // 0 marks an empty bucket
    };

    void          build_index_();
    const Bucket* find_(int64_t key) const;
    std::size_t   bucket_of_(int64_t key) const;
    // Pulls probe batches until one has a live row. Returns false when the probe side is done.
    bool load_probe_batch_();
    void init_output_();
    static void gather_(Vector& out, const Vector& src, const std::vector<std::uint32_t>& rows);

    std::unique_ptr<Operator> probe_;
    std::unique_ptr<Operator> build_;
    std::size_t               probe_key_col_;
    std::size_t               build_key_col_;

    // Build side.
    bool                         built_      = false;
    bool                         probe_done_ = false;
    std::shared_ptr<const Chunk> build_rows_;  // sorted by key
    std::vector<Bucket>          buckets_;     // power-of-two size; empty if build side is empty
    unsigned                     bucket_shift_ = 64;

    // Current probe batch and position within it.
    std::optional<Batch>       probe_batch_;
    std::vector<std::uint32_t> probe_rows_;  // live rows of probe_batch_
    std::vector<int64_t>       probe_keys_;  // decoded key per entry of probe_rows_
    std::size_t                probe_pos_ = 0;
    std::uint32_t              match_pos_ = 0;  // build matches already emitted for probe_pos_

    // Output.
    // Output columns owned by the operator and refilled in place by every next() (Int64/Float64
    // joins). A join with a String column falls back to sealing a fresh chunk per batch.
    std::vector<std::unique_ptr<Vector>> out_vecs_;
    bool                                 out_reusable_ = true;
    std::optional<ChunkBuilder>          out_probe_builder_;
    std::optional<ChunkBuilder>          out_build_builder_;
    std::shared_ptr<const Chunk>         out_probe_;
    std::shared_ptr<const Chunk>         out_build_;
    std::vector<const Vector*>           out_cols_;
    std::vector<std::uint32_t>   out_probe_idx_;
    std::vector<std::uint32_t>   out_build_idx_;
};

}  // namespace wb
