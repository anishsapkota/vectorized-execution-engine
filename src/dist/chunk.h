#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <variant>
#include <vector>

#include "exec/batch.h"
#include "storage/schema.h"
#include "storage/vector.h"

namespace wb {

// An owned, immutable set of columns: the unit that travels between slices.
// A Batch is a borrowed view that dies at the next next() call, so anything that has to outlive
// the producing operator (exchange payloads, a hash join's build side) must be copied into one.
class Chunk {
   public:
    Chunk(std::vector<std::unique_ptr<Vector>> columns, std::uint32_t num_rows,
          std::uint64_t wire_bytes);

    std::uint32_t num_rows() const { return num_rows_; }
    std::size_t   num_columns() const { return columns_.size(); }
    const Vector& column(std::size_t i) const { return *columns_[i]; }

    // Serialized size: 8 bytes per int64/float64 value, 4-byte length prefix + payload per
    // string. What NetworkStats charges when the chunk moves.
    std::uint64_t wire_bytes() const { return wire_bytes_; }

    // A batch viewing every row of this chunk. Valid while the chunk is alive.
    Batch as_batch() const;

   private:
    std::vector<std::unique_ptr<Vector>> columns_;
    std::vector<const Vector*>           views_;  // borrowed by as_batch()
    std::uint32_t                        num_rows_;
    std::uint64_t                        wire_bytes_;
};

std::vector<DataType> column_types(const Batch& batch);

// Copies selected rows out of batches into owned columns. Copies column-at-a-time rather than
// row-at-a-time so the type dispatch happens once per column per call, not once per value.
// Columns are immutable, so rows accumulate in raw buffers and are encoded in seal().
class ChunkBuilder {
   public:
    explicit ChunkBuilder(std::vector<DataType> column_types);

    // rows are absolute indices into batch's columns (as produced by for_each_live_row).
    void append_rows(const Batch& batch, std::span<const std::uint32_t> rows);

    std::uint32_t num_rows() const { return num_rows_; }

    // Hands over the accumulated rows and leaves the builder empty and reusable.
    std::shared_ptr<const Chunk> seal();

   private:
    using ColumnBuffer =
        std::variant<std::vector<int64_t>, std::vector<double>, std::vector<std::string>>;

    std::vector<DataType>     column_types_;
    std::vector<ColumnBuffer> buffers_;
    std::uint32_t             num_rows_   = 0;
    std::uint64_t             wire_bytes_ = 0;

    void start_empty_columns_();
};

}  // namespace wb
