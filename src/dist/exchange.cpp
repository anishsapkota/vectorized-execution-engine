#include "dist/exchange.h"

#include <stdexcept>

namespace wb {

namespace {

// Rows per chunk before a sender flushes. Matches the scan's default batch size, so a chunk
// round-trips into roughly one batch on the receiving side.
constexpr std::uint32_t kChunkRows = 4096;

class ExchangeSourceOperator final : public Operator {
   public:
    explicit ExchangeSourceOperator(std::vector<std::shared_ptr<const Chunk>> chunks)
        : chunks_(std::move(chunks)) {}

    std::optional<Batch> next() override {
        while (next_chunk_ < chunks_.size()) {
            const Chunk& chunk = *chunks_[next_chunk_++];
            if (chunk.num_rows() > 0) return chunk.as_batch();
        }
        return std::nullopt;
    }

   private:
    // Holding every chunk until destruction keeps each emitted batch valid for as long as the
    // Operator contract requires, and longer.
    std::vector<std::shared_ptr<const Chunk>> chunks_;
    std::size_t                               next_chunk_ = 0;
};

}  // namespace

Exchange::Exchange(Cluster& cluster) : cluster_(cluster) {
    inboxes_.reserve(cluster_.num_slices());
    for (std::size_t slice = 0; slice < cluster_.num_slices(); ++slice)
        inboxes_.push_back(std::make_unique<Inbox>());
}

void Exchange::deliver(std::size_t src_slice, std::size_t dst_slice,
                       std::shared_ptr<const Chunk> chunk) {
    cluster_.network().record(cluster_.node_of(src_slice), cluster_.node_of(dst_slice),
                              chunk->wire_bytes(), chunk->num_rows());
    Inbox&           inbox = *inboxes_[dst_slice];
    std::scoped_lock lock{inbox.mutex};
    inbox.chunks.push_back(std::move(chunk));
}

void Exchange::deliver_to_all(std::size_t src_slice, const std::shared_ptr<const Chunk>& chunk) {
    const std::size_t src_node = cluster_.node_of(src_slice);
    for (std::size_t node = 0; node < cluster_.num_nodes(); ++node) {
        cluster_.network().record(src_node, node, chunk->wire_bytes(), chunk->num_rows());
    }
    for (std::size_t dst = 0; dst < cluster_.num_slices(); ++dst) {
        Inbox&           inbox = *inboxes_[dst];
        std::scoped_lock lock{inbox.mutex};
        inbox.chunks.push_back(chunk);
    }
}

void Exchange::redistribute(Operator& input, std::size_t src_slice, std::size_t key_col) {
    const std::size_t num_slices = cluster_.num_slices();

    // Created from the first batch's column types; per destination slice.
    std::vector<ChunkBuilder> builders;
    // Reused across batches so routing allocates only while capacity grows.
    std::vector<std::vector<std::uint32_t>> rows_for_slice(num_slices);

    while (auto batch = input.next()) {
        if (builders.empty()) {
            if (batch->column(key_col).type() != DataType::Int64) {
                throw std::invalid_argument("redistribute: key column must be int64");
            }
            const auto types = column_types(*batch);
            builders.reserve(num_slices);
            for (std::size_t dst = 0; dst < num_slices; ++dst)
                builders.emplace_back(types);
        }

        for (auto& rows : rows_for_slice)
            rows.clear();
        static_cast<const Int64Vector&>(batch->column(key_col)).visit([&](auto codes, int64_t base) {
            for_each_live_row(*batch, [&](std::uint32_t row) {
                const int64_t key = Int64Vector::decode(base, codes[row]);
                rows_for_slice[slice_for_key(key, num_slices)].push_back(row);
            });
        });

        for (std::size_t dst = 0; dst < num_slices; ++dst) {
            if (rows_for_slice[dst].empty()) continue;
            builders[dst].append_rows(*batch, rows_for_slice[dst]);
            if (builders[dst].num_rows() >= kChunkRows) {
                deliver(src_slice, dst, builders[dst].seal());
            }
        }
    }

    for (std::size_t dst = 0; dst < builders.size(); ++dst) {
        if (builders[dst].num_rows() > 0) deliver(src_slice, dst, builders[dst].seal());
    }
}

void Exchange::broadcast(Operator& input, std::size_t src_slice) {
    std::optional<ChunkBuilder> builder;
    std::vector<std::uint32_t>  live_rows;

    while (auto batch = input.next()) {
        if (!builder) builder.emplace(column_types(*batch));

        live_rows.clear();
        for_each_live_row(*batch, [&](std::uint32_t row) { live_rows.push_back(row); });
        builder->append_rows(*batch, live_rows);
        if (builder->num_rows() >= kChunkRows) deliver_to_all(src_slice, builder->seal());
    }

    if (builder && builder->num_rows() > 0) deliver_to_all(src_slice, builder->seal());
}

std::unique_ptr<Operator> Exchange::receive(std::size_t dst_slice) {
    Inbox&                                    inbox = *inboxes_[dst_slice];
    std::vector<std::shared_ptr<const Chunk>> chunks;
    {
        std::scoped_lock lock{inbox.mutex};
        chunks.swap(inbox.chunks);
    }
    return std::make_unique<ExchangeSourceOperator>(std::move(chunks));
}

}  // namespace wb
