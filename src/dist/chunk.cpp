#include "dist/chunk.h"

#include <stdexcept>
#include <type_traits>

namespace wb {

Chunk::Chunk(std::vector<std::unique_ptr<Vector>> columns, std::uint32_t num_rows,
             std::uint64_t wire_bytes)
    : columns_(std::move(columns)),
      num_rows_(num_rows),
      wire_bytes_(wire_bytes) {
    views_.reserve(columns_.size());
    for (const auto& column : columns_)
        views_.push_back(column.get());
}

Batch Chunk::as_batch() const { return Batch{views_, 0, num_rows_}; }

std::vector<DataType> column_types(const Batch& batch) {
    std::vector<DataType> types;
    types.reserve(batch.num_columns());
    for (std::size_t i = 0; i < batch.num_columns(); ++i)
        types.push_back(batch.column(i).type());
    return types;
}

ChunkBuilder::ChunkBuilder(std::vector<DataType> column_types)
    : column_types_(std::move(column_types)) {
    start_empty_columns_();
}

void ChunkBuilder::start_empty_columns_() {
    buffers_.clear();
    for (DataType type : column_types_) {
        switch (type) {
            case DataType::Int64:
                buffers_.emplace_back(std::vector<int64_t>{});
                break;
            case DataType::Float64:
                buffers_.emplace_back(std::vector<double>{});
                break;
            case DataType::String:
                buffers_.emplace_back(std::vector<std::string>{});
                break;
        }
    }
    num_rows_   = 0;
    wire_bytes_ = 0;
}

void ChunkBuilder::append_rows(const Batch& batch, std::span<const std::uint32_t> rows) {
    if (batch.num_columns() != column_types_.size()) {
        throw std::invalid_argument("ChunkBuilder: batch has a different number of columns");
    }

    for (std::size_t c = 0; c < column_types_.size(); ++c) {
        const Vector& source = batch.column(c);
        switch (column_types_[c]) {
            case DataType::Int64: {
                auto& target = std::get<std::vector<int64_t>>(buffers_[c]);
                static_cast<const Int64Vector&>(source).visit([&](auto codes, int64_t base) {
                    for (std::uint32_t row : rows)
                        target.push_back(Int64Vector::decode(base, codes[row]));
                });
                wire_bytes_ += rows.size() * sizeof(int64_t);
                break;
            }
            case DataType::Float64: {
                auto& target = std::get<std::vector<double>>(buffers_[c]);
                static_cast<const Float64Vector&>(source).visit([&](auto values) {
                    for (std::uint32_t row : rows)
                        target.push_back(static_cast<double>(values[row]));
                });
                wire_bytes_ += rows.size() * sizeof(double);
                break;
            }
            case DataType::String: {
                const auto& values = static_cast<const StringVector&>(source).data();
                auto&       target = std::get<std::vector<std::string>>(buffers_[c]);
                for (std::uint32_t row : rows) {
                    target.push_back(values[row]);
                    wire_bytes_ += sizeof(std::uint32_t) + values[row].size();
                }
                break;
            }
        }
    }
    num_rows_ += static_cast<std::uint32_t>(rows.size());
}

std::shared_ptr<const Chunk> ChunkBuilder::seal() {
    std::vector<std::unique_ptr<Vector>> columns;
    columns.reserve(buffers_.size());
    for (auto& buffer : buffers_) {
        std::visit(
            [&](auto& values) {
                using T = typename std::decay_t<decltype(values)>::value_type;
                if constexpr (std::is_same_v<T, int64_t>) {
                    columns.push_back(std::make_unique<Int64Vector>(values));
                } else if constexpr (std::is_same_v<T, double>) {
                    columns.push_back(std::make_unique<Float64Vector>(std::move(values)));
                } else {
                    columns.push_back(std::make_unique<StringVector>(std::move(values)));
                }
            },
            buffer);
    }
    auto chunk = std::make_shared<const Chunk>(std::move(columns), num_rows_, wire_bytes_);
    start_empty_columns_();
    return chunk;
}

}  // namespace wb
