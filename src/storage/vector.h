// src/storage/vector.h
#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "storage/schema.h"

namespace wb {

// Columns are immutable once constructed: SegmentBuilder buffers raw values and builds the
// encoded column in seal().
class Vector {
   public:
    virtual ~Vector() noexcept       = default;
    virtual DataType    type() const = 0;
    virtual std::size_t size() const = 0;
    // Heap bytes owned by this column (allocated capacity, not just size).
    virtual std::size_t memory_bytes() const = 0;
};

// Untyped view of a column's storage, for code that cannot use visit() (the codegen host binds
// these pointers into generated code).
struct Int64Raw {
    const void*  codes;  // width-byte unsigned codes; value = base + code
    int64_t      base;
    std::uint8_t width;  // 1, 2, 4 or 8
};

struct Float64Raw {
    const void* values;  // float if narrow, else double
    bool        narrow;
};

// Frame-of-reference encoded int64 column. Stores `value - base` (base = column minimum) in the
// narrowest unsigned width that holds the column's range: 1, 2, 4 or 8 bytes per value.
// Kernels read through visit(), which hands them the typed code array once per batch so the
// inner loop is a plain widen-and-add.
class Int64Vector final : public Vector {
   public:
    Int64Vector() = default;
    explicit Int64Vector(std::span<const int64_t> values);

    DataType    type() const override { return DataType::Int64; }
    std::size_t size() const override { return size_; }
    std::size_t memory_bytes() const override;

    int64_t     base() const { return base_; }
    std::size_t width_bytes() const;

    template <typename Code>
    static int64_t decode(int64_t base, Code code) {
        // Unsigned add: wraps instead of overflowing when the range needs all 64 bits.
        return static_cast<int64_t>(static_cast<uint64_t>(base) + code);
    }

    // Calls f(std::span<const UInt> codes, int64_t base); value i is decode(base, codes[i]).
    template <typename F>
    void visit(F&& f) const {
        std::visit([&](const auto& codes) { f(std::span{codes}, base_); }, codes_);
    }

    // Scalar access, for cold paths. Hot loops should use visit().
    int64_t value(std::size_t i) const;

    Int64Raw raw() const;

    // For operator-owned output vectors only: turn this into an unencoded column of n values
    // (width 8, base 0, so decoding is the identity) and return its storage for writing. Keeps
    // the existing capacity, so refilling a batch-sized output column allocates nothing.
    int64_t* reset_plain(std::size_t n);

   private:
    std::variant<std::vector<uint8_t>, std::vector<uint16_t>, std::vector<uint32_t>,
                 std::vector<uint64_t>>
                codes_;
    int64_t     base_ = 0;
    std::size_t size_ = 0;
};

// Float64 column, stored as double or, when narrow is set, as float. narrow is lossy: it is
// only exact for inputs with at most 6 significant decimal digits (FLT_DIG). Reads always
// widen to double.
class Float64Vector final : public Vector {
   public:
    Float64Vector() = default;
    explicit Float64Vector(std::vector<double> values, bool narrow = false);

    DataType    type() const override { return DataType::Float64; }
    std::size_t size() const override;
    std::size_t memory_bytes() const override;

    bool narrow() const { return std::holds_alternative<std::vector<float>>(values_); }

    // Calls f(std::span<const double>) or f(std::span<const float>).
    template <typename F>
    void visit(F&& f) const {
        std::visit([&](const auto& values) { f(std::span{values}); }, values_);
    }

    // Scalar access, for cold paths. Hot loops should use visit().
    double value(std::size_t i) const;

    Float64Raw raw() const;

    // For operator-owned output vectors only: turn this into a double column of n values and
    // return its storage for writing. Keeps the existing capacity.
    double* reset(std::size_t n);

   private:
    std::variant<std::vector<double>, std::vector<float>> values_;
};

class StringVector final : public Vector {
   public:
    StringVector()                    = default;
    ~StringVector() noexcept override = default;
    explicit StringVector(std::vector<std::string> values) : values_(std::move(values)) {}

    DataType    type() const override { return DataType::String; }
    std::size_t size() const override { return values_.size(); }
    std::size_t memory_bytes() const override {
        std::size_t bytes = values_.capacity() * sizeof(std::string);
        for (const auto& s : values_)
            if (s.capacity() > std::string{}.capacity()) bytes += s.capacity() + 1;  // spilled past SSO
        return bytes;
    }

    const std::vector<std::string>& data() const { return values_; }

   private:
    std::vector<std::string> values_;
};

}  // namespace wb
