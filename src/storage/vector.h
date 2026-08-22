// src/storage/vector.h
#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "storage/schema.h"

namespace wb {

class Vector {
   public:
    virtual ~Vector() noexcept       = default;
    virtual DataType    type() const = 0;
    virtual std::size_t size() const = 0;
};

class Int64Vector final : public Vector {
   public:
    Int64Vector()                    = default;
    ~Int64Vector() noexcept override = default;
    explicit Int64Vector(std::vector<int64_t> values) : values_(std::move(values)) {}

    DataType    type() const override { return DataType::Int64; }
    std::size_t size() const override { return values_.size(); }

    std::span<const int64_t> data() const { return values_; }

    void append(int64_t value) { values_.push_back(value); }

    void reset(std::vector<int64_t> values) { values_ = std::move(values); }

   private:
    std::vector<int64_t> values_;
};

class Float64Vector final : public Vector {
   public:
    Float64Vector()                    = default;
    ~Float64Vector() noexcept override = default;
    explicit Float64Vector(std::vector<double> values) : values_(std::move(values)) {}

    DataType    type() const override { return DataType::Float64; }
    std::size_t size() const override { return values_.size(); }

    std::span<const double> data() const { return values_; }

    void append(double value) { values_.push_back(value); }

    void reset(std::vector<double> values) { values_ = std::move(values); }

   private:
    std::vector<double> values_;
};

class StringVector final : public Vector {
   public:
    StringVector()                    = default;
    ~StringVector() noexcept override = default;
    explicit StringVector(std::vector<std::string> values) : values_(std::move(values)) {}

    DataType    type() const override { return DataType::String; }
    std::size_t size() const override { return values_.size(); }

    const std::vector<std::string>& data() const { return values_; }

    void append(std::string value) { values_.push_back(std::move(value)); }

    void reset(std::vector<std::string> values) { values_ = std::move(values); }

   private:
    std::vector<std::string> values_;
};

}  // namespace wb