#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

namespace wb {
// Single-purpose open-addressing hash table for int64 -> AggState aggregatio. Linear probing, power
// of two capacity, sentinel-key empty markers. Built for one job: high-throughput GROUP BY
// aggregation
class AggHashTable {
   public:
    struct Value {
        double  sum   = 0.0;
        int64_t count = 0;
    };

    AggHashTable(AggHashTable&&) noexcept = default;

    AggHashTable& operator=(AggHashTable&&) noexcept = default;

    explicit AggHashTable(std::size_t initial_capacity = 1024);

    // Find existing entry for key, or insert a new one (with default value).
    // Returns reference to the entry's value, valid until the next mutation.
    Value& find_or_insert(int64_t key);

    std::size_t size() const { return size_; }

    // Iterate all live entries. Callback signature: (int64_t key, const Value& val).
    template <typename F>
    void for_each(F&& f) const {
        for (const auto& slot : slots_) {
            if (slot.key != kEmptyKey) f(slot.key, slot.value);
        }
    }

   private:
    static constexpr int64_t kEmptyKey = INT64_MIN;
    struct Slot {
        int64_t key = kEmptyKey;
        Value   value;
    };

    void resize_(std::size_t new_capacity);

    std::vector<Slot> slots_;
    std::size_t       size_     = 0;  // count of occupied slots;
    std::size_t       capacity_ = 0;  // == slots.size(), always power of 2
};
}  // namespace wb