#include "exec/hash_table.h"

namespace wb {

AggHashTable::AggHashTable(std::size_t initial_capacity) {
    std::size_t cap = 1;
    while (cap < initial_capacity)
        cap <<= 1;  // round up to next power of 2
    slots_.assign(cap, Slot{});
    capacity_ = cap;
}

AggHashTable::Value& AggHashTable::find_or_insert(int64_t key) {
    if (size_ * 2 >= capacity_) {
        resize_(capacity_ * 2);
    }

    // Multiply - shift hash,then mask to capacity.
    // Knuth's multiplicative hash
    uint64_t    h    = static_cast<uint64_t>(key) * 0x9E3779B97F4A7C15ULL;
    std::size_t mask = capacity_ - 1;
    std::size_t idx  = h & mask;

    while (true) {
        Slot& slot = slots_[idx];
        if (slot.key == key) {
            return slot.value;  // existing
        }
        if (slot.key == kEmptyKey) {
            slot.key = key;
            ++size_;
            return slot.value;  // fresh
        }
        idx = (idx + 1) & mask;  // probe next
    }
}

void AggHashTable::resize_(std::size_t new_capacity) {
    std::vector<Slot> old = std::move(slots_);
    slots_.assign(new_capacity, Slot{});  // all kEmptyKey
    capacity_ = new_capacity;
    size_     = 0;

    for (const auto& slot : old) {
        if (slot.key != kEmptyKey) {
            Value& v = find_or_insert(slot.key);
            v        = slot.value;
        }
    }
}
}  // namespace wb