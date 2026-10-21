#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace wb {

// Maps int64 group keys to dense group ids 0, 1, 2, ... in first-seen order, so per-group
// aggregate state can live in plain arrays indexed by id. Open addressing with linear probing on
// the low bits of the multiplicative hash; power-of-two capacity, grown when half full. The
// codegen runtime's GroupTable implements the same algorithm.
class GroupIndex {
   public:
    explicit GroupIndex(std::size_t initial_capacity = 1024);

    // The id of `key`, assigning the next id if the key is new. The lookup of an existing key is
    // inline: it runs once per input row.
    uint32_t find_or_insert(int64_t key) {
        if (key != kEmptyKey && !slots_.empty()) {
            const std::size_t mask = slots_.size() - 1;
            for (std::size_t i = (static_cast<uint64_t>(key) * 0x9E3779B97F4A7C15ULL) & mask;; i = (i + 1) & mask) {
                const Slot& slot = slots_[i];
                if (slot.key == key) return slot.gid;
                if (slot.key == kEmptyKey) break;
            }
        }
        return insert_(key);
    }

    uint32_t    size() const { return static_cast<uint32_t>(keys_.size()); }
    int64_t     key_of(uint32_t gid) const { return keys_[gid]; }
    std::size_t capacity() const { return slots_.size(); }

   private:
    static constexpr int64_t kEmptyKey = INT64_MIN;

    struct Slot {
        int64_t  key = kEmptyKey;
        uint32_t gid = 0;
    };

    uint32_t insert_(int64_t key);  // key not present: assign the next id
    void     grow_();

    std::vector<Slot>    slots_;
    std::vector<int64_t> keys_;  // gid -> key

    // INT64_MIN is the empty-slot marker, so that key lives outside the slot array.
    bool     has_min_key_ = false;
    uint32_t min_key_gid_ = 0;
};

}  // namespace wb
