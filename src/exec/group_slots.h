#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace wb {

// Hash table from int64 group key to a fixed-size block of per-group aggregate state, stored
// inline in the slot: [key int64 | gid uint32 | pad | state ...], 16-byte aligned. Group ids are
// dense and first-seen (0, 1, 2, ...), for output order and side structures. Same hash (low bits
// of the multiplicative hash), linear probing, initial capacity and growth rule as the codegen
// runtime's GroupRows.
//
// State pointers stay valid until the table grows; call reserve(size() + n) before inserting up
// to n new keys to keep every pointer taken in between valid.
class GroupSlots {
   public:
    GroupSlots(std::size_t state_bytes, std::size_t state_align, std::size_t initial_capacity = 1024);

    // Grows now, if needed, so that `groups` groups fit without a rehash.
    void reserve(std::size_t groups);

    // The state block of `key`'s group. For a new group `fresh` is set and the block is zeroed;
    // the caller initializes it.
    std::byte* find_or_insert(int64_t key, bool& fresh) {
        if (key != kEmptyKey && !slots_.empty()) {
            const std::size_t mask = capacity_ - 1;
            for (std::size_t i = (static_cast<uint64_t>(key) * 0x9E3779B97F4A7C15ULL) & mask;; i = (i + 1) & mask) {
                std::byte* slot = slot_at_(i);
                int64_t    k;
                std::memcpy(&k, slot, sizeof k);
                if (k == key) {
                    fresh = false;
                    return slot + kHeader;
                }
                if (k == kEmptyKey) break;
            }
        }
        return insert_(key, fresh);
    }

    uint32_t    size() const { return static_cast<uint32_t>(keys_.size()); }
    int64_t     key_of(uint32_t gid) const { return keys_[gid]; }
    std::byte*  state_of(uint32_t gid);
    std::size_t capacity() const { return capacity_; }

   private:
    static constexpr int64_t     kEmptyKey = INT64_MIN;
    static constexpr std::size_t kHeader   = 16;  // key, gid, padding
    static constexpr uint32_t    kMinSlot  = UINT32_MAX;

    struct alignas(16) Unit {
        std::byte bytes[16];
    };

    std::byte* slot_at_(std::size_t i) { return reinterpret_cast<std::byte*>(slots_.data() + i * stride_units_); }
    std::byte* insert_(int64_t key, bool& fresh);
    void       grow_();

    std::size_t           stride_units_;  // slot size in 16-byte units
    std::size_t           capacity_ = 0;
    std::vector<Unit>     slots_;
    std::vector<int64_t>  keys_;     // gid -> key
    std::vector<uint32_t> slot_of_;  // gid -> slot index (kMinSlot for INT64_MIN)

    // INT64_MIN is the empty-slot marker, so that key's state lives outside the slot array.
    bool              has_min_key_ = false;
    std::vector<Unit> min_state_;
};

}  // namespace wb
