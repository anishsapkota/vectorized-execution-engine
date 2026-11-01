#include "exec/group_slots.h"

#include <stdexcept>

namespace wb {

GroupSlots::GroupSlots(std::size_t state_bytes, std::size_t state_align, std::size_t initial_capacity)
    : stride_units_(1 + (state_bytes + 15) / 16) {
    if (state_align > 16) throw std::invalid_argument("GroupSlots: state alignment above 16 is not supported");
    std::size_t cap = 1;
    while (cap < initial_capacity)
        cap <<= 1;
    capacity_ = cap;
    slots_.assign(capacity_ * stride_units_, Unit{});
    for (std::size_t i = 0; i < capacity_; ++i)
        std::memcpy(slot_at_(i), &kEmptyKey, sizeof kEmptyKey);
    min_state_.assign(stride_units_ - 1 > 0 ? stride_units_ - 1 : 1, Unit{});
}

void GroupSlots::reserve(std::size_t groups) {
    while (groups * 2 > capacity_)
        grow_();
}

std::byte* GroupSlots::state_of(uint32_t gid) {
    const uint32_t slot = slot_of_[gid];
    if (slot == kMinSlot) return reinterpret_cast<std::byte*>(min_state_.data());
    return slot_at_(slot) + kHeader;
}

std::byte* GroupSlots::insert_(int64_t key, bool& fresh) {
    if (key == kEmptyKey) {
        fresh = !has_min_key_;
        if (fresh) {
            has_min_key_ = true;
            slot_of_.push_back(kMinSlot);
            keys_.push_back(key);
        }
        return reinterpret_cast<std::byte*>(min_state_.data());
    }
    if (keys_.size() * 2 >= capacity_) grow_();

    const std::size_t mask = capacity_ - 1;
    for (std::size_t i = (static_cast<uint64_t>(key) * 0x9E3779B97F4A7C15ULL) & mask;; i = (i + 1) & mask) {
        std::byte* slot = slot_at_(i);
        int64_t    k;
        std::memcpy(&k, slot, sizeof k);
        if (k == key) {  // present (callers normally take the inline hit path)
            fresh = false;
            return slot + kHeader;
        }
        if (k == kEmptyKey) {
            const uint32_t gid = size();
            std::memcpy(slot, &key, sizeof key);
            std::memcpy(slot + 8, &gid, sizeof gid);
            keys_.push_back(key);
            slot_of_.push_back(static_cast<uint32_t>(i));
            fresh = true;
            return slot + kHeader;
        }
    }
}

void GroupSlots::grow_() {
    std::vector<Unit> old = std::move(slots_);
    const std::size_t old_capacity = capacity_;
    capacity_ *= 2;
    slots_.assign(capacity_ * stride_units_, Unit{});
    for (std::size_t i = 0; i < capacity_; ++i)
        std::memcpy(slot_at_(i), &kEmptyKey, sizeof kEmptyKey);

    const std::size_t mask = capacity_ - 1;
    const std::size_t bytes = stride_units_ * sizeof(Unit);
    for (std::size_t i = 0; i < old_capacity; ++i) {
        const std::byte* src = reinterpret_cast<const std::byte*>(old.data() + i * stride_units_);
        int64_t          key;
        std::memcpy(&key, src, sizeof key);
        if (key == kEmptyKey) continue;
        std::size_t j = (static_cast<uint64_t>(key) * 0x9E3779B97F4A7C15ULL) & mask;
        for (;; j = (j + 1) & mask) {
            int64_t k;
            std::memcpy(&k, slot_at_(j), sizeof k);
            if (k == kEmptyKey) break;
        }
        std::memcpy(slot_at_(j), src, bytes);  // key, gid and state move together
        uint32_t gid;
        std::memcpy(&gid, src + 8, sizeof gid);
        slot_of_[gid] = static_cast<uint32_t>(j);
    }
}

}  // namespace wb
