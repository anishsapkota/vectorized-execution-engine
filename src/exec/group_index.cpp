#include "exec/group_index.h"

namespace wb {

GroupIndex::GroupIndex(std::size_t initial_capacity) {
    std::size_t cap = 1;
    while (cap < initial_capacity)
        cap <<= 1;
    slots_.assign(cap, Slot{});
}

uint32_t GroupIndex::insert_(int64_t key) {
    if (key == kEmptyKey) {
        if (!has_min_key_) {
            has_min_key_ = true;
            min_key_gid_ = size();
            keys_.push_back(key);
        }
        return min_key_gid_;
    }
    if (keys_.size() * 2 >= slots_.size()) grow_();

    const std::size_t mask = slots_.size() - 1;
    for (std::size_t i = (static_cast<uint64_t>(key) * 0x9E3779B97F4A7C15ULL) & mask;; i = (i + 1) & mask) {
        Slot& slot = slots_[i];
        if (slot.key == key) return slot.gid;
        if (slot.key == kEmptyKey) {
            slot = {key, size()};
            keys_.push_back(key);
            return slot.gid;
        }
    }
}

void GroupIndex::grow_() {
    std::vector<Slot> old = std::move(slots_);
    slots_.assign(old.size() * 2, Slot{});
    const std::size_t mask = slots_.size() - 1;
    for (const Slot& s : old) {
        if (s.key == kEmptyKey) continue;
        std::size_t i = (static_cast<uint64_t>(s.key) * 0x9E3779B97F4A7C15ULL) & mask;
        while (slots_[i].key != kEmptyKey)
            i = (i + 1) & mask;
        slots_[i] = s;
    }
}

}  // namespace wb
