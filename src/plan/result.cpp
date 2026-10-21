#include "plan/result.h"

#include <algorithm>
#include <bit>
#include <numeric>
#include <sstream>

namespace wb::plan {

namespace {

bool same_bits(double a, double b) {
    return std::bit_cast<std::uint64_t>(a) == std::bit_cast<std::uint64_t>(b);
}

void fnv_mix(std::uint64_t& h, std::uint64_t v) {
    for (int i = 0; i < 8; ++i) {
        h ^= (v >> (8 * i)) & 0xFF;
        h *= 0x100000001B3ULL;
    }
}

}  // namespace

void QueryResult::canonicalize() {
    if (!grouped) return;
    std::vector<std::size_t> order(keys.size());
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return keys[a] < keys[b]; });

    auto permute = [&](auto& column) {
        auto copy = column;
        for (std::size_t i = 0; i < order.size(); ++i)
            column[i] = copy[order[i]];
    };
    permute(keys);
    permute(sums);
    permute(counts);
}

bool QueryResult::operator==(const QueryResult& other) const { return diff(other).empty(); }

std::uint64_t QueryResult::checksum() const {
    std::uint64_t h = 0xCBF29CE484222325ULL;
    fnv_mix(h, grouped);
    for (std::size_t i = 0; i < num_rows(); ++i) {
        if (grouped) fnv_mix(h, static_cast<std::uint64_t>(keys[i]));
        fnv_mix(h, std::bit_cast<std::uint64_t>(sums[i]));
        fnv_mix(h, static_cast<std::uint64_t>(counts[i]));
    }
    return h;
}

std::string QueryResult::diff(const QueryResult& other) const {
    std::ostringstream out;
    out.precision(17);
    if (grouped != other.grouped) {
        out << "grouped " << grouped << " vs " << other.grouped;
        return out.str();
    }
    if (num_rows() != other.num_rows()) {
        out << "row count " << num_rows() << " vs " << other.num_rows();
        return out.str();
    }
    for (std::size_t i = 0; i < num_rows(); ++i) {
        const bool key_differs = grouped && keys[i] != other.keys[i];
        if (key_differs || !same_bits(sums[i], other.sums[i]) || counts[i] != other.counts[i]) {
            out << "row " << i << ": (" << (grouped ? keys[i] : 0) << ", " << sums[i] << ", "
                << counts[i] << ") vs (" << (grouped ? other.keys[i] : 0) << ", " << other.sums[i]
                << ", " << other.counts[i] << ")";
            return out.str();
        }
    }
    return "";
}

}  // namespace wb::plan
