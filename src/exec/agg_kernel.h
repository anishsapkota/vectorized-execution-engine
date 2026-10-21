#pragma once

#include <cstdint>

#include "exec/hash_table.h"
#include "storage/vector.h"

namespace wb {

// GROUP BY keys: SUM(vals), COUNT(*) into ht, over the rows produced by for_each_row.
// for_each_row(f) must call f(std::uint32_t row) for every live row. Encoding dispatch
// (key width × value width) happens once per call, outside the row loop.
template <typename ForEachRow>
void accumulate_sum_count(AggHashTable& ht, const Int64Vector& keys, const Float64Vector& vals,
                          ForEachRow&& for_each_row) {
    keys.visit([&](auto codes, int64_t base) {
        vals.visit([&](auto values) {
            for_each_row([&](std::uint32_t i) {
                auto& entry = ht.find_or_insert(Int64Vector::decode(base, codes[i]));
                entry.sum += values[i];
                entry.count += 1;
            });
        });
    });
}

}  // namespace wb
