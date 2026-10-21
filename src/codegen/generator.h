#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "exec/predicate.h"
#include "plan/plan.h"

namespace wb::codegen {

// What the generator and the host agree on about one Scan in the plan. Scans are numbered in
// pre-order, visiting a HashJoin's build side before its probe side; that is also the order
// pipelines run in and the order of QueryInput::scans.
struct ScanBinding {
    const plan::Scan*          scan = nullptr;
    std::vector<std::size_t>   int_cols;    // plan positions of the scan's Int64 columns
    std::vector<std::uint32_t> width_sigs;  // distinct signatures over the table's segments, sorted
    // Zone-map / sparse-index predicate, in table column numbering: the first conjunct of a
    // Filter directly above this scan. Both engines prune with exactly this.
    std::optional<Predicate> prune;
};

std::vector<ScanBinding> bind_scans(const plan::Node& root);

// Width signature of `seg` for `binding` (runtime.h: width_code of the k-th Int64 column in bits
// 2k..2k+1).
std::uint32_t width_sig(const ScanBinding& binding, const Segment& seg);

// C++ source for a validated SumCount plan: one fused loop per pipeline (produce/consume),
// specialized to each scan's width signatures. Exports wb_query and wb_abi_version.
// Throws CodegenError for plans the generator does not support.
std::string generate(const plan::Node& root, const std::vector<ScanBinding>& scans);

}  // namespace wb::codegen
