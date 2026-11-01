#pragma once

#include <vector>

#include "codegen/compiler.h"
#include "codegen/generator.h"
#include "plan/plan.h"
#include "plan/result.h"

namespace wb::codegen {

// A plan compiled once, runnable many times. Refers to the plan; keep it alive.
struct PreparedQuery {
    const plan::Node*        root = nullptr;
    std::vector<ScanBinding> scans;
    CompiledQuery            compiled;  // compiled.stats.generate_ms is filled in
    bool                     grouped = false;
};

// Validates, binds, generates and compiles `root`. Throws std::invalid_argument for an invalid
// plan and CodegenError for everything after that.
PreparedQuery prepare(const plan::Node& root, const CompileOptions& options = {});

// Prunes each scan's segments exactly as the vectorized ScanOperator does, binds their columns
// into QueryInput, calls wb_query and collects a canonicalized result. Throws CodegenError if
// the library reports an error (e.g. a width signature it was not compiled for).
plan::QueryResult execute(const PreparedQuery& query);

// prepare + execute. If `stats` is set, it receives the compile statistics.
plan::QueryResult run_compiled(const plan::Node& root, const CompileOptions& options = {},
                               CompileStats* stats = nullptr);

}  // namespace wb::codegen
