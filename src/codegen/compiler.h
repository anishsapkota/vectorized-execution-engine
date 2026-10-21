#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "codegen/runtime.h"

namespace wb::codegen {

// Every failure to generate, compile, load or run generated code. Never caught to fall back to
// the vectorized engine.
struct CodegenError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

struct CompileOptions {
    int  opt_level = 3;     // -O<n>
    bool use_cache = true;  // false: always compile and load a fresh library (cold-run timing)
};

struct CompileStats {
    double generate_ms = 0;  // filled by run_compiled, not by compile_source
    double compile_ms  = 0;  // clang wall time; 0 on a cache hit
    double load_ms     = 0;  // dlopen + dlsym + ABI check
    bool   cache_hit   = false;
};

using QueryFn = int (*)(const wb_rt::QueryInput*, wb_rt::QueryOutput*);

struct CompiledQuery {
    QueryFn       fn = nullptr;
    CompileStats  stats;
    std::uint64_t key = 0;
};

// FNV-1a 64 over source, compiler path, flags and the contents of codegen/runtime.h. Cached files
// are named <cache dir>/<key as 16 lowercase hex digits>.<library_extension()>.
std::uint64_t cache_key(const std::string& source, const CompileOptions& options);
// Test seam: as above, with the runtime header's contents supplied by the caller.
std::uint64_t cache_key(const std::string& source, const CompileOptions& options,
                        const std::string& runtime_text);

// Compiles `source` into a shared library and returns its wb_query entry point. Lookup order:
// in-memory cache, then the on-disk cache, then $WB_CXX (default clang++). Reads WB_CXX,
// WB_CODEGEN_CACHE (default .wb_cache) and WB_CODEGEN_EXTRA_FLAGS on every call. Libraries stay
// loaded for the life of the process. Throws CodegenError.
CompiledQuery compile_source(const std::string& source, const CompileOptions& options);

// Number of compiler processes started by this process, for tests.
std::size_t compiler_invocations();

// "dylib" on macOS, "so" elsewhere.
const char* library_extension();

}  // namespace wb::codegen
