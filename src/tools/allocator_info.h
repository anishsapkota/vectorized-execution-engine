#pragma once

// Which malloc this process runs on, for benchmark output: "jemalloc <version>" when the binary was
// linked with ALLOCATOR=jemalloc (jemalloc then also serves the dlopen'ed generated query code,
// through its default-zone registration), otherwise "system".

#include <dlfcn.h>

#include <cstddef>
#include <string>

namespace wb {

inline std::string allocator_name() {
    using Mallctl = int (*)(const char*, void*, std::size_t*, void*, std::size_t);
    auto mallctl  = reinterpret_cast<Mallctl>(dlsym(RTLD_DEFAULT, "mallctl"));
    if (!mallctl) return "system";
    const char* version = nullptr;
    std::size_t size    = sizeof version;
    if (mallctl("version", &version, &size, nullptr, 0) == 0 && version) return std::string{"jemalloc "} + version;
    return "jemalloc";
}

}  // namespace wb
