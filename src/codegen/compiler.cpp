#include "codegen/compiler.h"

#include <dlfcn.h>
#include <fcntl.h>
#include <spawn.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <sstream>
#include <unordered_map>
#include <vector>

#ifndef WB_SRC_DIR
#error "WB_SRC_DIR must point at the engine's src/ directory (set by the Makefile)"
#endif

extern char** environ;

namespace wb::codegen {

namespace {

using Clock = std::chrono::steady_clock;

std::atomic<std::size_t> g_invocations{0};
std::atomic<std::size_t> g_unique_counter{0};
std::mutex               g_cache_mutex;
std::unordered_map<std::uint64_t, QueryFn> g_loaded;  // key -> entry point; never unloaded

double ms_since(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

std::string env_or(const char* name, const char* fallback) {
    const char* value = std::getenv(name);
    return value && *value ? value : fallback;
}

std::string compiler_path() { return env_or("WB_CXX", "clang++"); }

std::vector<std::string> compile_flags(const CompileOptions& options) {
    std::vector<std::string> flags{"-std=c++20", "-O" + std::to_string(options.opt_level), "-march=native",
                                   "-fPIC", "-shared", std::string{"-I"} + WB_SRC_DIR};
    std::istringstream extra{env_or("WB_CODEGEN_EXTRA_FLAGS", "")};
    for (std::string flag; extra >> flag;)
        flags.push_back(flag);
    return flags;
}

const std::string& runtime_header_text() {
    static const std::string text = [] {
        std::ifstream      in{std::string{WB_SRC_DIR} + "/codegen/runtime.h"};
        std::ostringstream out;
        out << in.rdbuf();
        return out.str();
    }();
    return text;
}

void fnv_mix(std::uint64_t& h, const std::string& bytes) {
    for (unsigned char c : bytes) {
        h ^= c;
        h *= 0x100000001B3ULL;
    }
    h ^= 0xFF;  // separator, so ("ab","c") and ("a","bc") differ
    h *= 0x100000001B3ULL;
}

std::string hex_key(std::uint64_t key) {
    char buf[17];
    std::snprintf(buf, sizeof buf, "%016llx", static_cast<unsigned long long>(key));
    return buf;
}

std::filesystem::path cache_dir() {
    std::filesystem::path dir = env_or("WB_CODEGEN_CACHE", ".wb_cache");
    std::error_code       ec;
    if (std::filesystem::create_directories(dir, ec)) {
        std::filesystem::permissions(dir, std::filesystem::perms::owner_all,
                                     std::filesystem::perm_options::replace, ec);
    }
    struct stat st {};
    if (stat(dir.c_str(), &st) != 0 || !S_ISDIR(st.st_mode)) {
        throw CodegenError("codegen cache directory " + dir.string() + " cannot be created");
    }
    // Every library found here is dlopen'ed into this process: refuse a directory someone else
    // could have written to.
    if (st.st_uid != getuid()) {
        throw CodegenError("codegen cache directory " + dir.string() + " is not owned by the current user");
    }
    if (st.st_mode & (S_IWGRP | S_IWOTH)) {
        throw CodegenError("codegen cache directory " + dir.string() +
                           " is writable by group or others; chmod 700 it or set WB_CODEGEN_CACHE");
    }
    return dir;
}

std::string read_file(const std::filesystem::path& path) {
    std::ifstream      in{path};
    std::ostringstream out;
    out << in.rdbuf();
    return out.str();
}

// Runs the compiler; on success the library is at `out`.
void run_compiler(const std::filesystem::path& src, const std::filesystem::path& out,
                  const CompileOptions& options) {
    const std::string        cxx = compiler_path();
    std::vector<std::string> args{cxx};
    for (auto& flag : compile_flags(options))
        args.push_back(std::move(flag));
    args.insert(args.end(), {"-o", out.string(), src.string()});

    std::vector<char*> argv;
    for (auto& a : args)
        argv.push_back(a.data());
    argv.push_back(nullptr);

    const std::filesystem::path err = src.string() + ".err";
    posix_spawn_file_actions_t  actions;
    posix_spawn_file_actions_init(&actions);
    posix_spawn_file_actions_addopen(&actions, STDOUT_FILENO, err.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
    posix_spawn_file_actions_adddup2(&actions, STDOUT_FILENO, STDERR_FILENO);

    ++g_invocations;
    pid_t     pid = 0;
    const int rc  = posix_spawnp(&pid, cxx.c_str(), &actions, nullptr, argv.data(), environ);
    posix_spawn_file_actions_destroy(&actions);
    if (rc != 0) {
        throw CodegenError("cannot start compiler '" + cxx + "': " + std::strerror(rc) +
                           " (source kept at " + src.string() + ")");
    }

    int status = 0;
    while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {
    }
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        const int code = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
        throw CodegenError("compiler '" + cxx + "' failed with exit status " + std::to_string(code) +
                           " for " + src.string() + ":\n" + read_file(err));
    }
}

QueryFn load(const std::filesystem::path& lib) {
    void* handle = dlopen(lib.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!handle) throw CodegenError("cannot load " + lib.string() + ": " + dlerror());

    auto* version = reinterpret_cast<std::uint32_t (*)()>(dlsym(handle, "wb_abi_version"));
    auto* query   = reinterpret_cast<QueryFn>(dlsym(handle, "wb_query"));
    if (!version || !query) {
        const char* why = dlerror();
        throw CodegenError("library " + lib.string() + " lacks wb_query/wb_abi_version: " +
                           (why ? why : "symbol missing"));
    }
    if (version() != wb_rt::kAbiVersion) {
        throw CodegenError("library " + lib.string() + " has ABI version " + std::to_string(version()) +
                           ", host expects " + std::to_string(wb_rt::kAbiVersion));
    }
    return query;
}

}  // namespace

const char* library_extension() {
#if defined(__APPLE__)
    return "dylib";
#else
    return "so";
#endif
}

std::size_t compiler_invocations() { return g_invocations.load(); }

std::uint64_t cache_key(const std::string& source, const CompileOptions& options,
                        const std::string& runtime_text) {
    std::uint64_t h = 0xCBF29CE484222325ULL;
    fnv_mix(h, source);
    fnv_mix(h, compiler_path());
    for (const auto& flag : compile_flags(options))
        fnv_mix(h, flag);
    fnv_mix(h, runtime_text);
    return h;
}

std::uint64_t cache_key(const std::string& source, const CompileOptions& options) {
    return cache_key(source, options, runtime_header_text());
}

CompiledQuery compile_source(const std::string& source, const CompileOptions& options) {
    CompiledQuery result;
    result.key = cache_key(source, options);

    std::scoped_lock lock{g_cache_mutex};
    if (options.use_cache) {
        if (auto it = g_loaded.find(result.key); it != g_loaded.end()) {
            result.fn              = it->second;
            result.stats.cache_hit = true;
            return result;
        }
    }

    const auto dir  = cache_dir();
    const auto name = hex_key(result.key);
    const auto lib  = dir / (name + "." + library_extension());

    if (options.use_cache && std::filesystem::exists(lib)) {
        const auto start       = Clock::now();
        result.fn              = load(lib);
        result.stats.load_ms   = ms_since(start);
        result.stats.cache_hit = true;
        g_loaded.emplace(result.key, result.fn);
        return result;
    }

    // Cold path. Scratch files (source, compiler output, stderr) are per process and per call,
    // so concurrent processes compiling the same key never share a file. A cached library is
    // then renamed onto <key>.<ext>: rename is atomic, and every writer produces the same
    // library, so the last one wins harmlessly. A bypassed cache keeps the unique name, so
    // dlopen cannot hand back an already-loaded copy.
    const std::string stem = name + "." + std::to_string(getpid()) + "." + std::to_string(++g_unique_counter);
    const auto        src  = dir / (stem + ".cpp");
    const auto        tmp  = dir / (stem + ".tmp." + library_extension());
    const auto        out  = options.use_cache ? lib : dir / (stem + "." + library_extension());
    {
        std::ofstream file{src};
        file << source;
        file.close();
        if (!file) throw CodegenError("cannot write " + src.string());
    }

    auto start = Clock::now();
    run_compiler(src, tmp, options);
    result.stats.compile_ms = ms_since(start);

    std::error_code ec;
    std::filesystem::rename(tmp, out, ec);  // atomic: a reader never sees a half-written library
    if (ec) throw CodegenError("cannot move " + tmp.string() + " to " + out.string() + ": " + ec.message());
    std::filesystem::remove(src, ec);  // kept only when compilation fails, for debugging
    std::filesystem::remove(src.string() + ".err", ec);

    start                = Clock::now();
    result.fn            = load(out);
    result.stats.load_ms = ms_since(start);
    if (options.use_cache) g_loaded.emplace(result.key, result.fn);
    return result;
}

}  // namespace wb::codegen
