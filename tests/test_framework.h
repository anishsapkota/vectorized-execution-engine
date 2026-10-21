#pragma once

#include <functional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// A deliberately tiny test harness: TEST registers a function, CHECK/CHECK_EQ throw on failure.
// A test that hits a NotImplemented stub is reported as TODO, not FAIL.

namespace wbtest {

struct TestCase {
    const char*           name;
    std::function<void()> body;
};

inline std::vector<TestCase>& registry() {
    static std::vector<TestCase> tests;
    return tests;
}

struct Registrar {
    Registrar(const char* name, std::function<void()> body) {
        registry().push_back({name, std::move(body)});
    }
};

struct TestFailure : std::runtime_error {
    using std::runtime_error::runtime_error;
};

}  // namespace wbtest

#define TEST(name)                                                  \
    static void                name();                              \
    static wbtest::Registrar   name##_registrar{#name, name};       \
    static void                name()

#define CHECK(cond)                                                                     \
    do {                                                                                \
        if (!(cond)) {                                                                  \
            throw wbtest::TestFailure(std::string(__FILE__) + ":" +                     \
                                      std::to_string(__LINE__) + ": CHECK(" #cond ")"); \
        }                                                                               \
    } while (0)

#define CHECK_EQ(actual, expected)                                                      \
    do {                                                                                \
        const auto& actual_value_   = (actual);                                         \
        const auto& expected_value_ = (expected);                                       \
        if (!(actual_value_ == expected_value_)) {                                      \
            std::ostringstream message_;                                                \
            message_ << __FILE__ << ":" << __LINE__ << ": CHECK_EQ(" #actual ", " #expected \
                     << ") got " << actual_value_ << ", expected " << expected_value_;   \
            throw wbtest::TestFailure(message_.str());                                  \
        }                                                                               \
    } while (0)
