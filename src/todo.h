#pragma once

#include <stdexcept>

namespace wb {

// Thrown by stubs that are deliberately left for you to implement. The test runner reports
// these as TODO rather than FAIL, so the suite stays green while the work is in progress.
struct NotImplemented : std::logic_error {
    using std::logic_error::logic_error;
};

}  // namespace wb
