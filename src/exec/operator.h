#pragma once

#include <exec/batch.h>
#include <optional>

namespace wb {

class Operator {
   public:
    virtual ~Operator() = default;

    // Returns the next batch of rows. Empty optional = exhausted.
    // The returned batch is valid only until the next call to next() on this operator or any
    // operator upstream of it
    virtual std::optional<Batch> next() = 0;
};

}  // namespace wb