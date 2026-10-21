#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "exec/operator.h"
#include "exec/predicate.h"

namespace wb {

class FilterOperator final : public Operator {
   public:
    FilterOperator(std::unique_ptr<Operator> child, Predicate predicate);

    std::optional<Batch> next() override;

   private:
    std::unique_ptr<Operator> child_;
    Predicate                 predicate_;
    // Reused across next() calls and borrowed by the emitted batch's selection vector.
    std::vector<std::uint32_t> survivors_buf_;
};

}  // namespace wb