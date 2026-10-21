#pragma once

#include <cstddef>
#include <memory>
#include <mutex>
#include <vector>

#include "dist/chunk.h"
#include "dist/cluster.h"
#include "exec/operator.h"

namespace wb {

// Moves rows between slices for one exchange point in a plan.
//
// Execution is staged (see stream.h): every sender drains its input completely before any
// receiver starts. That is why an inbox is a mutex-guarded list rather than a bounded queue,
// and why receivers need no end-of-stream counting: the barrier between streams is the EOS.
// A pipelined exchange would need both, plus backpressure; that is a later exercise.
class Exchange {
   public:
    explicit Exchange(Cluster& cluster);

    // Sender side. Each call drains `input` on behalf of `src_slice`.
    //
    // redistribute: every live row goes to slice_for_key(row[key_col]); key_col must be int64.
    void redistribute(Operator& input, std::size_t src_slice, std::size_t key_col);

    // broadcast: every live row goes to every slice. A chunk is charged to the network once
    // per destination node; slices on the same node share the received copy.
    void broadcast(Operator& input, std::size_t src_slice);

    // Receiver side: an operator emitting everything sent to `dst_slice`. Takes ownership of
    // the inbox contents, so call it once per slice, after the sending stream has finished.
    std::unique_ptr<Operator> receive(std::size_t dst_slice);

   private:
    struct Inbox {
        std::mutex                                mutex;
        std::vector<std::shared_ptr<const Chunk>> chunks;
    };

    void deliver(std::size_t src_slice, std::size_t dst_slice, std::shared_ptr<const Chunk> chunk);
    void deliver_to_all(std::size_t src_slice, const std::shared_ptr<const Chunk>& chunk);

    Cluster&                            cluster_;
    std::vector<std::unique_ptr<Inbox>> inboxes_;  // index = destination slice
};

}  // namespace wb
