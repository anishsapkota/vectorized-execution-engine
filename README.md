A simple vectorized execution engine to understand how pull based vectorized execution model works and how it differs from traditional row based volcano iterator model

## weekendRedshift: slices, distribution styles and join data movement

A simulated Redshift cluster on top of the engine: N nodes x S slices, one thread per slice,
tables spread by DISTSTYLE (EVEN / KEY / ALL), and joins planned as Redshift's DS_* strategies.

    make test
    make wb_dist  # then ./wb_dist --experiment strategies|skew|crossover|all

The three tasks, each specified in its header:
1. `distribute_table` (src/dist/distribute.h): the loader for EVEN / KEY / ALL.
2. `HashJoinOperator` (src/exec/hash_join.h): inner equi-join on an int64 key.
3. `is_valid_strategy` + `choose_join_strategy` (src/dist/planner.h): the planner rule.

Infrastructure: `Cluster` (dist/cluster.h), `Exchange` with redistribute/broadcast
(dist/exchange.h), `run_stream` (dist/stream.h), `run_distributed_join` (dist/dist_join.h).
