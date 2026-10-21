// wb_mem: peak memory of one aggregate query on one engine, above the loaded table.
//
//   ./wb_mem <q4|q7|q8|q7f> <vec|cg> [--csv data/readings.csv]
//
// One query per process: the OS peak counter only ever grows, so each measurement needs a fresh
// process. q7f runs q7 on a copy of readings whose `value` column is stored as float.

#include <cstdio>
#include <cstdlib>
#include <string>

#if defined(__APPLE__)
#include <mach/mach.h>
#else
#include <fstream>
#endif

#include "codegen/host.h"
#include "plan/lower_vec.h"
#include "plan/plan.h"
#include "storage/csv.h"
#include "storage/table.h"

namespace {

using namespace wb;
using namespace wb::plan;

struct Footprint {
    std::size_t now  = 0;
    std::size_t peak = 0;
};

Footprint footprint() {
#if defined(__APPLE__)
    task_vm_info_data_t    info;
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO, reinterpret_cast<task_info_t>(&info), &count) != KERN_SUCCESS) {
        return {};
    }
    return {static_cast<std::size_t>(info.phys_footprint), static_cast<std::size_t>(info.ledger_phys_footprint_peak)};
#else
    Footprint     f;
    std::ifstream status{"/proc/self/status"};
    for (std::string key; status >> key;) {
        std::size_t kib = 0;
        if (key == "VmRSS:" && status >> kib) f.now = kib * 1024;
        if (key == "VmHWM:" && status >> kib) f.peak = kib * 1024;
    }
    return f;
#endif
}

int usage() {
    std::fprintf(stderr, "usage: wb_mem <q4|q7|q8|q7f> <vec|cg> [--csv path]\n");
    return 1;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) return usage();
    const std::string query = argv[1], engine = argv[2];
    std::string       csv   = "data/readings.csv";
    if (argc == 5 && std::string{argv[3]} == "--csv") csv = argv[4];
    if (engine != "vec" && engine != "cg") return usage();

    try {
        Table t{"readings",
                {{"device_id", DataType::Int64},
                 {"ts", DataType::Int64},
                 {"value", DataType::Float64, /*narrow_float=*/query == "q7f"}},
                /*sort_key_col=*/1};
        load_csv(t, csv, 65536);
        std::size_t rows = 0;
        for (const auto& seg : t.segments())
            rows += seg->num_rows();

        NodePtr plan;
        if (query == "q4") plan = sum_count(scan(t, {0, 2}), 0, 1);
        else if (query == "q7" || query == "q7f") plan = aggregate(scan(t, {0, 2}), 0, {{AggFn::Median, 1}});
        else if (query == "q8") plan = aggregate(scan(t, {2}), std::nullopt, {{AggFn::Median, 0}});
        else return usage();

        codegen::PreparedQuery prepared;
        if (engine == "cg") prepared = codegen::prepare(*plan);

        const Footprint before = footprint();
        if (engine == "cg") codegen::execute(prepared);
        else run_vectorized(*plan);
        const Footprint after = footprint();

        const std::size_t extra = after.peak > before.now ? after.peak - before.now : 0;
        std::printf("%s %s peak above loaded table: %.1f MiB (%.1f B/row)\n", query.c_str(), engine.c_str(),
                    static_cast<double>(extra) / (1024.0 * 1024.0),
                    rows ? static_cast<double>(extra) / static_cast<double>(rows) : 0.0);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
