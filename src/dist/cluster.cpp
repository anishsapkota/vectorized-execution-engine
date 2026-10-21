#include "dist/cluster.h"

#include <stdexcept>

namespace wb {

const char* dist_style_name(DistStyle style) {
    switch (style) {
        case DistStyle::Even:
            return "EVEN";
        case DistStyle::Key:
            return "KEY";
        case DistStyle::All:
            return "ALL";
    }
    return "unknown";
}

Cluster::Cluster(ClusterConfig config) : config_(config) {
    if (config_.nodes == 0 || config_.slices_per_node == 0) {
        throw std::invalid_argument("cluster needs at least one node and one slice per node");
    }
}

void Cluster::declare_table(const std::string& name, Schema schema, Distribution distribution,
                            std::optional<std::size_t> sort_key_col) {
    if (tables_.contains(name)) {
        throw std::invalid_argument("table '" + name + "' is already declared");
    }
    if (distribution.style == DistStyle::Key) {
        if (distribution.key_col >= schema.size() ||
            schema[distribution.key_col].type != DataType::Int64) {
            throw std::invalid_argument("KEY distribution of '" + name +
                                        "' must name an int64 column");
        }
    }

    TableEntry table_entry{schema, distribution, {}};
    table_entry.parts.reserve(num_slices());
    for (std::size_t slice = 0; slice < num_slices(); ++slice) {
        table_entry.parts.push_back(std::make_unique<Table>(name, schema, sort_key_col));
    }
    tables_.emplace(name, std::move(table_entry));
}

bool Cluster::has_table(const std::string& name) const { return tables_.contains(name); }

const Cluster::TableEntry& Cluster::entry(const std::string& name) const {
    auto it = tables_.find(name);
    if (it == tables_.end()) throw std::invalid_argument("unknown table '" + name + "'");
    return it->second;
}

const Schema& Cluster::schema(const std::string& name) const { return entry(name).schema; }

const Distribution& Cluster::distribution(const std::string& name) const {
    return entry(name).distribution;
}

Table& Cluster::local_part(std::size_t slice, const std::string& name) {
    const TableEntry& table_entry = entry(name);
    if (slice >= num_slices()) throw std::invalid_argument("slice out of range");
    if (table_entry.distribution.style == DistStyle::All &&
        slice != first_slice_of(node_of(slice))) {
        throw std::invalid_argument("ALL table '" + name +
                                    "' is stored only on the first slice of each node");
    }
    return *table_entry.parts[slice];
}

const Table& Cluster::scan_target(std::size_t slice, const std::string& name) const {
    const TableEntry& table_entry = entry(name);
    if (table_entry.distribution.style == DistStyle::All) {
        return *table_entry.parts[first_slice_of(node_of(slice))];
    }
    return *table_entry.parts[slice];
}

std::uint64_t Cluster::row_count(const std::string& name) const {
    const TableEntry& table_entry = entry(name);
    auto              rows_in     = [](const Table& part) {
        std::uint64_t rows = 0;
        for (const auto& segment : part.segments())
            rows += segment->num_rows();
        return rows;
    };

    if (table_entry.distribution.style == DistStyle::All) return rows_in(*table_entry.parts[0]);

    std::uint64_t total = 0;
    for (const auto& part : table_entry.parts)
        total += rows_in(*part);
    return total;
}

}  // namespace wb
