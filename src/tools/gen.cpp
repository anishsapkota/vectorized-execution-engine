#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>
#include <string>

int main(int argc, char** argv) {
    // 1) Parse args: --rows N --out path. Provide defaults.
    std::int64_t rows = 10'000'000;
    std::string  out  = "data/readings.csv";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--rows" && i + 1 < argc) {
            rows = std::stoll(argv[++i]);
        } else if (arg == "--out" && i + 1 < argc) {
            out = argv[++i];
        } else {
            std::cerr << "unknown arg: " << arg << "\n";
            return 1;
        }
    }

    // 2) Open the output stream.
    std::ofstream f{out};
    if (!f) {
        std::cerr << "cannot open " << out << " for writing\n";
        return 1;
    }
    f << "device_id,ts,value\n";

    // 3) Set up RNG. Fixed seed so the dataset is reproducible across runs.
    std::mt19937_64 rng{42};

    // For Zipfian device_id over D devices, the trick is to sample from a power
    // distribution rather than fight with rejection sampling. With device IDs
    // 1..D, sampling pow(uniform(0,1), exponent) * D yields a skewed distribution:
    // small exponent -> uniform-ish, larger -> more skewed.
    const std::int64_t D             = 10'000;
    const double       zipf_exponent = 2.0;  // moderately skewed

    std::uniform_real_distribution<double> unif{0.0, 1.0};
    std::normal_distribution<double>       value_dist{50.0, 15.0};
    std::uniform_int_distribution<int>     jitter{-30, 30};

    // 4) Generate rows. ts starts at some epoch and advances by 1 second per
    // row, with small jitter to simulate real ingestion.
    std::int64_t ts = 1'700'000'000;  // 2023-11-14 UTC, doesn't matter

    for (std::int64_t i = 0; i < rows; ++i) {
        double       u      = unif(rng);
        std::int64_t device = 1 + static_cast<std::int64_t>(std::pow(u, zipf_exponent) * D);

        ts += 1 + jitter(rng);  // mostly increases, occasionally goes backward
        double value = value_dist(rng);

        f << device << ',' << ts << ',' << value << '\n';

        if (i > 0 && i % 1'000'000 == 0) {
            std::cerr << i / 1'000'000 << "M rows...\n";
        }
    }

    std::cerr << "wrote " << rows << " rows to " << out << "\n";
}