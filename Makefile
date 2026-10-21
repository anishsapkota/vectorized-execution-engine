CXX := g++
CXXFLAGS := -std=c++20 -Wall -Wextra -Wpedantic -O3 -march=native -g -Isrc -pthread -MMD -MP \
            -DWB_SRC_DIR=\"$(CURDIR)/src\"
LDFLAGS :=
ifneq ($(shell uname),Darwin)
LDFLAGS += -ldl
endif

# ALLOCATOR=jemalloc links every binary against jemalloc (brew install jemalloc). Only the link
# step differs, so switch with `make clean` first. Generated query libraries need no change: they
# call malloc, which jemalloc serves once it registers as the default malloc zone.
ALLOCATOR ?= system
ifeq ($(ALLOCATOR),jemalloc)
JEMALLOC_PREFIX ?= $(shell brew --prefix jemalloc 2>/dev/null)
LDFLAGS += -L$(JEMALLOC_PREFIX)/lib -ljemalloc -Wl,-rpath,$(JEMALLOC_PREFIX)/lib
else ifneq ($(ALLOCATOR),system)
$(error ALLOCATOR must be system or jemalloc)
endif

LIB_SRC := $(wildcard src/storage/*.cpp) $(wildcard src/exec/*.cpp) $(wildcard src/dist/*.cpp) \
           $(wildcard src/plan/*.cpp) $(wildcard src/codegen/*.cpp)
SRC := $(LIB_SRC) src/main.cpp
OBJ := $(SRC:.cpp=.o)
BIN := wb

TEST_SRC := $(wildcard tests/*.cpp) $(LIB_SRC)
TEST_OBJ := $(TEST_SRC:.cpp=.o)
TEST_BIN := wb_test

GEN_SRC := src/tools/gen.cpp
GEN_BIN := wb_gen

DIST_SRC := src/tools/dist_bench.cpp $(LIB_SRC)
DIST_BIN := wb_dist

BENCH_SRC := src/tools/bench.cpp $(LIB_SRC)
BENCH_OBJ := $(BENCH_SRC:.cpp=.o)
BENCH_BIN := wb_bench

CODEGEN_SRC := src/tools/codegen_bench.cpp $(LIB_SRC)
CODEGEN_BIN := wb_codegen

MEM_SRC := src/tools/mem_probe.cpp $(LIB_SRC)
MEM_BIN := wb_mem

.PHONY: all test bench clean lint fix-includes install-hooks

all: $(BIN) $(GEN_BIN) $(DIST_BIN) $(BENCH_BIN) $(CODEGEN_BIN) $(MEM_BIN)

$(BIN): $(OBJ)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

$(GEN_BIN): $(GEN_SRC:.cpp=.o)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

$(DIST_BIN): $(DIST_SRC:.cpp=.o)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

$(BENCH_BIN): $(BENCH_OBJ)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

$(CODEGEN_BIN): $(CODEGEN_SRC:.cpp=.o)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

$(MEM_BIN): $(MEM_SRC:.cpp=.o)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

data/readings.csv: $(GEN_BIN)
	./$(GEN_BIN) --rows 10000000 --out $@

bench: $(BENCH_BIN) data/readings.csv
	./$(BENCH_BIN) --csv data/readings.csv

test: $(TEST_BIN)
	./$(TEST_BIN)

# Unused #includes (clang-tidy misc-include-cleaner, see .clang-tidy and tools/fix_includes.sh).
lint:
	tools/fix_includes.sh --check

fix-includes:
	tools/fix_includes.sh --test

# Runs tools/git-hooks/pre-commit on every commit: removes unused includes from staged files.
install-hooks:
	git config core.hooksPath tools/git-hooks

$(TEST_BIN): $(TEST_OBJ)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

-include $(OBJ:.o=.d) $(TEST_OBJ:.o=.d) $(DIST_SRC:.cpp=.d) $(GEN_SRC:.cpp=.d) $(BENCH_SRC:.cpp=.d) $(CODEGEN_SRC:.cpp=.d) $(MEM_SRC:.cpp=.d)

clean:
	rm -f $(OBJ) $(TEST_OBJ) src/tools/*.o $(BIN) $(TEST_BIN) $(GEN_BIN) $(DIST_BIN) $(BENCH_BIN) $(CODEGEN_BIN) $(MEM_BIN)
	find src tests -name "*.d" -delete
