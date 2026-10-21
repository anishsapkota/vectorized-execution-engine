CXX := g++
CXXFLAGS := -std=c++20 -Wall -Wextra -Wpedantic -O3 -march=native -g -Isrc -pthread -MMD -MP \
            -DWB_SRC_DIR=\"$(CURDIR)/src\"
LDFLAGS :=
ifneq ($(shell uname),Darwin)
LDFLAGS += -ldl
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

.PHONY: all test bench clean

all: $(BIN) $(GEN_BIN) $(DIST_BIN) $(BENCH_BIN) $(CODEGEN_BIN)

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

data/readings.csv: $(GEN_BIN)
	./$(GEN_BIN) --rows 10000000 --out $@

bench: $(BENCH_BIN) data/readings.csv
	./$(BENCH_BIN) --csv data/readings.csv

test: $(TEST_BIN)
	./$(TEST_BIN)

$(TEST_BIN): $(TEST_OBJ)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

-include $(OBJ:.o=.d) $(TEST_OBJ:.o=.d) $(DIST_SRC:.cpp=.d) $(GEN_SRC:.cpp=.d) $(BENCH_SRC:.cpp=.d) $(CODEGEN_SRC:.cpp=.d)

clean:
	rm -f $(OBJ) $(TEST_OBJ) src/tools/*.o $(BIN) $(TEST_BIN) $(GEN_BIN) $(DIST_BIN) $(BENCH_BIN) $(CODEGEN_BIN)
	find src tests -name "*.d" -delete
