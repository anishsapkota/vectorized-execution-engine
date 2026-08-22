CXX := g++
CXXFLAGS := -std=c++20 -Wall -Wextra -Wpedantic -O3 -march=native -g -Isrc
LDFLAGS :=

SRC := $(wildcard src/storage/*.cpp) $(wildcard src/exec/*.cpp) src/main.cpp
OBJ := $(SRC:.cpp=.o)
BIN := wb

TEST_SRC := $(wildcard tests/*.cpp) $(wildcard src/storage/*.cpp) $(wildcard src/exec/*.cpp)
TEST_OBJ := $(TEST_SRC:.cpp=.o)
TEST_BIN := wb_test

GEN_SRC := src/tools/gen.cpp
GEN_BIN := wb_gen

.PHONY: all test clean

all: $(BIN) $(GEN_BIN)

$(BIN): $(OBJ)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)


$(GEN_BIN): $(GEN_SRC:.cpp=.o)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

test: $(TEST_BIN)
	./$(TEST_BIN)

$(TEST_BIN): $(TEST_OBJ)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) $(TEST_OBJ) $(BIN) $(TEST_BIN)