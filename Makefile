# Compiler settings
CXX = g++-m
CXXFLAGS = -fopt-info-vec-optimized -ftree-vectorize -ffast-math -std=c++26 -march=native -O3 -I./include
# CXXFLAGS = -O3 -march=native -ftree-vectorize -f -std=c++26 -I./include

# Target executable
TARGET = aats_benchmarks
SRC = benchmarks/benchmarks_dep.cpp

.PHONY: all compile run julia plot clean full_pipeline

all: compile

compile: $(TARGET)

$(TARGET): $(SRC)
	@echo "Compiling AATS benchmarks..."
	$(CXX) $(CXXFLAGS) $< -o $@
	@echo "Compilation successful. Executable: $(TARGET)"

run: $(TARGET)
	@echo "Running C++ AATS Benchmarks..."
	./$(TARGET)

julia:
	@echo "Running Julia SciML Baseline Benchmarks..."
	julia benchmarks/julia_bench.jl

plot:
	@echo "Generating comparison plots..."
	python3.11 scripts/plotting.py

clean:
	@echo "Cleaning up generated files..."
	rm -f $(TARGET) *.csv *.png *.jpg

# Helper to run the entire reproducibility pipeline at once
full_pipeline: compile run julia plot
	@echo "Full pipeline complete! Check the root directory for plots and CSVs."
