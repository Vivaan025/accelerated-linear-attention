# Benchmark Running Guide

This guide provides steps to compile and run the attention throughput benchmarks on your machine.

## Prerequisites

Ensure you have the following installed:

- **CUDA Toolkit** (specifically `nvcc`)
- **CMake** (3.18 or higher)
- **Make** (optional, but recommended for ease of use)
- **C++ Compiler** (g++ or MSVC)

## Quick Start (Using Make)

If you have `make` installed, you can simply run:

```bash
# Clean previous builds
make clean

# Run the throughput benchmark (generates the table of figures)
make throughput
```

This will compile the code and execute `benchmark_throughput`, which prints the performance comparison table.

## Manual Compilation (Using CMake)

If you don't have `make`, use these standard CMake commands:

### Linux / macOS

```bash
# Create build directory
mkdir -p build
cd build

# Configure (Release mode for maximum performance)
cmake -DCMAKE_BUILD_TYPE=Release ..

# Build
cmake --build . --config Release

# Run
./benchmark_throughput
```

### Windows (PowerShell)

```powershell
# Create build directory
mkdir build
cd build

# Configure
cmake ..

# Build
cmake --build . --config Release

# Run
.\Release\benchmark_throughput.exe
```

## Understanding the Results

The benchmark will output a table for various sequence lengths (128 to 256K).

- **Delta (ms)**: Our optimized Delta Attention (5090 kernel).
- **KDA (ms)**: The "Proper" Kimi Delta Attention (more complex).
- **LogLinear (ms)**: Log-space implementation.
- **cuDNN FA**: NVIDIA's Flash Attention (will show 0 or N/A if cuDNN is not linked).

Look for the **"Best"** column to see which implementation wins at each sequence length.
