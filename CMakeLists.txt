cmake_minimum_required(VERSION 3.18)
project(cuda_benchmarks LANGUAGES CXX CUDA)

# Enable CUDA
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 90)

# Set CUDA flags (works on both Windows and Linux)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -use_fast_math --ptxas-options=-v")

# Add benchmark executable (includes other .cu files)
add_executable(benchmark benchmark.cu)

# Add throughput benchmark executable (includes other .cu files)
add_executable(benchmark_throughput benchmark_throughput.cu)

# Output to build/ directory on all platforms
set_target_properties(benchmark benchmark_throughput PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}
)
