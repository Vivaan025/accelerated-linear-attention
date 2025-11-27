#include <iostream>
#include <chrono>
#include "kernel.cu"

int main() {
    const int n = 1000000; // Size of arrays

    // Record start time
    auto start = std::chrono::high_resolution_clock::now();

    // Run the CUDA kernel
    runKernel(n);

    // Record end time
    auto end = std::chrono::high_resolution_clock::now();

    // Calculate and print execution time
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Execution time: " << duration.count() << " microseconds" << std::endl;

    return 0;
}
