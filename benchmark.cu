#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "delta_attention_5090.cu"
#include "kda_optimized_5090.cu"
#include "log_linear_attn_5090.cu"
#include "flash_attention_cudnn.cu"

void init_random(float* data, int size) {
    for (int i = 0; i < size; i++) {
        data[i] = (float)rand() / RAND_MAX;
    }
}

size_t get_free_memory() {
    size_t free_bytes, total_bytes;
    cudaMemGetInfo(&free_bytes, &total_bytes);
    return free_bytes;
}

int main() {
    int B = 2;
    int H = 8;
    int D = 64;

    printf("RTX 5090 Latency Benchmark\n\n");

    int seq_lengths[] = {128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144};
    int num_tests = sizeof(seq_lengths) / sizeof(seq_lengths[0]);

    printf("%-10s %-12s %-15s %-15s %-15s %-15s %-12s\n",
           "Seq Len", "VRAM (GB)", "Delta (ms)", "KDA (ms)", "LogLinear (ms)", "cuDNN FA (ms)", "Best");
    printf("-------------------------------------------------------------------------------------------\n");

    for (int i = 0; i < num_tests; i++) {
        int N = seq_lengths[i];
        size_t total_size = (size_t)B * H * N * D * sizeof(float);
        size_t required_memory = total_size * 4;
        float memory_gb = required_memory / (1024.0f * 1024.0f * 1024.0f);

        size_t free_mem = get_free_memory();
        if (required_memory > free_mem * 0.9f) {
            printf("%-10d %-12.2f SKIPPED (Out of memory)\n", N, memory_gb);
            continue;
        }

        float* Q = (float*)malloc(total_size);
        float* K = (float*)malloc(total_size);
        float* V = (float*)malloc(total_size);
        float* O = (float*)malloc(total_size);

        if (!Q || !K || !V || !O) {
            printf("%-10d %-12.2f SKIPPED (CPU allocation failed)\n", N, memory_gb);
            if (Q) free(Q);
            if (K) free(K);
            if (V) free(V);
            if (O) free(O);
            break;
        }

        init_random(Q, B * H * N * D);
        init_random(K, B * H * N * D);
        init_random(V, B * H * N * D);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // Warmup Delta
        run_delta_attention_5090(Q, K, V, O, B, H, N, D);
        cudaDeviceSynchronize();

        // Benchmark Delta Attention
        cudaEventRecord(start);
        run_delta_attention_5090(Q, K, V, O, B, H, N, D);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float delta_time = 0;
        cudaEventElapsedTime(&delta_time, start, stop);

        // Warmup KDA
        run_kda_optimized_5090(Q, K, V, O, B, H, N, D);
        cudaDeviceSynchronize();

        // Benchmark KDA
        cudaEventRecord(start);
        run_kda_optimized_5090(Q, K, V, O, B, H, N, D);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float kda_time = 0;
        cudaEventElapsedTime(&kda_time, start, stop);

        // Warmup Log-Linear
        run_log_linear_attention_5090(Q, K, V, O, B, H, N, D);
        cudaDeviceSynchronize();

        // Benchmark Log-Linear Attention
        cudaEventRecord(start);
        run_log_linear_attention_5090(Q, K, V, O, B, H, N, D);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float loglinear_time = 0;
        cudaEventElapsedTime(&loglinear_time, start, stop);

        // Warmup cuDNN
        run_cudnn_flash_attention(Q, K, V, O, B, H, N, D);
        cudaDeviceSynchronize();

        // cuDNN Flash Attention (FP16 internally with FP32 conversion)
        cudaEventRecord(start);
        run_cudnn_flash_attention(Q, K, V, O, B, H, N, D);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float cudnn_time = 0;
        cudaEventElapsedTime(&cudnn_time, start, stop);

        float best_time = min(min(min(delta_time, kda_time), loglinear_time), cudnn_time);
        const char* best = (best_time == delta_time) ? "Delta" :
                           (best_time == kda_time) ? "KDA" :
                           (best_time == loglinear_time) ? "LogLinear" : "cuDNN";

        printf("%-10d %-12.2f %-15.3f %-15.3f %-15.3f %-15.3f %s\n",
               N, memory_gb, delta_time, kda_time, loglinear_time, cudnn_time, best);

        free(Q);
        free(K);
        free(V);
        free(O);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    return 0;
}
