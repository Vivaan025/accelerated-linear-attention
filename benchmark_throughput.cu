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
    int H = 8;
    int D = 64;

    int seq_lengths[] = {128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144};
    int num_seq_tests = sizeof(seq_lengths) / sizeof(seq_lengths[0]);

    printf("RTX 5090 Throughput Benchmark\n");
    printf("Batch Scaling Test\n\n");

    for (int seq_idx = 0; seq_idx < num_seq_tests; seq_idx++) {
        int N = seq_lengths[seq_idx];

        printf("\n=== Sequence Length: %d tokens (%.0fK) ===\n", N, N / 1024.0f);
        printf("%-8s %-12s %-15s %-15s %-15s %-15s %-12s\n",
               "Batch", "VRAM (GB)", "Delta (ms)", "KDA (ms)", "LogLinear (ms)", "cuDNN FA (ms)", "Best");
        printf("-------------------------------------------------------------------------------------------\n");

        int batch_sizes[] = {1, 2, 4, 8, 16, 32, 64, 128, 256};
        int num_batch_tests = sizeof(batch_sizes) / sizeof(batch_sizes[0]);

        for (int b_idx = 0; b_idx < num_batch_tests; b_idx++) {
            int B = batch_sizes[b_idx];

            size_t total_size = (size_t)B * H * N * D * sizeof(float);
            size_t required_memory = total_size * 4;
            float vram_gb = required_memory / (1024.0f * 1024.0f * 1024.0f);

            float* Q = (float*)malloc(total_size);
            float* K = (float*)malloc(total_size);
            float* V = (float*)malloc(total_size);
            float* O = (float*)malloc(total_size);

            if (!Q || !K || !V || !O) {
                printf("%-8d %-12.2f CPU malloc failed\n", B, vram_gb);
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

            // Test Delta Attention
            float delta_time = -1;
            cudaGetLastError();
            run_delta_attention_5090(Q, K, V, O, B, H, N, D);
            cudaError_t err = cudaDeviceSynchronize();
            if (err == cudaSuccess) {
                cudaEventRecord(start);
                run_delta_attention_5090(Q, K, V, O, B, H, N, D);
                cudaEventRecord(stop);
                cudaEventSynchronize(stop);
                cudaEventElapsedTime(&delta_time, start, stop);
            }

            // Test KDA
            float kda_time = -1;
            cudaGetLastError();
            run_kda_optimized_5090(Q, K, V, O, B, H, N, D);
            err = cudaDeviceSynchronize();
            if (err == cudaSuccess) {
                cudaEventRecord(start);
                run_kda_optimized_5090(Q, K, V, O, B, H, N, D);
                cudaEventRecord(stop);
                cudaEventSynchronize(stop);
                cudaEventElapsedTime(&kda_time, start, stop);
            }

            // Test Log-Linear Attention
            float loglinear_time = -1;
            cudaGetLastError();
            run_log_linear_attention_5090(Q, K, V, O, B, H, N, D);
            err = cudaDeviceSynchronize();
            if (err == cudaSuccess) {
                cudaEventRecord(start);
                run_log_linear_attention_5090(Q, K, V, O, B, H, N, D);
                cudaEventRecord(stop);
                cudaEventSynchronize(stop);
                cudaEventElapsedTime(&loglinear_time, start, stop);
            }

            // Test cuDNN Flash Attention
            float cudnn_time = -1;
            cudaGetLastError();
            run_cudnn_flash_attention(Q, K, V, O, B, H, N, D);
            err = cudaDeviceSynchronize();
            if (err == cudaSuccess) {
                cudaEventRecord(start);
                run_cudnn_flash_attention(Q, K, V, O, B, H, N, D);
                cudaEventRecord(stop);
                cudaEventSynchronize(stop);
                cudaEventElapsedTime(&cudnn_time, start, stop);
            }

            // Find best
            float best_time = 1e9;
            const char* best = "N/A";
            if (delta_time > 0 && delta_time < best_time) { best_time = delta_time; best = "Delta"; }
            if (kda_time > 0 && kda_time < best_time) { best_time = kda_time; best = "KDA"; }
            if (loglinear_time > 0 && loglinear_time < best_time) { best_time = loglinear_time; best = "LogLinear"; }
            if (cudnn_time > 0 && cudnn_time < best_time) { best_time = cudnn_time; best = "cuDNN"; }

            char delta_str[16], kda_str[16], loglinear_str[16], cudnn_str[16];
            if (delta_time > 0) sprintf(delta_str, "%.3f", delta_time);
            else sprintf(delta_str, "OOM");
            if (kda_time > 0) sprintf(kda_str, "%.3f", kda_time);
            else sprintf(kda_str, "OOM");
            if (loglinear_time > 0) sprintf(loglinear_str, "%.3f", loglinear_time);
            else sprintf(loglinear_str, "OOM");
            if (cudnn_time > 0) sprintf(cudnn_str, "%.3f", cudnn_time);
            else sprintf(cudnn_str, "N/A");

            printf("%-8d %-12.2f %-15s %-15s %-15s %-15s %s\n",
                   B, vram_gb, delta_str, kda_str, loglinear_str, cudnn_str, best);

            free(Q);
            free(K);
            free(V);
            free(O);

            cudaEventDestroy(start);
            cudaEventDestroy(stop);
        }
    }

    return 0;
}
