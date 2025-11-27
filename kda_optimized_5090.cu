#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>

// Highly Optimized Kimi Delta Attention for RTX 5090
// Strategy: Reduce register pressure + better memory layout
// No wmma due to algorithm mismatch - focus on memory optimizations

#define KDA_OPT_D 64
#define KDA_OPT_CHUNK 64    // Increased from 32 for better amortization
#define KDA_OPT_THREADS 128 // 4 warps

__device__ __forceinline__ float fast_sigmoid_opt(float x) {
    return __frcp_rn(1.0f + __expf(-fmaxf(-10.0f, fminf(10.0f, x))));
}

__device__ __forceinline__ float warp_reduce_sum_kda(float val) {
    #pragma unroll
    for (int mask = 16; mask > 0; mask /= 2) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

// Optimized KDA kernel - register pressure reduction
// Grid: (num_chunks, H, B)
// Block: 128 threads
__global__ void kda_optimized_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int B, int H, int N
) {
    const int chunk_idx = blockIdx.x;
    const int h = blockIdx.y;
    const int b = blockIdx.z;
    const int tid = threadIdx.x;

    const int chunk_start = chunk_idx * KDA_OPT_CHUNK;
    const int chunk_end = min(chunk_start + KDA_OPT_CHUNK, N);
    const int chunk_len = chunk_end - chunk_start;

    if (chunk_start >= N) return;

    const int offset = b * H * N * KDA_OPT_D + h * N * KDA_OPT_D;
    const float* q_base = Q + offset;
    const float* k_base = K + offset;
    const float* v_base = V + offset;
    float* o_base = O + offset;

    __shared__ half s_state[KDA_OPT_D][KDA_OPT_D];  // 8KB
    __shared__ float s_q[KDA_OPT_CHUNK][KDA_OPT_D]; // 16KB
    __shared__ float s_k[KDA_OPT_CHUNK][KDA_OPT_D]; // 16KB
    __shared__ float s_v[KDA_OPT_CHUNK][KDA_OPT_D]; // 16KB
    __shared__ float s_temp[KDA_OPT_D][KDA_OPT_D];  // 16KB - workspace
    __shared__ float s_alpha[KDA_OPT_D];            // 256B
    __shared__ float s_beta;                        // 4B
    // Total: 72KB shared memory

    // Initialize state
    for (int idx = tid; idx < KDA_OPT_D * KDA_OPT_D; idx += KDA_OPT_THREADS) {
        s_state[idx / KDA_OPT_D][idx % KDA_OPT_D] = __float2half(0.0f);
    }
    __syncthreads();

    // Load chunk with vectorized loads
    for (int idx = tid; idx < chunk_len * (KDA_OPT_D / 4); idx += KDA_OPT_THREADS) {
        int t = idx / (KDA_OPT_D / 4);
        int d4 = idx % (KDA_OPT_D / 4);
        int global_t = chunk_start + t;
        if (global_t < N) {
            float4 q4 = *reinterpret_cast<const float4*>(&q_base[global_t * KDA_OPT_D + d4 * 4]);
            float4 k4 = *reinterpret_cast<const float4*>(&k_base[global_t * KDA_OPT_D + d4 * 4]);
            float4 v4 = *reinterpret_cast<const float4*>(&v_base[global_t * KDA_OPT_D + d4 * 4]);
            *reinterpret_cast<float4*>(&s_q[t][d4 * 4]) = q4;
            *reinterpret_cast<float4*>(&s_k[t][d4 * 4]) = k4;
            *reinterpret_cast<float4*>(&s_v[t][d4 * 4]) = v4;
        }
    }
    __syncthreads();

    __shared__ float s_beta_sums[4];  // One per warp

    // Process each token
    for (int t = 0; t < chunk_len; t++) {
        // Compute gates and beta in parallel
        float local_sum = 0.0f;
        for (int d = tid; d < KDA_OPT_D; d += KDA_OPT_THREADS) {
            float gate = fast_sigmoid_opt(s_k[t][d]);
            s_alpha[d] = gate;
            local_sum += gate;
        }

        // Warp-level reduction
        int warp_id = tid / 32;
        int lane_id = tid % 32;
        local_sum = warp_reduce_sum_kda(local_sum);

        // First thread of each warp writes to shared memory
        if (lane_id == 0) {
            s_beta_sums[warp_id] = local_sum;
        }
        __syncthreads();

        // Final reduction and broadcast
        if (tid == 0) {
            float total = s_beta_sums[0] + s_beta_sums[1] + s_beta_sums[2] + s_beta_sums[3];
            s_beta = total / KDA_OPT_D;
        }
        __syncthreads();

        // Step 1: Compute k ⊗ v^T in shared memory
        for (int idx = tid; idx < KDA_OPT_D * KDA_OPT_D; idx += KDA_OPT_THREADS) {
            int row = idx / KDA_OPT_D;
            int col = idx % KDA_OPT_D;
            s_temp[row][col] = s_k[t][row] * s_v[t][col] * s_beta;
        }
        __syncthreads();

        // Step 2: Apply delta rule
        for (int idx = tid; idx < KDA_OPT_D * KDA_OPT_D; idx += KDA_OPT_THREADS) {
            int row = idx / KDA_OPT_D;
            int col = idx % KDA_OPT_D;

            float state_val = __half2float(s_state[row][col]);
            float alpha_row = s_alpha[row];

            // Apply Diag(α) to current state
            float diag_state = state_val * alpha_row;

            // Compute delta correction: β (k k^T) Diag(α) S
            float k_row = s_k[t][row];
            float correction = 0.0f;
            #pragma unroll 8
            for (int k_idx = 0; k_idx < KDA_OPT_D; k_idx++) {
                float kkt = k_row * s_k[t][k_idx];
                float contrib = s_beta * kkt * s_alpha[k_idx];
                correction = __fmaf_rn(contrib, __half2float(s_state[k_idx][col]), correction);
            }

            // S_new = Diag(α)*S - β*kk^T*Diag(α)*S + β*kv^T
            float new_state = diag_state - correction + s_temp[row][col];
            s_state[row][col] = __float2half(new_state);
        }
        __syncthreads();

        // Compute output: o = q @ S
        int global_t = chunk_start + t;
        if (global_t < N) {
            for (int d = tid; d < KDA_OPT_D; d += KDA_OPT_THREADS) {
                float sum = 0.0f;
                #pragma unroll 8
                for (int k_idx = 0; k_idx < KDA_OPT_D; k_idx++) {
                    sum = __fmaf_rn(s_q[t][k_idx], __half2float(s_state[k_idx][d]), sum);
                }
                o_base[global_t * KDA_OPT_D + d] = sum;
            }
        }
        __syncthreads();
    }
}

void run_kda_optimized_5090(
    const float* Q, const float* K, const float* V,
    float* O, int B, int H, int N, int D
) {
    if (D != KDA_OPT_D) {
        printf("Error: Optimized KDA requires D=%d (got D=%d)\n", KDA_OPT_D, D);
        return;
    }

    float *d_Q, *d_K, *d_V, *d_O;
    size_t size = (size_t)B * H * N * D * sizeof(float);

    cudaError_t err;
    err = cudaMalloc(&d_Q, size);
    if (err != cudaSuccess) return;
    err = cudaMalloc(&d_K, size);
    if (err != cudaSuccess) { cudaFree(d_Q); return; }
    err = cudaMalloc(&d_V, size);
    if (err != cudaSuccess) { cudaFree(d_Q); cudaFree(d_K); return; }
    err = cudaMalloc(&d_O, size);
    if (err != cudaSuccess) { cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); return; }

    cudaMemcpy(d_Q, Q, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, K, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, V, size, cudaMemcpyHostToDevice);

    int num_chunks = (N + KDA_OPT_CHUNK - 1) / KDA_OPT_CHUNK;
    dim3 grid(num_chunks, H, B);
    dim3 block(KDA_OPT_THREADS);

    // Clear any previous errors
    cudaGetLastError();

    kda_optimized_kernel<<<grid, block>>>(
        d_Q, d_K, d_V, d_O, B, H, N
    );

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("KDA Opt error: %s\n", cudaGetErrorString(err));
    }

    cudaMemcpy(O, d_O, size, cudaMemcpyDeviceToHost);

    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_O);
}
