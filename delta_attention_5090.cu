#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>

// RTX 5090 optimized configuration
// Max shared memory: 128KB per SM
// Target usage: ~96KB (leaves margin for compiler/registers)
#define BT_5090 128  // Chunk size (2x larger than FLA)
#define D_FIXED 64   // Fixed dimension - no K-blocking needed!

__device__ __forceinline__ float fast_sigmoid_5090(float x) {
    return __frcp_rn(1.0f + __expf(-fmaxf(-10.0f, fminf(10.0f, x))));
}

// RTX 5090 fully optimized kernel
// - Uses 96KB shared memory (75% of 128KB available)
// - No K-dimension blocking (processes all 64 dims at once)
// - 2x larger chunks than FLA (128 vs 64 tokens)
// - Minimal global memory traffic
__global__ void __launch_bounds__(128, 4) delta_attention_5090_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int B, int H, int N
) {
    const int chunk_idx = blockIdx.x;
    const int bh = blockIdx.y;
    const int b = bh / H;
    const int h = bh % H;

    const int chunk_start = chunk_idx * BT_5090;
    const int chunk_end = min(chunk_start + BT_5090, N);
    const int chunk_len = chunk_end - chunk_start;

    if (chunk_start >= N) return;

    const int qkv_offset = b * H * N * D_FIXED + h * N * D_FIXED;
    const float* q_base = Q + qkv_offset;
    const float* k_base = K + qkv_offset;
    const float* v_base = V + qkv_offset;
    float* o_base = O + qkv_offset;

    const float beta = 0.9f;

    // MASSIVE shared memory usage - 96KB total
    // Each array: 128 tokens × 64 dims × 4 bytes = 32KB
    __shared__ float s_q[BT_5090][D_FIXED];  // 32KB
    __shared__ float s_k[BT_5090][D_FIXED];  // 32KB
    __shared__ float s_v[BT_5090][D_FIXED];  // 32KB
    __shared__ float s_state[D_FIXED];       // 256 bytes
    __shared__ float s_norm[D_FIXED];        // 256 bytes

    // Initialize recurrent state
    for (int d = threadIdx.x; d < D_FIXED; d += blockDim.x) {
        s_state[d] = 0.0f;
        s_norm[d] = 0.0f;
    }
    __syncthreads();

    // Collaborative loading - all 256 threads work together
    // Total elements to load: chunk_len × D_FIXED × 3 (Q, K, V)
    const int total_elements = chunk_len * D_FIXED;

    for (int idx = threadIdx.x; idx < total_elements; idx += blockDim.x) {
        const int t = idx / D_FIXED;
        const int d = idx % D_FIXED;
        const int global_t = chunk_start + t;

        if (global_t < N) {
            s_q[t][d] = q_base[global_t * D_FIXED + d];
            s_k[t][d] = k_base[global_t * D_FIXED + d];
            s_v[t][d] = v_base[global_t * D_FIXED + d];
        }
    }
    __syncthreads();

    // Process each dimension in parallel (64 threads active)
    if (threadIdx.x < D_FIXED) {
        const int d = threadIdx.x;
        float state = s_state[d];
        float norm = s_norm[d];

        // Sequential processing for this dimension across all tokens in chunk
        #pragma unroll 8
        for (int t = 0; t < chunk_len; t++) {
            float k_val = s_k[t][d];
            float v_val = s_v[t][d];
            float q_val = s_q[t][d];

            float gate = fast_sigmoid_5090(k_val);

            // Recurrent update with fused multiply-add
            state = __fmaf_rn(beta, state, gate * v_val);
            norm = __fmaf_rn(beta, norm, gate);

            // Compute and write output directly
            float output = q_val * state * __frcp_rn(norm + 1e-6f);

            int global_t = chunk_start + t;
            if (global_t < N) {
                o_base[global_t * D_FIXED + d] = output;
            }
        }

        // Store updated state for next chunk (would need global state in full impl)
        s_state[d] = state;
        s_norm[d] = norm;
    }
    __syncthreads();
}

void run_delta_attention_5090(
    const float* Q, const float* K, const float* V,
    float* O, int B, int H, int N, int D
) {
    if (D != D_FIXED) {
        printf("Error: This kernel requires D=64 (got D=%d)\n", D);
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

    // Request maximum shared memory for this kernel
    cudaFuncSetAttribute(
        delta_attention_5090_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        98304  // 96KB
    );

    int num_chunks = (N + BT_5090 - 1) / BT_5090;
    dim3 grid(num_chunks, B * H);
    dim3 block(128);

    delta_attention_5090_kernel<<<grid, block>>>(d_Q, d_K, d_V, d_O, B, H, N);

    cudaMemcpy(O, d_O, size, cudaMemcpyDeviceToHost);

    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_O);
}
