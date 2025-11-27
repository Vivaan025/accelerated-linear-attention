#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>

#define LOG_D 64
#define LOG_CHUNK 128
#define LOG_THREADS 256
#define MAX_LOG_LEVELS 12

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int mask = 16; mask > 0; mask /= 2) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

__device__ __forceinline__ float warp_prefix_sum(float val, int lane_id) {
    #pragma unroll
    for (int offset = 1; offset < 32; offset *= 2) {
        float n = __shfl_up_sync(0xffffffff, val, offset);
        if (lane_id >= offset) val += n;
    }
    return val;
}

__global__ void __launch_bounds__(LOG_THREADS, 2) log_linear_attn_fwd_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    const float* __restrict__ gates,
    const float* __restrict__ level_scales,
    float* __restrict__ O,
    float* __restrict__ kv_states_global,
    int B, int H, int N, int num_levels
) {
    const int bh = blockIdx.x;
    const int b = bh / H;
    const int h = bh % H;
    const int tid = threadIdx.x;
    const int lane_id = tid % 32;

    const int qkv_offset = b * H * N * LOG_D + h * N * LOG_D;
    const float* q_base = Q + qkv_offset;
    const float* k_base = K + qkv_offset;
    const float* v_base = V + qkv_offset;
    float* o_base = O + qkv_offset;

    const int gate_offset = (b * H + h) * N;
    const float* gate_base = gates + gate_offset;

    const int state_offset = ((b * H + h) * num_levels) * LOG_D * LOG_D;
    float* kv_states = kv_states_global + state_offset;

    __shared__ float s_q[LOG_CHUNK][LOG_D];
    __shared__ float s_k[LOG_CHUNK][LOG_D];
    __shared__ float s_v[LOG_CHUNK][LOG_D];
    __shared__ float s_cumsum[LOG_CHUNK];
    __shared__ float s_scales[MAX_LOG_LEVELS];
    __shared__ float s_kv_cache[LOG_D][LOG_D];

    if (tid < num_levels) {
        s_scales[tid] = __ldg(&level_scales[tid]);
    }
    __syncthreads();

    const int num_chunks = (N + LOG_CHUNK - 1) / LOG_CHUNK;

    for (int chunk_idx = 0; chunk_idx < num_chunks; chunk_idx++) {
        const int chunk_start = chunk_idx * LOG_CHUNK;
        const int chunk_end = min(chunk_start + LOG_CHUNK, N);
        const int chunk_len = chunk_end - chunk_start;

        for (int idx = tid; idx < chunk_len * (LOG_D / 4); idx += LOG_THREADS) {
            int t = idx / (LOG_D / 4);
            int d4 = idx % (LOG_D / 4);
            int global_t = chunk_start + t;

            float4 q4 = *reinterpret_cast<const float4*>(&q_base[global_t * LOG_D + d4 * 4]);
            float4 k4 = *reinterpret_cast<const float4*>(&k_base[global_t * LOG_D + d4 * 4]);
            float4 v4 = *reinterpret_cast<const float4*>(&v_base[global_t * LOG_D + d4 * 4]);

            *reinterpret_cast<float4*>(&s_q[t][d4 * 4]) = q4;
            *reinterpret_cast<float4*>(&s_k[t][d4 * 4]) = k4;
            *reinterpret_cast<float4*>(&s_v[t][d4 * 4]) = v4;
        }
        __syncthreads();

        for (int base = 0; base < chunk_len; base += 32) {
            int t = base + lane_id;
            float gate_val = (t < chunk_len) ? __ldg(&gate_base[chunk_start + t]) : 0.0f;
            float prefix = warp_prefix_sum(gate_val, lane_id);

            if (t < chunk_len) {
                s_cumsum[t] = prefix + (base > 0 ? s_cumsum[base - 1] : 0.0f);
            }
        }
        __syncthreads();

        for (int i = tid; i < chunk_len; i += LOG_THREADS) {
            int global_i = chunk_start + i;
            float cumsum_i = s_cumsum[i];

            float output_reg[LOG_D];
            #pragma unroll
            for (int d = 0; d < LOG_D; d++) {
                output_reg[d] = 0.0f;
            }

            for (int level = 0; level < num_levels; level++) {
                int j_local = i ^ (1 << level);

                if (j_local >= 0 && j_local < i && j_local < chunk_len) {
                    float cumsum_j = (j_local > 0) ? s_cumsum[j_local - 1] : 0.0f;
                    float segsum = cumsum_i - cumsum_j;
                    float h_val = __expf(segsum * s_scales[level]);

                    float qk_dot = 0.0f;
                    #pragma unroll
                    for (int d = 0; d < LOG_D; d++) {
                        qk_dot = __fmaf_rn(s_q[i][d], s_k[j_local][d], qk_dot);
                    }

                    float weight = h_val * qk_dot;

                    #pragma unroll
                    for (int d = 0; d < LOG_D; d++) {
                        output_reg[d] = __fmaf_rn(weight, s_v[j_local][d], output_reg[d]);
                    }
                }
            }

            if (chunk_idx > 0) {
                for (int level = 0; level < num_levels; level++) {
                    int level_stride = 1 << level;
                    if (chunk_idx >= level_stride && (chunk_idx % level_stride == 0)) {
                        if (i == 0) {
                            for (int idx = tid; idx < LOG_D * LOG_D; idx += LOG_THREADS) {
                                s_kv_cache[idx / LOG_D][idx % LOG_D] = kv_states[level * LOG_D * LOG_D + idx];
                            }
                        }
                        __syncthreads();

                        float decay = __expf(-cumsum_i * s_scales[level]);

                        #pragma unroll
                        for (int d1 = 0; d1 < LOG_D; d1++) {
                            float q_val = s_q[i][d1];
                            float decay_q = decay * q_val;
                            #pragma unroll
                            for (int d2 = 0; d2 < LOG_D; d2++) {
                                output_reg[d2] = __fmaf_rn(decay_q, s_kv_cache[d1][d2], output_reg[d2]);
                            }
                        }
                        __syncthreads();
                    }
                }
            }

            #pragma unroll
            for (int d4 = 0; d4 < LOG_D / 4; d4++) {
                float4 out4;
                out4.x = output_reg[d4 * 4];
                out4.y = output_reg[d4 * 4 + 1];
                out4.z = output_reg[d4 * 4 + 2];
                out4.w = output_reg[d4 * 4 + 3];
                *reinterpret_cast<float4*>(&o_base[global_i * LOG_D + d4 * 4]) = out4;
            }
        }
        __syncthreads();

        for (int level = 0; level < num_levels; level++) {
            int level_stride = 1 << level;

            if ((chunk_idx + 1) % level_stride == 0) {
                for (int idx = tid; idx < LOG_D * LOG_D; idx += LOG_THREADS) {
                    int d1 = idx / LOG_D;
                    int d2 = idx % LOG_D;

                    float kv_sum = 0.0f;
                    #pragma unroll 4
                    for (int t = 0; t < chunk_len; t++) {
                        float gate_decay = __expf(-s_cumsum[t] * s_scales[level]);
                        float kv_contrib = gate_decay * s_k[t][d1];
                        kv_sum = __fmaf_rn(kv_contrib, s_v[t][d2], kv_sum);
                    }

                    if (chunk_idx == 0) {
                        kv_states[level * LOG_D * LOG_D + idx] = kv_sum;
                    } else {
                        float prev_decay = __expf(-s_cumsum[chunk_len - 1] * s_scales[level]);
                        kv_states[level * LOG_D * LOG_D + idx] =
                            kv_states[level * LOG_D * LOG_D + idx] * prev_decay + kv_sum;
                    }
                }
                __syncthreads();
            }
        }
    }
}

void run_log_linear_attention_5090(
    const float* Q, const float* K, const float* V,
    float* O, int B, int H, int N, int D
) {
    if (D != LOG_D) {
        printf("Error: Log-linear attention requires D=%d (got D=%d)\n", LOG_D, D);
        return;
    }

    float *d_Q, *d_K, *d_V, *d_O, *d_gates, *d_level_scales, *d_kv_states;
    size_t size = (size_t)B * H * N * D * sizeof(float);

    int num_levels = min((int)ceil(log2((float)N)), MAX_LOG_LEVELS);
    size_t gate_size = (size_t)B * H * N * sizeof(float);
    size_t scale_size = num_levels * sizeof(float);
    size_t state_size = (size_t)B * H * num_levels * LOG_D * LOG_D * sizeof(float);

    cudaError_t err;
    err = cudaMalloc(&d_Q, size);
    if (err != cudaSuccess) return;
    err = cudaMalloc(&d_K, size);
    if (err != cudaSuccess) { cudaFree(d_Q); return;
}
    err = cudaMalloc(&d_V, size);
    if (err != cudaSuccess) { cudaFree(d_Q); cudaFree(d_K); return; }
    err = cudaMalloc(&d_O, size);
    if (err != cudaSuccess) { cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); return; }
    err = cudaMalloc(&d_gates, gate_size);
    if (err != cudaSuccess) {
        cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O);
        return;
    }
    err = cudaMalloc(&d_level_scales, scale_size);
    if (err != cudaSuccess) {
        cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O); cudaFree(d_gates);
        return;
    }
    err = cudaMalloc(&d_kv_states, state_size);
    if (err != cudaSuccess) {
        cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O); cudaFree(d_gates); cudaFree(d_level_scales);
        return;
    }

    float* h_gates = (float*)malloc(gate_size);
    float* h_level_scales = (float*)malloc(scale_size);

    for (int i = 0; i < B * H * N; i++) {
        h_gates[i] = 0.1f * ((float)rand() / RAND_MAX);
    }

    for (int i = 0; i < num_levels; i++) {
        h_level_scales[i] = 1.0f / (1 << i);
    }

    cudaMemcpy(d_Q, Q, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, K, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, V, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_gates, h_gates, gate_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_level_scales, h_level_scales, scale_size, cudaMemcpyHostToDevice);

    cudaMemset(d_O, 0, size);
    cudaMemset(d_kv_states, 0, state_size);

    dim3 grid(B * H);
    dim3 block(LOG_THREADS);

    log_linear_attn_fwd_kernel<<<grid, block>>>(
        d_Q, d_K, d_V, d_gates, d_level_scales, d_O, d_kv_states,
        B, H, N, num_levels
    );

    cudaMemcpy(O, d_O, size, cudaMemcpyDeviceToHost);

    free(h_gates);
    free(h_level_scales);
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O);
    cudaFree(d_gates); cudaFree(d_level_scales); cudaFree(d_kv_states);
}
