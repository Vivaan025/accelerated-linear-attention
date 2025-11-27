#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <unordered_map>
#include <memory>
#include <map>

#ifdef CUDNN_AVAILABLE
#include <cudnn.h>
#include <cudnn_frontend.h>
namespace fe = cudnn_frontend;

#define Q_UID 1
#define K_UID 2
#define V_UID 3
#define O_UID 4

// FP32 -> FP16 conversion kernel
__global__ void fp32_to_fp16_kernel(const float* in, half* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __float2half(in[idx]);
    }
}

// FP16 -> FP32 conversion kernel
__global__ void fp16_to_fp32_kernel(const half* in, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __half2float(in[idx]);
    }
}

// Cache structure for graph reuse
struct GraphCache {
    std::shared_ptr<fe::graph::Graph> graph;
    cudnnHandle_t handle;
    int64_t workspace_size;
    void* workspace;
    int B, H, N, D;

    GraphCache() : handle(nullptr), workspace_size(0), workspace(nullptr), B(0), H(0), N(0), D(0) {}

    GraphCache(const GraphCache&) = delete;
    GraphCache& operator=(const GraphCache&) = delete;

    GraphCache(GraphCache&& other) noexcept
        : graph(std::move(other.graph)), handle(other.handle),
          workspace_size(other.workspace_size), workspace(other.workspace),
          B(other.B), H(other.H), N(other.N), D(other.D) {
        other.handle = nullptr;
        other.workspace = nullptr;
        other.workspace_size = 0;
    }

    ~GraphCache() {
        if (workspace) cudaFree(workspace);
        if (handle) cudnnDestroy(handle);
    }
};

static std::map<std::tuple<int,int,int,int>, std::unique_ptr<GraphCache>> graph_cache;

void run_cudnn_flash_attention(
    const float* Q, const float* K, const float* V,
    float* O, int B, int H, int N, int D
) {
    auto key = std::make_tuple(B, H, N, D);
    GraphCache* cache = nullptr;

    // Check if graph exists in cache
    auto it = graph_cache.find(key);
    if (it == graph_cache.end()) {
        // Build new graph
        auto new_cache = std::make_unique<GraphCache>();

        if (cudnnCreate(&new_cache->handle) != CUDNN_STATUS_SUCCESS) {
            return;
        }

        auto graph = std::make_shared<fe::graph::Graph>();
        graph->set_io_data_type(fe::DataType_t::HALF)
             .set_intermediate_data_type(fe::DataType_t::FLOAT)
             .set_compute_data_type(fe::DataType_t::FLOAT);

        auto Q_tensor = graph->tensor(fe::graph::Tensor_attributes()
                                          .set_name("Q")
                                          .set_uid(Q_UID)
                                          .set_dim({B, H, N, D})
                                          .set_stride({H * N * D, N * D, D, 1}));

        auto K_tensor = graph->tensor(fe::graph::Tensor_attributes()
                                          .set_name("K")
                                          .set_uid(K_UID)
                                          .set_dim({B, H, N, D})
                                          .set_stride({H * N * D, N * D, D, 1}));

        auto V_tensor = graph->tensor(fe::graph::Tensor_attributes()
                                          .set_name("V")
                                          .set_uid(V_UID)
                                          .set_dim({B, H, N, D})
                                          .set_stride({H * N * D, N * D, D, 1}));

        float attn_scale = 1.0f / sqrtf((float)D);
        auto sdpa_options = fe::graph::SDPA_attributes()
                                .set_name("flash_attention")
                                .set_attn_scale(attn_scale)
                                .set_generate_stats(false);

        auto [O_tensor, Stats] = graph->sdpa(Q_tensor, K_tensor, V_tensor, sdpa_options);

        O_tensor->set_output(true)
                .set_dim({B, H, N, D})
                .set_stride({H * N * D, N * D, D, 1})
                .set_uid(O_UID);

        auto status = graph->build(new_cache->handle, {fe::HeurMode_t::A});
        if (!status.is_good()) {
            cudnnDestroy(new_cache->handle);
            return;
        }

        auto ws_status = graph->get_workspace_size(new_cache->workspace_size);
        (void)ws_status;

        if (new_cache->workspace_size > 0) {
            cudaMalloc(&new_cache->workspace, new_cache->workspace_size);
        }

        new_cache->graph = graph;
        new_cache->B = B;
        new_cache->H = H;
        new_cache->N = N;
        new_cache->D = D;

        cache = new_cache.get();
        graph_cache[key] = std::move(new_cache);
    } else {
        cache = it->second.get();
    }

    size_t fp32_size = (size_t)B * H * N * D * sizeof(float);
    size_t fp16_size = (size_t)B * H * N * D * sizeof(half);

    // Allocate device memory
    float *d_Q_fp32, *d_K_fp32, *d_V_fp32, *d_O_fp32;
    half *d_Q_fp16, *d_K_fp16, *d_V_fp16, *d_O_fp16;

    cudaError_t err;
    err = cudaMalloc(&d_Q_fp32, fp32_size);
    if (err != cudaSuccess) return;
    err = cudaMalloc(&d_K_fp32, fp32_size);
    if (err != cudaSuccess) { cudaFree(d_Q_fp32); return; }
    err = cudaMalloc(&d_V_fp32, fp32_size);
    if (err != cudaSuccess) { cudaFree(d_Q_fp32); cudaFree(d_K_fp32); return; }
    err = cudaMalloc(&d_O_fp32, fp32_size);
    if (err != cudaSuccess) { cudaFree(d_Q_fp32); cudaFree(d_K_fp32); cudaFree(d_V_fp32); return; }
    err = cudaMalloc(&d_Q_fp16, fp16_size);
    if (err != cudaSuccess) { cudaFree(d_Q_fp32); cudaFree(d_K_fp32); cudaFree(d_V_fp32); cudaFree(d_O_fp32); return; }
    err = cudaMalloc(&d_K_fp16, fp16_size);
    if (err != cudaSuccess) { cudaFree(d_Q_fp32); cudaFree(d_K_fp32); cudaFree(d_V_fp32); cudaFree(d_O_fp32); cudaFree(d_Q_fp16); return; }
    err = cudaMalloc(&d_V_fp16, fp16_size);
    if (err != cudaSuccess) { cudaFree(d_Q_fp32); cudaFree(d_K_fp32); cudaFree(d_V_fp32); cudaFree(d_O_fp32); cudaFree(d_Q_fp16); cudaFree(d_K_fp16); return; }
    err = cudaMalloc(&d_O_fp16, fp16_size);
    if (err != cudaSuccess) { cudaFree(d_Q_fp32); cudaFree(d_K_fp32); cudaFree(d_V_fp32); cudaFree(d_O_fp32); cudaFree(d_Q_fp16); cudaFree(d_K_fp16); cudaFree(d_V_fp16); return; }

    // Copy inputs
    cudaMemcpy(d_Q_fp32, Q, fp32_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K_fp32, K, fp32_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V_fp32, V, fp32_size, cudaMemcpyHostToDevice);

    // Convert FP32 -> FP16
    int total_elements = B * H * N * D;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    fp32_to_fp16_kernel<<<blocks, threads>>>(d_Q_fp32, d_Q_fp16, total_elements);
    fp32_to_fp16_kernel<<<blocks, threads>>>(d_K_fp32, d_K_fp16, total_elements);
    fp32_to_fp16_kernel<<<blocks, threads>>>(d_V_fp32, d_V_fp16, total_elements);

    // Execute
    std::unordered_map<fe::graph::Tensor_attributes::uid_t, void*> variant_pack = {
        {Q_UID, d_Q_fp16},
        {K_UID, d_K_fp16},
        {V_UID, d_V_fp16},
        {O_UID, d_O_fp16}
    };

    auto exec_status = cache->graph->execute(cache->handle, variant_pack, cache->workspace);
    (void)exec_status;
    cudaDeviceSynchronize();

    // Convert FP16 -> FP32
    fp16_to_fp32_kernel<<<blocks, threads>>>(d_O_fp16, d_O_fp32, total_elements);
    cudaDeviceSynchronize();

    // Copy result back
    cudaMemcpy(O, d_O_fp32, fp32_size, cudaMemcpyDeviceToHost);

    // Cleanup temporary allocations
    cudaFree(d_Q_fp32); cudaFree(d_K_fp32); cudaFree(d_V_fp32); cudaFree(d_O_fp32);
    cudaFree(d_Q_fp16); cudaFree(d_K_fp16); cudaFree(d_V_fp16); cudaFree(d_O_fp16);
}

#else

void run_cudnn_flash_attention(
    const float* Q, const float* K, const float* V,
    float* O, int B, int H, int N, int D
) {
    // cuDNN not available
}

#endif
