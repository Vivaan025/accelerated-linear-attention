# CUDA Delta Attention

Linear attention kernels optimized for long context. Runs on RTX 5090.

## How to Run

```bash
# run the benchmark
make throughput
```

Or if you need to rebuild:

```bash
make clean && make throughput
```

## What's in here

- `delta_attention_baseline.cu` - basic sequential version, just for comparison
- `delta_attention_optimized.cu` - uses shared memory, float4 loads, parallel dimension scans
- `delta_attention_fla.cu` - inspired by Flash Linear Attention paper, chunk-wise grid
- `delta_attention_5090.cu` - tuned for 5090's 128KB shared memory, uses 96KB
- `kda_optimized_5090.cu` - full Kimi Delta Attention with matrix state, fp16 storage

## Results

Tested on RTX 5090, batch size 1:

| Seq Length | Linear (best) | cuDNN Flash | Speedup |
| ---------- | ------------- | ----------- | ------- |
| 2K         | 0.905 ms      | 1.292 ms    | 1.43x   |
| 8K         | 3.531 ms      | 4.654 ms    | 1.32x   |
| 32K        | 15.656 ms     | 25.981 ms   | 1.66x   |
| 128K       | 62.487 ms     | 215.825 ms  | 3.45x   |
| 256K       | 125.599 ms    | 722.891 ms  | 5.76x   |

Linear attention scales O(N) vs Flash Attention's O(N²). The difference shows up more at longer sequences.

KDA (the "proper" version with matrix state) is surprisingly fast - often matches the simplified delta attention even though it does more work per token.

Full logs in [throughput_results.md](./throughput_results.md).

## How it works

Chunk-wise processing:

- parallel within each chunk (like transformers)
- sequential between chunks (like RNNs)

The key is keeping the recurrent state in shared memory instead of going back to VRAM every time. The 5090 kernel uses 96KB of the available 128KB SRAM per SM.

## Build flags

```
-O3 -arch=sm_90a -use_fast_math --ptxas-options=-v
```
