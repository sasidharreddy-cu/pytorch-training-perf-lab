# pytorch-training-perf-lab

Measured experiments on making PyTorch training and inference faster on NVIDIA GPUs. Every number below comes from code in this repo, run on a free Google Colab GPU.

**Hardware:** NVIDIA Tesla T4 (16 GB) · PyTorch 2.11.0 (CUDA 13.0) · Google Colab

| # | Experiment | Baseline | Best | Speedup | Key finding |
|---|---|---|---|---|---|
| 00 | [GPU performance from first principles](#00--gpu-performance-from-first-principles) | `x.cos().cos()` eager: 4.57 ms | compiled: 2.12 ms | 2.15× | Elementwise ops use <1% of the GPU's math; fusion removes memory trips |
| 01 | Profile a training step | | | | *in progress* |
| 02 | DataLoader tuning | img/s | img/s | | |
| 03 | AMP + torch.compile | ms/step | ms/step | | |
| 04 | DDP, 1 vs 2 GPUs | img/s | img/s | % eff. | |
| 05 | ONNX / TensorRT inference | ms | ms | | |
| 06 | Fused bias + GELU Triton kernel | µs | µs | | |

---

## 00 · GPU performance from first principles

This experiment reproduces the ideas in Horace He's [*Making Deep Learning Go Brrrr From First Principles*](https://horace.io/brrr_intro.html) with my own measurements. A GPU workload is limited by one of three things: **compute** (math), **memory bandwidth** (moving data), or **overhead** (Python, PyTorch and kernel launches). Each needs a different fix.

[Notebook](00_brrr_first_principles/brrr_first_principles.ipynb) · [Raw results (JSON)](00_brrr_first_principles/brrr_results.json)

![Measured roofline on a Tesla T4](00_brrr_first_principles/roofline.png)

### Results (Tesla T4)

| Question | Measurement |
|---|---|
| Memory bandwidth reached | **248 GB/s** (78% of the 320 GB/s datasheet) |
| When does an elementwise kernel turn compute-bound? | Between **64 and 128** FLOPs per element. Theory predicts **~106** (3.3 TFLOPS ÷ 31 G elements/s) |
| Math capacity used by a 1-FLOP op (ReLU, add) | **0.9%** |
| Kernel fusion, `x.cos().cos()` | **4.57 ms** eager (2 kernels) → **2.12 ms** compiled (1 kernel) |
| Does activation complexity matter? | No. GELU 2.28 ms vs ReLU 2.31 ms; both are memory-bound |
| Fixed overhead per tiny op | **~15 µs**, flat from 1 to 262,144 elements |
| Batch-size test (3-layer MLP) | Batch 1 → 1,024: 146 → 180 µs (1,024× the work, +23% time) |
| CUDA Graphs (20-layer MLP, batch 16, 40 kernels) | Eager **1,103 µs** → torch.compile 1,029 µs → CUDA Graph **161 µs** (**6.9×**) |
| Matmul, fp32 vs fp16 Tensor Cores | **3.8** vs **20.5** TFLOPS (**5.4×**) |
| BERT-base MLP block (4,096 tokens, fp16) | Eager **2.96 ms** (8 kernels) → compiled **2.58 ms** (5 kernels), 1.15× |

### What I learned

- **Most non-matmul ops are memory-bound.** ReLU, GELU, sigmoid, exp and cos all ran at the same speed (~2.3 ms on 67M elements), at ~232 GB/s. The math is effectively free, and the cost is reading and writing memory.
- **Fusion is the fix for memory-bound code.** Fusing two `cos` calls into one kernel halved the memory traffic and the runtime. In the BERT MLP block, the two matmuls take an estimated ~1.9 ms at measured fp16 peak. The remaining ~1.1 ms (about a third of the runtime) is elementwise and LayerNorm work that is under 1% of the FLOPs. Fusion cut that remainder to ~0.7 ms.
- **Tiny ops are overhead-bound.** Colab's CPU managed ~5.7M Python additions per second, while the GPU did 3.3 trillion multiplies per second. Below ~400K elements every op cost the same ~15 µs. CUDA Graphs removed most of it by replaying 40 kernel launches as one.
- **Big matmuls are compute-bound, and precision is the lever.** fp16 on Tensor Cores ran 5.4× faster than fp32. That is the case for mixed precision (AMP) in compute-bound training.

### What didn't work

- **My first CUDA Graphs test was too small to show anything.** A 3-layer MLP launches only 5 kernels. torch.compile ran **1.6× slower** than eager (238 vs 145 µs), and `mode="reduce-overhead"` gave no gain. On Colab's slow CPU, the compiler's per-call guard checks cost about as much as the launches they save. A 20-layer model (40 kernels) with manual `torch.cuda.CUDAGraph` capture showed the real effect: 6.9×.
- **Run-to-run variance on matmuls is ~15%.** Two runs on different Colab T4 sessions gave fp16 peaks of 23.7 and 20.5 TFLOPS. The T4 is power-capped at 70 W and its clock varies. Bandwidth-bound numbers varied by under 2%. The table above is from a single complete run.

### Method

- Timing: 3 warm-up calls, `torch.cuda.synchronize()` before stopping every timer, median of 5 trials. Without `synchronize()`, a 1.8 ms op measured as 0.37 ms.
- Kernel counts and names come from `torch.profiler`. Example: the fused kernel `triton_poi_fused_cos_0`.
- FLOPs and bytes are counted analytically: an n×n matmul is 2n³ FLOPs and 3n² elements; an elementwise op reads and writes each element once.

### How to reproduce

Click **Open in Colab** above, then **Runtime → Change runtime type → T4 GPU**, then **Runtime → Run all** (about 10 minutes).
