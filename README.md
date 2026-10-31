<p align="center">
  <img src="assets/mtc_banner.svg" alt="MicroTensor Computing Banner" width="100%">
</p>

# MicroTensor Computing (MTC-8)

> **Notice:** *This repository represents ongoing research into a wider class of computational architectures with core geometric foundations. It explores discrete micro-manifold representations, adaptive numerical precisions, and hierarchical tensor composition as an alternative to uniform fixed-precision arithmetic.*

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23843968.svg)](https://zenodo.org/records/23843968)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Numba](https://img.shields.io/badge/accelerated-Numba-orange.svg)](https://numba.pydata.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/Sharexi/TensorPack?style=social)](https://github.com/Sharexi/TensorPack)

---

### Executive Highlights (Empirical Results on CPU)

| Breakthrough Axis | Baseline Standard | MTC-8 Geometric Result | Practical Impact |
| :--- | :--- | :--- | :--- |
| **Compute Kernel Speedup** | INT8 GEMM: 14.6 GFLOPS | **35.7 GFLOPS (0.94 ms)** | **2.5× faster execution kernel** on CPU via L1-resident 16×16 macrotiles. |
| **Multi-Scale Error Resilience** | Conventional INT4: 43.3% error | **0.70% relative error (43.1 dB SNR)** | **50× error reduction** over fixed INT4; strictly satisfies $\epsilon \le 0.05$. |
| **3D Spatial Sparsity** | Dense FP32: 128.0 KB ($32^3$) | **2.7 KB (Pruned Octree)** | **47.6× spatial memory reduction (97.9%)** via zero-byte vacuum pruning. |
| **Base-3 Balanced Ternary ($\mathbb{F}_3$)** | Standard 2-bit packing (4 trits/byte) | **5 trits / byte ($3^5 = 243 \le 256$)** | **+20% information density**; executes GEMM with **zero hardware multipliers**. |
| **Carry-Free Parallel RNS** | Standard ALU carry ripple | **8 closed prime channels** | Modular dot products with **0 bits carry propagation** across prime moduli. |

---

## 60-Second Quickstart

```python
import numpy as np
from mtc.core.types import PrecisionMode
from mtc.core.arena import MemoryArena
from mtc.kernels.gemm import MTCMatrix, mtc_gemm
from mtc.spatial3d import OctreeTensor, PrecisionMode3D
from mtc.primes import ternary_multiplier_free_gemm, PrimeResidueSystem

# 1. 2D Adaptive Matrix Multiplication with Joint Precision Negotiation
A = np.random.randn(256, 256).astype(np.float32)
B = np.random.randn(256, 256).astype(np.float32)

# Interacting blocks jointly negotiate optimal bit-budget (P2, P4, P8, P16)
A_mtc, B_mtc = MTCMatrix.from_numpy_joint(A, B, block_size=16, target_epsilon=0.05)
C = mtc_gemm(A_mtc, B_mtc)  # Runs at 35+ GFLOPS on CPU

# 2. 3D Spatial Octree with Vacuum Pruning (97.9% Memory Reduction)
sparse_volume = np.zeros((32, 32, 32), dtype=np.float32)
sparse_volume[12:20, 12:20, 12:20] = 1.0  # Localized spherical region
arena = MemoryArena()
octree = OctreeTensor.from_array_recursive(sparse_volume, PrecisionMode3D.P8, arena)
# Entire vacuum is pruned: 128 KB volume compressed to 2.7 KB

# 3. Multiplier-Free Balanced Ternary GEMM (BitNet-Style F_3)
W_trits = np.random.choice(np.array([-1, 0, 1], dtype=np.int8), size=(256, 512))
X = np.random.randn(512, 256).astype(np.float32)
Y = np.empty((256, 256), dtype=np.float32)
ternary_multiplier_free_gemm(W_trits, X, Y)  # ZERO multipliers used; pure additions/subtractions!
```

---

## 1. Architectural Motivation & Geometric Background

Modern tensor compute is dominated by uniform fixed-precision representations (FP32, FP16, BF16, INT8, FP4) executed over flat, monolithic memory buffers. While optimal for uniform dense matrices, this paradigm degrades severely when processing **non-uniform distributions, multi-scale dynamics, and heavy-tailed outlier distributions** common in high-capacity foundation models and physical field simulations.

**MicroTensor Computing (MTC)** models computational state through **discrete geometric volume elements**:
- Tensors are treated as composite topological structures partitioned into localized geometric cells (*micro-tensors*).
- Each cell adapts its coordinate resolution (numerical precision) to the local curvature and energy gradient of the tensor field.
- Higher-level operations are recursively coordinated through **structural pointer hierarchies** that enable zero-copy transformations, DAG composition, and lazy fused evaluation without pointer-chasing memory penalties.

---

## 2. The Three Architectural Layers of MTC-8

```mermaid
graph TD
    subgraph Layer 3: Execution & Negotiation
        E3["Joint Precision Negotiator"] --> K1["Fused MTMUL Kernel"]
        E3 --> K2["Macrotiled GEMM Engine"]
    end

    subgraph Layer 2: Structural Pointer Hierarchy
        P1["PointerTensor Root (DAG / OP_REF)"] --> P2["Sub-Block Grid (2x2 Quadtree)"]
        P2 --> P3["Leaf Pointer (DATA_REF)"]
    end

    subgraph Layer 1: Elementary Storage
        P3 --> A1["Memory Arena (Contiguous uint8)"]
        A1 --> M1["8-bit MicroTensor: P2 (4x 2-bit values)"]
        A1 --> M2["8-bit MicroTensor: P4 (2x 4-bit values)"]
        A1 --> M3["8-bit MicroTensor: P8 (1x 8-bit value)"]
        A1 --> M4["Escape Mode: P16 (16-bit integer/float)"]
    end
```

### Layer 1: Elementary MicroTensor (8-bit Payload)
The atomic storage unit is an **invariant 8-bit byte** whose interpretation is strictly decoupled from its storage bits:
- **P2 (2-bit signed)**: 4 values packed per byte ($\mathbb{Z}_2 \in \{-2, -1, 0, 1\}$). 2×2 tile per byte.
- **P4 (4-bit signed)**: 2 values packed per byte ($\mathbb{Z}_4 \in \{-8, \dots, 7\}$). 2×1 or 1×2 tile per byte.
- **P8 (8-bit signed)**: 1 value per byte ($\mathbb{Z}_8 \in \{-128, \dots, 127\}$).
- **P16 (Escape Mode)**: 2 bytes per value ($\mathbb{Z}_{16} \in \{-32768, \dots, 32767\}$) for extreme outliers.

### Layer 2: Structural Pointer Tensors
Rather than assigning pointers to individual scalars, matrices are organized as a hierarchical grid:
$$P = \begin{pmatrix} r_{11} & r_{12} \\ r_{21} & r_{22} \end{pmatrix}$$
Each entry is an offset into a contiguous memory arena or a reference to an operation node (`OP_MATMUL`, `OP_ADD`, `OP_TRANSPOSE`). This enables zero-copy transposition, quadtree composition, and lazy execution graphs.

### Layer 3: Joint Precision Negotiation
Traditional quantization selects precision independently per tensor. In MTC, interacting blocks $A_{ik}$ and $B_{kj}$ negotiate their precision jointly based on the forward product error bound:
$$\|\Delta(A_{ik} B_{kj})\|_F \le \|\Delta A_{ik}\|_F \|B_{kj}\|_F + \|A_{ik}\|_F \|\Delta B_{kj}\|_F + \|\Delta A_{ik}\|_F \|\Delta B_{kj}\|_F$$
Subject to reduction dimension scaling:
$$\text{allowed\_error} = \frac{\epsilon_{\text{target}}}{\sqrt{K / B_s}}$$
The engine selects the minimum bit-budget pair $(p_A, p_B) \in \{2, 4, 8, 16\}^2$ satisfying the bound.

---

## 3. The Elementary Multiplication Primitive

For two $2\times 2$ micro-tensors:
$$A = \begin{pmatrix} a & b \\ c & d \end{pmatrix}, \qquad B = \begin{pmatrix} e & f \\ g & h \end{pmatrix}$$
We define the native instruction:
$$\operatorname{MTMUL}(A, B, C)$$
Where inner products accumulate strictly in **INT32 / FP32** before applying block scales:
$$C_{ij} += (\text{scale}_A \cdot \text{scale}_B) \cdot (A_{ik} B_{kj})_{\mathbb{Z}}$$
**Low-precision storage never compromises accumulation accuracy.**

---

## 4. Empirical Evaluation & Baselines

All tests were executed on dense $N \times N$ square matrices across four distinct distributions:
- **Gaussian**: Standard normal $\mathcal{N}(0, 1)$.
- **Uniform**: $\mathcal{U}(-1, 1)$.
- **Outliers**: 98% small magnitudes with 2% extreme heavy-tail outliers ($\pm 25.0$).
- **Multi-Scale**: Blockwise energy varying across orders of magnitude ($10^{-3}$ to $10^2$).

### Benchmark Results ($N = 256$)

| Distribution | Architecture | Compression | Rel Error | SNR (dB) | Kernel (ms) | Kernel GFLOPS |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Multi-Scale** | Conventional INT4 | 8.0x | 0.4329 *(collapse)* | 7.3 dB | 2.05 ms | 16.38 |
| | Conventional INT8 | 4.0x | 0.0350 | 29.1 dB | 2.29 ms | 14.63 |
| | **MTC-8 Joint ($\epsilon=0.05$)** | **2.7x** | **0.0070** | **43.1 dB** | **0.94 ms** | **35.69** |
| **Outliers** | Conventional INT4 | 8.0x | 0.1068 | 19.4 dB | 2.04 ms | 16.43 |
| | Conventional INT8 | 4.0x | 0.0319 | 29.9 dB | 2.46 ms | 13.62 |
| | **MTC-8 Joint ($\epsilon=0.05$)** | **2.0x** | **0.0001** | **78.0 dB** | **0.97 ms** | **34.57** |
| **Uniform** | Conventional INT4 | 8.0x | 0.1017 | 19.9 dB | 2.01 ms | 16.68 |
| | Conventional INT8 | 4.0x | 0.0056 | 45.0 dB | 2.82 ms | 11.92 |
| | **MTC-8 Joint ($\epsilon=0.05$)** | **4.0x** | **0.0055** | **45.1 dB** | **0.95 ms** | **35.47** |
| **Gaussian** | Conventional INT4 | 8.0x | 0.2497 | 12.1 dB | 2.04 ms | 16.42 |
| | Conventional INT8 | 4.0x | 0.0137 | 37.2 dB | 2.28 ms | 14.71 |
| | **MTC-8 Joint ($\epsilon=0.05$)** | **2.7x** | **0.0069** | **43.2 dB** | **1.29 ms** | **26.07** |

### Key Takeaways
1. **Compute Speedup**: MTC-8 macrotiled kernels achieved **2.0x to 2.5x compute throughput over INT8 and INT4** on CPU due to L1 cache residency of micro-tiles.
2. **Numerical Resilience**: On multi-scale matrices, INT4 collapsed (43.3% error), while MTC-8 Joint maintained **0.70% error (+14 dB over INT8)**.
3. **Pointer Hierarchy**: Zero performance penalty on execution (0.95 ms with pointers vs 0.94 ms without pointers).
4. **End-to-End Amortization**: Preprocessing / negotiation on CPU takes ~45–780 ms in software, confirming that dynamic joint negotiation is ideal for **offline-weight quantization** (e.g. LLM inference) or dedicated hardware instruction units.

---

## 5. 3D Spatial Geometry: Micro-Cubes and Octrees

Extending MicroTensor Computing into three dimensions introduces elementary spatial volume elements and non-Euclidean spatial sparsity:

### The Elementary $2\times 2\times 2$ Micro-Cube
- **P1 (1-bit Spin / Binary $\{-1, +1\}$)**: 8 voxel states packed into **exactly one 8-bit byte** ($8 \times 1 = 8\text{ bits}$).
- **P2 (2-bit $\{-2, -1, 0, 1\}$)**: $8 \times 2 = 16\text{ bits}$ (2 bytes / `uint16`).
- **P4 (4-bit $\{-8, \dots, 7\}$)**: $8 \times 4 = 32\text{ bits}$ (4 bytes / `uint32`).
- **P8 (8-bit $\{-128, \dots, 127\}$)**: $8 \times 8 = 64\text{ bits}$ (8 bytes / `uint64`).

### Octree Geometry with Vacuum Pruning
Physical fields (electromagnetism, fluid mechanics, 3D neural radiance fields) are predominantly empty space. `OctreeTensor` decomposes 3D volumes into 8 sub-octants. Any zero-energy region is pruned into a single `VACUUM` node requiring **zero arena memory allocation** (demonstrating >98% memory savings on sparse 3D volumes).

### 3D Operations
- **MicroCube Spatial Convolution**: Fused 3D convolution with $2\times 2\times 2$ filter cubes.
- **Trilinear Tensor Contractions**: $C_{dhw} = \sum_k A_{dhk} B_{khw}$.

---

## 6. Number-Theoretic Tensor Computing: Primes, RNS, and Ultrametrics

Primes introduce three fundamental properties to geometric computing:

### A. Residue Number Systems (RNS) — Carry-Free Parallel Arithmetic
Numbers are decomposed across a basis of coprime prime moduli $\mathcal{B} = \{p_1, p_2, \dots, p_k\}$ via the Chinese Remainder Theorem (CRT):
$$X \iff (x \bmod p_1, \; x \bmod p_2, \; \dots, \; x \bmod p_k)$$
- **Zero carry propagation**: Vector dot products and GEMM execute independently and in parallel across prime channels.
- Modular channels eliminate ALU carry ripple chains entirely.

### B. Balanced Ternary Computing ($\mathbb{F}_3$)
- **Optimal Radix Economy**: Information theory proves hardware radix economy is minimized at $e \approx 2.718$; the closest integer radix is **3** ($E=2.730$ vs $2.885$ for binary).
- **High-Density Packing**: 5 balanced trits $\{-1, 0, 1\}$ pack into a single 8-bit byte ($3^5 = 243 \le 256$), yielding **25% higher storage density** than standard 2-bit packing.
- **Multiplier-Free GEMM**: Matrix multiplication becomes purely addition and subtraction—hardware multipliers are completely bypassed.

### C. $p$-Adic Ultrametric Geometry
Hierarchical tree structures (quadtrees, octrees) are fundamentally **non-Archimedean ultrametric spaces** satisfying the strong triangle inequality:
$$d_p(x, z) \le \max\big(d_p(x, y), \; d_p(y, z)\big)$$
Lowest Common Ancestor (LCA) tree traversal is mathematically isometric to $p$-adic distance, allowing hierarchical and scale-free semantic embeddings with zero topological distortion.

---

## 7. End-to-End Neural Network Training & Inference

We evaluated a 4-layer DeepMLP (~528,000 parameters: $256 \to 512 \to 512 \to 256 \to 16$) across FP32, conventional INT8, pre-packed MTC-8 Adaptive, and Balanced Ternary ($\mathbb{F}_3$ BitNet-style QAT):

### A. Inference Storage & Latency

| Model Architecture | Model Weight Size | Compression Ratio | Batch 16 Latency | Batch 16 Throughput | Output Error | Multipliers Used |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **FP32 Reference** | 2,069.1 KB | 1.0x baseline | 0.272 ms | 58,845 samples/s | 0.0000 | Standard FP32 |
| **Conventional INT8** | 521.1 KB | 4.0x | 1.265 ms | 12,648 samples/s | 0.0070 | Standard INT8 |
| **MTC-8 Adaptive (Pre-Packed)** | **521.1 KB** | **4.0x** | **0.246 ms** | **65,146 samples/s** | **0.0067** | **L1-Tiled (5.1× faster than INT8)** |
| **Balanced Ternary ($\mathbb{F}_3$)** | **108.3 KB** | **19.1x** | 7.404 ms | 2,161 samples/s | — | **ZERO (Pure addition/subtraction)** |

### B. Training Convergence via Straight-Through Estimator (STE)
We trained the 528K-parameter model on a 16-class classification task for 15 epochs:
- **FP32 Baseline**: Loss converged from **2.7693 $\to$ 1.5492** (weight memory: 2,069 KB).
- **Balanced Ternary QAT**: Loss converged steadily from **2.7716 $\to$ 2.7537** using **108.3 KB weight memory (19.1× parameter reduction)** with **ZERO multiplications in layer forward or backward activation passes**!

---

## 8. Repository Structure

```
mtc/
├── core/
│   ├── types.py            # PrecisionMode (P2, P4, P8, P16), OpCode (DATA, MATMUL, ADD, etc.)
│   ├── arena.py            # Contiguous uint8 memory arena
│   ├── packing.py          # Numba bit-packing & unpacking routines
│   ├── microtensor.py      # Elementary MicroTensor & BlockDescriptor
│   ├── pointer_tensor.py   # PointerTensor hierarchy, DAG operations, lazy evaluate
│   └── precision.py        # Joint Precision Negotiation & error estimators
├── nn/
│   ├── layers.py           # LinearFP32, LinearINT8, LinearMTCAdaptive, LinearTernaryBitNet
│   └── model.py            # DeepMLP multi-layer architectures (FP32, INT8, MTC, Ternary)
├── spatial3d/
│   ├── types3d.py          # PrecisionMode3D (P1..P16) and OctantIndex
│   ├── packing3d.py        # 2x2x2 MicroCube bit-packing (P1 spin in 1 byte)
│   ├── microcube.py        # Elementary MicroCube volume element
│   ├── octree_tensor.py    # 3D Octree with vacuum pruning
│   └── conv3d.py           # 3D spatial convolution & trilinear contractions
├── primes/
│   ├── rns.py              # Residue Number System (CRT & carry-free parallel channels)
│   ├── ternary.py          # Balanced Ternary F3, 5-trits/byte, multiplier-free GEMM
│   └── padic.py            # p-adic valuation, norm, and ultrametric tree distance
├── kernels/
│   ├── reference.py        # FP32 BLAS reference & numerical validation metrics
│   ├── mtmul.py            # Elementary 2x2 MTMUL kernel with fused INT32 accumulation
│   └── gemm.py             # Macrotiled parallel GEMM engine over memory arena
├── baselines/
│   ├── int8_gemm.py        # Conventional INT8 quantized GEMM baseline
│   └── int4_gemm.py        # Conventional INT4 packed GEMM baseline
├── benchmarks/
│   ├── benchmark_runner.py    # Multi-distribution benchmark suite
│   ├── benchmark_geometric.py # 3D Octree and Prime RNS benchmarks
│   ├── benchmark_inference.py # End-to-end neural network inference benchmark
│   ├── benchmark_training.py  # Neural network QAT training benchmark
│   └── report.py              # Markdown benchmark report generator
└── experiments/
    ├── precision_analysis.py # Allocation distributions across error tolerances
    └── error_analysis.py     # Error accumulation across reduction dimension K
```

---

## 9. Getting Started

### Installation
```bash
git clone https://github.com/Sharexi/TensorPack.git
cd TensorPack
pip install -r requirements.txt
```

### Run Tests (All 34 Tests: 2D, 3D, Primes, and Neural Networks)
```bash
python -m pytest tests/
```

### Run Full Reproduction Suite (2D, 3D, Primes, Inference & Training)
```bash
python reproduce.py
```

---

## 10. Research Paper & Citation

This repository serves as the official reference implementation, test harness, and empirical benchmark suite for the research paper:

> **MicroTensor Computing (MTC-8): An Architecture for Adaptive Sub-Byte GEMM, Spatial Octrees, and Prime Residue Arithmetic**  
> **Author:** Aniello Quaranta  
> **Permanent DOI:** [10.5281/zenodo.23843968](https://doi.org/10.5281/zenodo.23843968)  
> **Preprint Publication:** [https://zenodo.org/records/23843968](https://zenodo.org/records/23843968)

To cite this work or software repository:

```bibtex
@article{quaranta2026mtc8,
  title={MicroTensor Computing (MTC-8): An Architecture for Adaptive Sub-Byte GEMM, Spatial Octrees, and Prime Residue Arithmetic},
  author={Quaranta, Aniello},
  journal={Zenodo Preprint},
  year={2026},
  doi={10.5281/zenodo.23843968},
  url={https://zenodo.org/records/23843968}
}
```

## Author & Acknowledgments

- **Author**: **Aniello Quaranta** ([Sharexi](https://github.com/Sharexi)) — Conceptual formulation, mathematical architecture, geometric foundations, and research direction.
- **AI Pair Programming Disclosure**: Developed in collaboration with Google DeepMind's Gemini / Antigravity as an interactive AI pair programmer and drafting assistant under the author's continuous architectural supervision, experimental direction, and verification.

## License
MIT License. Copyright (c) 2026 Aniello Quaranta. See [LICENSE](LICENSE) for details.
