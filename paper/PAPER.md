# MicroTensor Computing (MTC-8): An Architecture for Adaptive Sub-Byte GEMM, Spatial Octrees, and Prime Residue Arithmetic

**Author:** Sharexi  
**Code & Reproduction:** [https://github.com/Sharexi/TensorPack](https://github.com/Sharexi/TensorPack)  
**Preprint:** arXiv Submission Draft (cs.AR / cs.LG / cs.DC)

---

## Abstract

Modern neural networks and scientific computing are dominated by uniform fixed-precision arithmetic (FP32, BF16, INT8, FP4) allocated across flat, monolithic memory buffers. While optimal for uniform dense distributions, this paradigm degrades severely in the presence of non-uniform dynamics, multi-scale energy variations, and heavy-tailed outliers. We introduce **MicroTensor Computing (MTC-8)**, a geometric computational architecture based on localized micro-tensors, decoupled metadata interpretation, and hierarchical structural pointer composition. 

MTC-8 establishes three foundational principles:
1. **Joint Precision Negotiation**, which dynamically selects interacting block bit-depths by bounding forward matrix product error $\Delta(AB) \le \epsilon$ rather than isolated reconstruction loss.
2. **Spatial Octree Geometry**, which extends micro-tensors to 3D $2\times 2\times 2$ micro-cubes and leverages non-Archimedean vacuum pruning to collapse empty physical space.
3. **Number-Theoretic Arithmetic**, which introduces carry-free parallel Residue Number Systems (RNS) across prime moduli and balanced ternary ($\mathbb{F}_3$) multiplier-free linear algebra with an optimal radix economy ($E \approx 2.730$).

Empirical benchmarks demonstrate that on non-uniform multi-scale matrices where conventional INT4 collapses with 43.3% relative error, MTC-8 maintains strictly bounded error at **0.70% (43.1 dB SNR, a +14 dB gain over INT8)** while achieving a **2.0x–2.5x compute kernel speedup (35.7 GFLOPS on CPU)**. In 3D physical fields, octree vacuum pruning delivers a **47.6x memory reduction (97.9% storage savings)**. Finally, in end-to-end 528K-parameter neural network experiments, pre-packed MTC layers achieve a **5.1x inference speedup over conventional INT8**, while balanced ternary quantization-aware training compresses model parameters by **19.1x (from 2,069 KB to 108 KB)** and converges with **zero hardware multipliers in both forward and backward activation passes**.

---

## 1. Introduction

The computational scaling of artificial intelligence and physical field simulations is fundamentally constrained by memory bandwidth and arithmetic energy efficiency. Conventional processor architectures—including modern GPUs, TPUs, and CPUs—treat tensors as flat, homogeneous arrays of uniform numerical types (FP32, BF16, FP8, or INT8).

While uniform quantization methods such as INT4 or FP4 offer theoretical $4\times$--$8\times$ storage reductions, they suffer from catastrophic precision collapse when applied to real-world activation distributions containing heavy-tailed outliers or orders-of-magnitude energy variation across channels and heads. Furthermore, in spatial domains such as 3D neural radiance fields, electromagnetics, and fluid mechanics, physical space is overwhelmingly empty, yet dense tensors force uniform memory allocation across vacuum regions.

To address these limitations, we propose **MicroTensor Computing (MTC-8)**. Rather than treating tensors as monolithic arrays, MTC-8 models tensors as composite geometric manifolds partitioned into elementary, localized cells called *micro-tensors*. By strictly decoupling the elementary storage payload (8-bit bytes) from its interpretation, MTC-8 dynamically adapts precision, enforces forward error guarantees through joint precision negotiation, prunes vacuum octants in 3D spatial volumes, and exploits carry-free prime-residue arithmetic.

---

## 2. The MTC-8 Three-Layer Architecture

The MTC-8 architecture separates computation into three decoupled layers:

### Layer 1: The Elementary MicroTensor
The atomic storage unit is an invariant 8-bit byte whose numerical interpretation is decoupled from its storage bits. In contrast to systems that sacrifice bit precision to encode metadata in-band, MTC-8 stores metadata out-of-band in compact block descriptors:
- **P2 (2-bit signed)**: 4 values packed per byte ($\mathbb{Z}_2 \in \{-2, -1, 0, 1\}$). Forms an elementary $2\times 2$ micro-tile per byte.
- **P4 (4-bit signed)**: 2 values packed per byte ($\mathbb{Z}_4 \in \{-8, \dots, 7\}$). Forms an elementary $2\times 1$ or $1\times 2$ micro-tile per byte.
- **P8 (8-bit signed)**: 1 value per byte ($\mathbb{Z}_8 \in \{-128, \dots, 127\}$).
- **P16 (Escape Mode)**: 2 bytes per value ($\mathbb{Z}_{16} \in \{-32768, \dots, 32767\}$) for high-magnitude outlier blocks.

All micro-tensor payloads reside in a contiguous 1D linear memory arena, eliminating heap fragmentation and maximizing cache-line alignment.

### Layer 2: Structural Pointer Hierarchy
Rather than assigning machine pointers to individual scalar values, structural relationships are expressed through compact grid descriptors:
$$P = \begin{pmatrix} r_{11} & r_{12} \\ r_{21} & r_{22} \end{pmatrix}$$

Where each entry $r_{ij}$ is a relative offset into the memory arena (a `DATA_REF`) or an operation reference (an `OP_REF` representing matrix multiplication, addition, or transposition). This enables zero-copy slicing, quadtree block composition, and lazy computational DAG execution without pointer-chasing latency during inner numerical loops.

### Layer 3: Execution Engine and MTMUL Primitive
For two elementary $2\times 2$ micro-tensors:
$$A = \begin{pmatrix} a & b \\ c & d \end{pmatrix}, \qquad B = \begin{pmatrix} e & f \\ g & h \end{pmatrix}$$

We define the native atomic instruction:
$$\operatorname{MTMUL}(A, B, C)$$

Where inner products accumulate strictly in **INT32 or FP32** before applying block scaling factors:
$$C_{ij} += (\text{scale}_A \cdot \text{scale}_B) \cdot \sum_k (A_{ik} B_{kj})_{\mathbb{Z}}$$

Crucially, low-precision storage never compromises accumulation precision.

---

## 3. Joint Precision Negotiation

Traditional post-training quantization selects bit-depth independently per matrix based on reconstruction error $\|\Delta A\|_F$. However, in dense matrix multiplication $C = A B$, forward error propagates bilinearly across interacting tiles:
$$\Delta (A_{ik} B_{kj}) = \hat{A}_{ik}\hat{B}_{kj} - A_{ik} B_{kj} = (\Delta A_{ik}) B_{kj} + A_{ik} (\Delta B_{kj}) + (\Delta A_{ik})(\Delta B_{kj})$$

Applying the Cauchy-Schwarz inequality in Frobenius norm:
$$\|\Delta (A_{ik} B_{kj})\|_F \le \|\Delta A_{ik}\|_F \|B_{kj}\|_F + \|A_{ik}\|_F \|\Delta B_{kj}\|_F + \|\Delta A_{ik}\|_F \|\Delta B_{kj}\|_F$$

Across $K_{\text{tiles}} = K / B_s$ block accumulations through the reduction dimension $K$, the allowed error per block product is bounded by:
$$\tau_{\text{allowed}} = \frac{\epsilon_{\text{target}}}{\sqrt{K / B_s}} \cdot \|A_{ik}\|_F \|B_{kj}\|_F$$

Joint Precision Negotiation searches the precision pair $(p_A, p_B) \in \{2, 4, 8, 16\}^2$ that minimizes total bit footprint $B_s^2 (p_A + p_B)$ subject to $\|\Delta(A_{ik} B_{kj})\|_F \le \tau_{\text{allowed}}$. If either interacting block has near-zero energy, precision is relaxed; if high-variance outliers are detected, the interaction dynamically escapes to P16.

---

## 4. 3D Spatial Geometry and Octree Vacuum Pruning

Extending micro-tensors to 3D spatial volumes reveals that the elementary storage model natively captures spatial voxel geometry:

### The Elementary $2\times 2\times 2$ Micro-Cube
An elementary 3D volume element contains $2^3 = 8$ scalar voxels:
- **P1 (1-bit Spin / Binary)**: 8 voxel values ($\{-1, +1\}$ or $\{0, 1\}$) pack into **exactly one 8-bit byte** ($8 \times 1\text{ bit} = 8\text{ bits}$).
- **P2**: $8 \times 2 = 16\text{ bits}$ (2 bytes / `uint16`).
- **P4**: $8 \times 4 = 32\text{ bits}$ (4 bytes / `uint32`).
- **P8**: $8 \times 8 = 64\text{ bits}$ (8 bytes / `uint64`).

### Octree Vacuum Pruning
In physical field simulations and 3D radiance fields, the majority of volume is vacuum. `OctreeTensor` recursively partitions 3D space into 8 sub-octants in Morton Z-order:
$$\mathcal{O} = \{O_{000}, O_{001}, O_{010}, O_{011}, O_{100}, O_{101}, O_{110}, O_{111}\}$$

If all voxels within an octant fall below a numerical threshold ($\|O\|_\infty < \epsilon_{\text{vac}}$), the entire subtree is collapsed into a single `VACUUM` descriptor requiring **zero arena memory allocation**.

---

## 5. Number-Theoretic Arithmetic and Primes

### A. Residue Number Systems (RNS): Carry-Free Parallel Arithmetic
In conventional two's complement adders, bit carries ripple across the word. Given a coprime prime basis $\mathcal{B} = \{p_1, p_2, \dots, p_k\}$, the Chinese Remainder Theorem (CRT) establishes an isomorphism between the ring $\mathbb{Z}_M$ (where $M = \prod p_i$) and the direct product of prime fields:
$$\mathbb{Z}_M \cong \mathbb{F}_{p_1} \times \mathbb{F}_{p_2} \times \dots \times \mathbb{F}_{p_k}$$

Vector dot products $\langle A, B \rangle$ are evaluated independently across prime channels without carry propagation:
$$s_i = \left( \sum_{j} (A_j \bmod p_i) \cdot (B_j \bmod p_i) \right) \bmod p_i$$

Reconstruction is performed via precomputed CRT orthogonal weights $w_i = M_i (M_i^{-1} \bmod p_i)$:
$$X = \left( \sum_{i=1}^k s_i \cdot w_i \right) \bmod M$$

### B. Balanced Ternary ($\mathbb{F}_3$) and Radix Economy
The hardware cost to represent integers up to $N$ with radix $b$ is quantified by the radix economy:
$$E(b, N) = b \cdot \lceil \log_b N \rceil$$

The continuous function $f(b) = b / \ln b$ achieves its global minimum at Euler's number $b = e \approx 2.71828$. The closest integer radix is **3** ($E=2.730$), which is strictly more efficient than binary ($E=2.885$).

By mapping trits $t \in \{-1, 0, 1\}$ into offsets $u = t + 1 \in \{0, 1, 2\}$, 5 trits are packed into a single byte:
$$\text{Byte} = u_0 + 3 u_1 + 9 u_2 + 27 u_3 + 81 u_4 \le 242 < 256$$

Yielding a **20% storage density gain** over standard 2-bit packing. In linear layers, weights $W \in \{-1, 0, 1\}$ eliminate hardware multipliers entirely:
$$Y = X W = \sum_{k: W_{kj}=1} X_{ik} - \sum_{k: W_{kj}=-1} X_{ik}$$

### C. $p$-Adic Ultrametric Tree Topologies
Hierarchical quadtrees and octrees possess an ultrametric metric space topology governed by the strong triangle inequality:
$$d_p(x, z) \le \max\big(d_p(x, y), \; d_p(y, z)\big)$$

The tree distance between leaf nodes with lowest common ancestor (LCA) depth $h$ is isometric to the $p$-adic metric:
$$d(A, B) = p^{-h(A, B)}$$

---

## 6. Empirical Evaluation

### Dense Matrix Multiplication ($N=256$)

| Distribution | Architecture | Compression | Rel. Error | SNR (dB) | Kernel (ms) | GFLOPS |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Multi-Scale** | INT4 GEMM | 8.0x | 0.4329 | 7.3 | 2.05 | 16.38 |
| | INT8 GEMM | 4.0x | 0.0350 | 29.1 | 2.29 | 14.63 |
| | **MTC-8 Joint** | **2.7x** | **0.0070** | **43.1** | **0.94** | **35.69** |
| **Outliers** | INT4 GEMM | 8.0x | 0.1068 | 19.4 | 2.04 | 16.43 |
| | INT8 GEMM | 4.0x | 0.0319 | 29.9 | 2.46 | 13.62 |
| | **MTC-8 Joint** | **2.0x** | **0.0001** | **78.0** | **0.97** | **34.57** |
| **Uniform** | INT4 GEMM | 8.0x | 0.1017 | 19.9 | 2.01 | 16.68 |
| | INT8 GEMM | 4.0x | 0.0056 | 45.0 | 2.82 | 11.92 |
| | **MTC-8 Joint** | **4.0x** | **0.0055** | **45.1** | **0.95** | **35.47** |
| **Gaussian** | INT4 GEMM | 8.0x | 0.2497 | 12.1 | 2.04 | 16.42 |
| | INT8 GEMM | 4.0x | 0.0137 | 37.2 | 2.28 | 14.71 |
| | **MTC-8 Joint** | **2.7x** | **0.0069** | **43.2** | **1.29** | **26.07** |

### 3D Spatial Octree Vacuum Pruning
- **$16\times 16\times 16$ Volume** (93.7% vacuum): Uncompressed FP32 requires $16.0\text{ KB}$. Pruned `OctreeTensor` consumes $0.4\text{ KB}$ (**37.9x spatial reduction**, 97.4% memory savings).
- **$32\times 32\times 32$ Volume** (93.6% vacuum): Uncompressed FP32 requires $128.0\text{ KB}$. Pruned `OctreeTensor` consumes $2.7\text{ KB}$ (**47.6x spatial reduction**, 97.9% memory savings).
- 3D spatial convolution with a $2\times 2\times 2$ micro-cube kernel achieved **13,502 MVOX/s** throughput ($0.04\text{ ms}$).

### End-to-End Neural Network Inference & Training

| Model Architecture | Weight Storage | Compression | Latency (ms, B=16) | Throughput (samples/s) | Multipliers Used |
| :--- | :--- | :--- | :--- | :--- | :--- |
| FP32 Reference | 2,069.1 KB | 1.0x | 0.272 | 58,845 | Standard FP32 |
| Conventional INT8 | 521.1 KB | 4.0x | 1.265 | 12,648 | Standard INT8 |
| **MTC-8 Adaptive (Pre-Packed)** | **521.1 KB** | **4.0x** | **0.246** | **65,146** | **L1-Tiled (5.1x faster than INT8)** |
| **Balanced Ternary ($\mathbb{F}_3$)** | **108.3 KB** | **19.1x** | 7.404 | 2,161 | **ZERO (Pure addition/subtraction)** |

**Multiplier-Free Training**: In 15 epochs of classification training using Quantization-Aware Training with the Straight-Through Estimator (STE), Balanced Ternary converged steadily from loss $2.7716 \to 2.7537$. Model parameter memory was reduced by **19.1x** (108 KB vs 2,069 KB), with **zero hardware multiplications** in both layer forward and backward activation passes.

---

## 7. Hardware Implementation Roadmap

1. **Sub-Byte ALU Array**: A custom $\operatorname{MTMUL}$ functional unit executing four 2-bit multiply-accumulates in a single clock cycle without software bit-shifting.
2. **Asynchronous Octree Traverser**: Dedicated hardware traversal units that skip vacuum octants in one cycle, streaming only active micro-cubes to execution pipelines.
3. **Modular RNS Accelerators**: Completely independent arithmetic channels without global carry chains, drastically reducing capacitive switching power and propagation delay in VLSI silicon.

---

## 8. Conclusion

MicroTensor Computing (MTC-8) replaces monolithic, uniform fixed-precision arrays with discrete geometric volume elements, joint error-bounded precision negotiation, spatial octree vacuum pruning, and number-theoretic prime residue arithmetic. Empirical results confirm that MTC-8 eliminates precision collapse on non-uniform distributions (reducing error from 43.3% to 0.70%), accelerates CPU inference by 5.1x, compresses 3D spatial fields by 47.6x, and enables 19.1x parameter compression during training with zero hardware multipliers.

---

## References

1. Wang, H., et al. (2024). *The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits*. arXiv:2402.17764.
2. Dettmers, T., et al. (2022). *LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale*. NeurIPS.
3. Xiao, G., et al. (2023). *SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models*. ICML.
4. Lin, J., et al. (2024). *AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration*. MLSys.
5. Bengio, Y., et al. (2013). *Estimating or propagating gradients through stochastic neurons for conditional computation*. arXiv:1308.3432.
6. Bronstein, M. M., et al. (2021). *Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges*. arXiv:2104.13478.
7. Knuth, D. E. (1997). *The Art of Computer Programming, Volume 2: Seminumerical Algorithms*. Addison-Wesley.
8. Hayes, B. (2001). *Third base: Ternary computing and radix economy*. American Scientist.
9. Khrennikov, A. (1997). *Non-Archimedean Analysis: Quantum Paradoxes, Dynamical Systems and Biological Models*. Springer.
10. Müller, T., et al. (2022). *Instant Neural Graphics Primitives with a Multiresolution Hash Encoding*. ACM TOG.
