# WikiText-2 perplexity, Qwen/Qwen2.5-0.5B (weight-only quantization)

Evaluated on 24,576 test tokens (seqlen 1024), identical for all methods. Calibration: 32 x 512 train tokens. Bits = total stored bits per quantized weight (payload + scales + side data). Embeddings / tied LM head kept in FP32 for all methods.

| Method | Bits / weight | Perplexity |
|---|---|---|
| FP32 (unquantized) | 32.000 | 13.986 |
| MTC-QT quadtree 32-4 @ 2.5 b | 2.500 | 266.504 |
| MTC-RD @ 2.5 b | 2.500 | 543.485 |
| RTN g128 INT3 | 3.125 | 202.627 |
| AWQ g128 INT3 | 3.125 | 113.600 |
| MTC-RD @ 3.125 b | 3.125 | 84.863 |
| MTC-RD, no importance (ablation) @ 3.125 b | 3.125 | 902.755 |
| MTC-RD 16x16, bf16 scales (ablation) @ 3.125 b | 3.125 | 68.663 |
| MTC-QT quadtree 32-4 @ 3.125 b | 3.125 | 29.757 |
| RTN g128 INT4 | 4.125 | 19.314 |
| AWQ g128 INT4 | 4.125 | 18.906 |
| MTC-RD @ 4.125 b | 4.125 | 16.556 |
| MTC-RD, no importance (ablation) @ 4.125 b | 4.125 | 22.537 |
| MTC-RD 16x16, bf16 scales (ablation) @ 4.125 b | 4.125 | 16.277 |
| MTC-QT quadtree 32-4 @ 4.125 b | 4.125 | 15.702 |
| MXFP4 | 4.250 | 17.422 |
| MTC-RD @ 4.25 b | 4.250 | 15.942 |
| MTC-QT quadtree 32-4 @ 4.25 b | 4.250 | 15.331 |
| RTN g32 INT4 (Q4_0-style) | 4.500 | 16.546 |
| NVFP4 | 4.500 | 15.462 |
| MTC-RD @ 4.5 b | 4.500 | 15.258 |
| MTC-QT quadtree 32-4 @ 4.5 b | 4.500 | 14.907 |
