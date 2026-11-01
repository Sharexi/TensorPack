# Accuracy-per-bit results (relative output error; lower is better)

Bits are total stored bits per element including scales and side data. Curves are compared at matched budgets by linear (realizable) interpolation between measured points; single-point formats are listed at their native bit width.

## gaussian (A@B, both quantized)

| Method | @2.5 b | @3.0 b | @4.25 b | @5.0 b | @6.0 b |
|---|---|---|---|---|---|
| Block 16x16 INT (RTN) | 0.8480 | 0.5090 | 0.1668 | 0.0949 | 0.0455 |
| Group INT g32 (Q4_0-style) | 0.9759 | 0.6505 | 0.1844 | 0.1007 | 0.0474 |
| Group INT g128 (GPTQ-RTN-style) | 0.8233 | 0.4805 | 0.1547 | 0.0883 | 0.0424 |
| Group INT g128 + fp16 outliers (best of 0.5-4%) | 1.0298 | 0.9699 | 0.2106 | 0.1295 | 0.0603 |
| MTC scales, fixed width (ablation) | 0.4246 | 0.3055 | 0.1361 | 0.0852 | 0.0439 |
| MTC-RD, no importance (ablation) | 0.4324 | 0.3102 | 0.1354 | 0.0852 | 0.0437 |
| MTC-RD fixed 16x16 (v1) | 0.4322 | 0.3105 | 0.1355 | 0.0852 | 0.0437 |
| MTC-QT quadtree 32-4 (ours) | 0.4150 | 0.2828 | 0.1298 | 0.0793 | 0.0406 |

Fixed formats: MXFP4 0.1620 @ 4.25 b, NVFP4 0.1341 @ 4.50 b, MXINT8 0.0117 @ 8.25 b, INT4 0.2819 @ 4.00 b, INT8 0.0155 @ 8.00 b

## uniform (A@B, both quantized)

| Method | @2.5 b | @3.0 b | @4.25 b | @5.0 b | @6.0 b |
|---|---|---|---|---|---|
| Block 16x16 INT (RTN) | 0.6149 | 0.3149 | 0.0941 | 0.0537 | 0.0257 |
| Group INT g32 (Q4_0-style) | 0.7982 | 0.5141 | 0.1300 | 0.0708 | 0.0334 |
| Group INT g128 (GPTQ-RTN-style) | 0.6109 | 0.3131 | 0.0936 | 0.0534 | 0.0256 |
| Group INT g128 + fp16 outliers (best of 0.5-4%) | 0.8211 | 0.8045 | 0.1536 | 0.0976 | 0.0454 |
| MTC scales, fixed width (ablation) | 0.3643 | 0.2412 | 0.0900 | 0.0533 | 0.0260 |
| MTC-RD, no importance (ablation) | 0.3776 | 0.2517 | 0.0906 | 0.0549 | 0.0268 |
| MTC-RD fixed 16x16 (v1) | 0.3776 | 0.2517 | 0.0906 | 0.0549 | 0.0268 |
| MTC-QT quadtree 32-4 (ours) | 0.3572 | 0.2147 | 0.0864 | 0.0478 | 0.0234 |

Fixed formats: MXFP4 0.2208 @ 4.25 b, NVFP4 0.1448 @ 4.50 b, MXINT8 0.0055 @ 8.25 b, INT4 0.1014 @ 4.00 b, INT8 0.0056 @ 8.00 b

## outliers (A@B, both quantized)

| Method | @2.5 b | @3.0 b | @4.25 b | @5.0 b | @6.0 b |
|---|---|---|---|---|---|
| Block 16x16 INT (RTN) | 0.4595 | 0.2436 | 0.0809 | 0.0542 | 0.0396 |
| Group INT g32 (Q4_0-style) | 0.2759 | 0.1825 | 0.0557 | 0.0366 | 0.0261 |
| Group INT g128 (GPTQ-RTN-style) | 0.3689 | 0.1980 | 0.0693 | 0.0478 | 0.0362 |
| Group INT g128 + fp16 outliers (best of 0.5-4%) | 0.1980 | 0.0450 | 0.0091 | 0.0072 | 0.0031 |
| MTC scales, fixed width (ablation) | 0.2729 | 0.1832 | 0.0759 | 0.0529 | 0.0396 |
| MTC-RD, no importance (ablation) | 0.2155 | 0.1341 | 0.0565 | 0.0428 | 0.0359 |
| MTC-RD fixed 16x16 (v1) | 0.2052 | 0.1289 | 0.0551 | 0.0422 | 0.0356 |
| MTC-QT quadtree 32-4 (ours) | 0.0273 | 0.0225 | 0.0148 | 0.0119 | 0.0085 |

Fixed formats: MXFP4 0.1620 @ 4.25 b, NVFP4 0.0569 @ 4.50 b, MXINT8 0.0187 @ 8.25 b, INT4 0.1074 @ 4.00 b, INT8 0.0320 @ 8.00 b

## multi_scale (A@B, both quantized)

| Method | @2.5 b | @3.0 b | @4.25 b | @5.0 b | @6.0 b |
|---|---|---|---|---|---|
| Block 16x16 INT (RTN) | 0.8504 | 0.5092 | 0.1651 | 0.0938 | 0.0451 |
| Group INT g32 (Q4_0-style) | 0.8968 | 0.6011 | 0.1769 | 0.0992 | 0.0480 |
| Group INT g128 (GPTQ-RTN-style) | 0.7293 | 0.4497 | 0.1683 | 0.1018 | 0.0518 |
| Group INT g128 + fp16 outliers (best of 0.5-4%) | 0.6951 | 0.5403 | 0.1627 | 0.0988 | 0.0494 |
| MTC scales, fixed width (ablation) | 0.4212 | 0.3032 | 0.1356 | 0.0846 | 0.0436 |
| MTC-RD, no importance (ablation) | 0.0186 | 0.0123 | 0.0050 | 0.0029 | 0.0015 |
| MTC-RD fixed 16x16 (v1) | 0.0172 | 0.0112 | 0.0044 | 0.0026 | 0.0013 |
| MTC-QT quadtree 32-4 (ours) | 0.0155 | 0.0099 | 0.0039 | 0.0023 | 0.0012 |

Fixed formats: MXFP4 0.1689 @ 4.25 b, NVFP4 0.1346 @ 4.50 b, MXINT8 0.0117 @ 8.25 b, INT4 0.4542 @ 4.00 b, INT8 0.0372 @ 8.00 b

## llm_like (weight-only, calibrated)

| Method | @2.5 b | @3.0 b | @4.25 b | @5.0 b | @6.0 b |
|---|---|---|---|---|---|
| Block 16x16 INT (RTN) | 0.7633 | 0.6321 | 0.3154 | 0.1885 | 0.0938 |
| Group INT g32 (Q4_0-style) | 0.7082 | 0.5469 | 0.2375 | 0.1399 | 0.0681 |
| Group INT g128 (GPTQ-RTN-style) | 0.7163 | 0.5737 | 0.2712 | 0.1630 | 0.0809 |
| Group INT g128 + fp16 outliers (best of 0.5-4%) | 0.7484 | 0.6889 | 0.2464 | 0.1452 | 0.0673 |
| AWQ g128 (activation-aware) | 0.3981 | 0.2929 | 0.1271 | 0.0754 | — |
| MTC scales, fixed width (ablation) | 0.3616 | 0.2940 | 0.1665 | 0.1162 | 0.0685 |
| MTC-RD, no importance (ablation) | 0.4827 | 0.3815 | 0.2101 | 0.1392 | 0.0749 |
| MTC-RD fixed 16x16 (v1) | 0.2556 | 0.2028 | 0.1084 | 0.0708 | 0.0386 |
| MTC-QT quadtree 32-4 (ours) | 0.2050 | 0.1558 | 0.0735 | 0.0453 | 0.0236 |

Fixed formats: MXFP4 0.1600 @ 4.25 b, NVFP4 0.1144 @ 4.50 b, MXINT8 0.0160 @ 8.25 b, INT4 0.9154 @ 4.00 b, INT8 0.1807 @ 8.00 b
