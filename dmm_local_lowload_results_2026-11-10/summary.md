# 本机低负载验证摘要

- 源码：`0b8b2b8df622842b03d3e398593a7c3504770e75`，`git dirty=false`
- 机器：WSL2 Ubuntu 24.04，i5-13600K，10 physical / 20 logical cores
- 工具链：GCC 16.2.0、CMake 4.4.5、Ninja 1.13.2、Python 3.12.16、uv 0.13.7
- 负载：1-min load 约 0.12–0.34（低负载，不是 idle）

## Model 选型

| 结果集 | 配置数 | scenarios | cells | 选型 |
|---|---:|---:|---:|---|
| `model` | 24 | 192 | 144 | Cannon 141 / base 2.5D 3 |
| `model_direct` | 30 | 240 | 144 | Cannon 116 / 2.5d-direct 28 |
| `model_padding` | 9 | 9 | 6 | Cannon 6 |

本机 counts 与 `0b8b2b8` release 逐字段一致（`F_total/W_crit/C_total/C_crit/S/peak_words/messages`，`diffs=0`）。

## 15-rep measurement

| n | layout | P1 median | P4 median | speedup | P4 GFLOP/s | residual (P4) |
|---:|---|---:|---:|---:|---:|---:|
| 256 | Cannon | 1.505 ms | 1.852 ms | 0.81× | 18.1 | +10.7% |
| 256 | Ring | 1.463 ms | 1.419 ms | 1.03× | 23.7 | +13.3% |
| 512 | Cannon | 13.47 ms | 5.00 ms | 2.70× | 53.7 | +12.5% |
| 512 | Ring | 13.48 ms | 5.16 ms | 2.61× | 52.0 | +23.6% |
| 1024 | Cannon | 112.5 ms | 32.30 ms | 3.48× | 66.5 | +19.4% |
| 1024 | Ring | 111.8 ms | 31.80 ms | 3.52× | 67.5 | +19.6% |

- `measure_15rep/`：5 warmup + 15 reps，每 context 630 样本。
- `measure_5rep/`：2 warmup + 5 reps，低负载基线。
- `measure_15rep/*/manifest.json` 记录 load before/after、physical/logical cores、git。

## 补充 2.5D 检查

| layout | P | c | n | padded | S | max_abs | fro_rel | peak words |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2.5d | 256 | 4 | 65 | 256 | 11 | 2.046e-6 | 1.331e-7 | 7168 |
| 2.5d-direct | 256 | 4 | 65 | 256 | 6 | 2.046e-6 | 1.331e-7 | 7168 |
| 2.5d | 108 | 3 | 33 | 108 | 9 | 9.246e-7 | 9.148e-8 | 2268 |
| 2.5d-direct | 108 | 3 | 33 | 108 | 6 | 9.246e-7 | 9.148e-8 | 2268 |
| 2.5d | 216 | 6 | 33 | 216 | 10 | 1.031e-6 | 1.135e-7 | 9072 |
| 2.5d-direct | 216 | 6 | 33 | 216 | 5 | 1.031e-6 | 1.135e-7 | 9072 |
| 2.5d | 125 | 5 | 33 | 125 | 10 | 8.235e-7 | 9.350e-8 | 4375 |
| 2.5d-direct | 125 | 5 | 33 | 125 | 5 | 8.235e-7 | 9.350e-8 | 4375 |

全部通过 FP64 误差规则和 `validate_coverage`。

## 结论

- 核心 code/测试/模型计数在本机低负载下可复现。
- base 2.5D 只赢 3 格，28 格属于 2.5d-direct；不要混写。
- timing 是本机特定结果，不能与 8 核 macOS release 直接比较。
