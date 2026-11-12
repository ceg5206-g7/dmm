# dmm local low-load validation results

CEG5206 团队项目 `dmm` 的本机低负载验证包。

- 生成日期：2026-11-10
- 源码：`0b8b2b8df622842b03d3e398593a7c3504770e75`
- 工作树：`git dirty = false`
- 本机：WSL2 Ubuntu 24.04，i5-13600K，10 physical / 20 logical cores
- 工具链：GCC 16.2.0（conda-forge）、CMake 4.4.5、Ninja 1.13.2、Python 3.12.16、uv 0.13.7
- 负载：测量窗口 1-min load 约 **0.12–0.34**；**低负载但不是 idle**

## 目录

```
dmm_local_lowload_results_2026-11-10/
├── README.md
├── summary.md / summary.json      ← 结果摘要
├── source_commit.txt              ← commit + git status
├── targeted_2.5d_checks.md        ← 2.5D 主网格/c≠2^k 的补充检查
├── SHA256SUMS                     ← 包内文件校验和
├── configs/                       ← 本次运行所用 YAML
├── inputs/                        ← 数据集 manifest（矩阵本身不打包）
├── model/                         ← 24 配置、192 scenarios、144 cells
├── model_direct/                  ← 30 配置、240 scenarios、144 cells
├── model_padding/                 ← 9 配置、6 cells
├── measure_5rep/
│   ├── serial/                    ← 2 warmup + 5 reps
│   └── threaded/                  ← P=4, 2 warmup + 5 reps
├── measure_15rep/
│   ├── serial/                    ← 5 warmup + 15 reps
│   └── threaded/                  ← P=4, 5 warmup + 15 reps
└── smoke/
    ├── model_smoke/               ← 4 cases × 2 sizes
    ├── benchmark_smoke.csv/.json  ← 72 samples
    └── benchmark_smoke_analysis/  ← summary/time/speedup
```

## 关键结果

### Model / counts

| 结果集 | 配置数 | scenarios | cells | 选型 |
|---|---:|---:|---:|---|
| `model` | 24 | 192 | 144 | Cannon 141，base 2.5D **3** |
| `model_direct` | 30 | 240 | 144 | Cannon 116，2.5d-direct **28** |
| `model_padding` | 9 | 9 | 6 | Cannon 6 |

本机重跑的 `F_total / W_crit / C_total / C_crit / S / peak_words / messages` 与
`0b8b2b8` release **逐字段一致**（192/240/9 行，`diffs=0`）。

### 15-rep measurement

| n | layout | P1 median | P4 median | P4 speedup | P4 GFLOP/s |
|---:|---|---:|---:|---:|---:|
| 256 | Cannon | 1.505 ms | 1.852 ms | 0.81× | 18.1 |
| 256 | Ring | 1.463 ms | 1.419 ms | 1.03× | 23.7 |
| 512 | Cannon | 13.47 ms | 5.00 ms | 2.70× | 53.7 |
| 512 | Ring | 13.48 ms | 5.16 ms | 2.61× | 52.0 |
| 1024 | Cannon | 112.5 ms | 32.30 ms | **3.48×** | 66.5 |
| 1024 | Ring | 111.8 ms | 31.80 ms | **3.52×** | 67.5 |

- 每 context **630 个 phase/end-to-end 样本**（5 warmup + 15 reps）。
- 本机是 x86/WSL2，不能把绝对时间与 8 核 macOS release 直接比较。
- n=256 P=4 只有 0.81×/1.03×，符合小矩阵 barrier/调度占比高的预期。

### 补充 2.5D 检查

见 `targeted_2.5d_checks.md`：`P=256,c=4`、`P=108,c=3`、`P=216,c=6`、`P=125,c=5`
的 base/direct 均通过 FP64 与 coverage 检查。

## 复现

```bash
source ~/.miniconda3/etc/profile.d/conda.sh
conda activate dmm
export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"

cd /tmp/opencode/dmm
uv sync --locked --python "$CONDA_PREFIX/bin/python"
uv run cmake --preset release && uv run cmake --build --preset release --parallel
ctest --preset release
uv run --locked pytest -q

# 数据集需要自行生成（本包只带 manifest）
uv run --locked dmm generate --config configs/dataset.yaml

# 注意：本包 configs 使用 ./data 相对路径；解包后按需修改 data_dir
```

## 说明与限制

- **低负载 ≠ idle**：15-rep measurement 的 1-min load 为 0.12–0.34，仍有系统背景活动；不要写成 dedicated host。
- **timing 不可跨机比较**：本机 10 physical/20 logical，release 是 8 核 macOS；绝对时间/speedup 只描述本机实现。
- **model/counts 与 load 无关**：确定性；本机在 clean tree 下复现 release 计数。
- `measure_5rep/` 是较早的低负载 5-rep 基线；`measure_15rep/` 是正式 5/15 协议。
- 数据集矩阵未打包，避免包体过大；`inputs/` 保留 seed/group/size/range manifest，configs 可重新生成。
- 人工高负载实验目录（`measure_*_load`）未包含在本包中。
- 部分 `model.json` / `manifest.json` 的 `trace_directory`、`data_dir`、`output_dir` 保留生成时的绝对路径；包内已包含对应 traces/CSV，离线读取不依赖这些路径。
