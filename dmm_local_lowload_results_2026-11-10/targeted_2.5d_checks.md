# 补充 2.5D 检查

`tests/cpp/test_two_point_five_d.cpp` 的 grid 列表没有覆盖 proposal 主配置
`P=256,c=4,q=8`，也没有覆盖非 2 的幂的 `c`。本机用当前 `0b8b2b8` 代码额外跑了以下检查。

每次检查：
- 随机 FP32 A/B，FP64 NumPy reference；
- `validate_coverage(trace)`；
- 使用 `SerialExecutor`；
- 记录 `S`、`max_abs_error`、Frobenius relative error、`peak_words`。

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

`P=256,c=4` 的 peak 7168 words，归一化 `7168 / (256²/256) = 28`，与 `model` trace 一致。

复现命令：

```bash
source ~/.miniconda3/etc/profile.d/conda.sh
conda activate dmm
cd /tmp/opencode/dmm
uv run --locked python - <<'PY'
import numpy as np
from dmm import matmul as shared
from dmm import _native
from dmm.execution import SerialExecutor

for layout, P, c, n in [
    ("2.5d", 256, 4, 65), ("2.5d-direct", 256, 4, 65),
    ("2.5d", 108, 3, 33), ("2.5d-direct", 108, 3, 33),
    ("2.5d", 216, 6, 33), ("2.5d-direct", 216, 6, 33),
    ("2.5d", 125, 5, 33), ("2.5d-direct", 125, 5, 33),
]:
    rng = np.random.default_rng([5206, P, c, n])
    A = rng.uniform(-1, 1, (n, n)).astype(np.float32)
    B = rng.uniform(-1, 1, (n, n)).astype(np.float32)
    ref = A.astype(np.float64) @ B.astype(np.float64)
    with SerialExecutor() as ex:
        C, trace = shared.matmul_traced(A, B, ex, layout=layout, P=P, c=c)
    _native.validate_coverage(trace)
    diff = np.abs(C.astype(np.float64) - ref)
    rel = np.linalg.norm(diff) / np.linalg.norm(ref)
    print(layout, P, c, n, len(trace.supersteps), diff.max(), rel, max(trace.peak_words))
PY
```
