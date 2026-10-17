import os
import subprocess
import sys


def test_package_import_and_thread_limits_outside_project(tmp_path):
    limits = dict.fromkeys((
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ), "8")
    code = (
        "import os; import dmm.datasets; "
        f"assert all(os.environ[name] == '1' for name in {tuple(limits)!r})"
    )
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env={**os.environ, **limits},
        check=True,
    )
