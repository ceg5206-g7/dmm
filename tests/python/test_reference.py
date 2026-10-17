import numpy as np
import pytest

from dmm.datasets import reference_matmul


def test_known_product_uses_fp64_without_changing_inputs(monkeypatch):
    a = np.array([[1, 2], [3, 4]], dtype=np.float32)
    b = np.array([[5, 6], [7, 8]], dtype=np.float32)
    a.flags.writeable = False
    b.flags.writeable = False
    calls = []
    multiply = np.matmul

    def tracked(left, right):
        calls.append((left.dtype, right.dtype))
        np.testing.assert_array_equal(left, a)
        np.testing.assert_array_equal(right, b)
        return multiply(left, right)

    monkeypatch.setattr(np, "matmul", tracked)
    result = reference_matmul(a, b)
    assert calls == [(np.dtype("float64"), np.dtype("float64"))]
    np.testing.assert_array_equal(result, [[19, 22], [43, 50]])
    np.testing.assert_array_equal(a, [[1, 2], [3, 4]])
    np.testing.assert_array_equal(b, [[5, 6], [7, 8]])
    assert result.dtype == np.float64
    assert not np.shares_memory(result, a) and not np.shares_memory(result, b)
    assert not a.flags.writeable and not b.flags.writeable


def test_negative_noncontiguous_inputs_match_an_independent_scalar_product():
    rng = np.random.default_rng(5206)
    a = rng.uniform(-8, 9, size=(8, 8)).astype(np.float32).T
    b = rng.uniform(-8, 9, size=(8, 8)).astype(np.float32)[:, ::-1]
    expected = [[sum(float(a[i, k]) * float(b[k, j]) for k in range(8))
                 for j in range(8)] for i in range(8)]
    np.testing.assert_allclose(reference_matmul(a, b), expected, rtol=1e-12, atol=1e-12)


def test_reference_retains_precision_beyond_fp32():
    value = 1 + 2**-23
    a = np.array([[value]], dtype=np.float32)
    result = reference_matmul(a, a)
    assert result[0, 0] == value * value
    assert result[0, 0] != float(np.float32(value * value))


def test_fp64_accumulation_preserves_cancellation_residual():
    a = np.array([[1e8, 1, -1e8], [0, 0, 0], [0, 0, 0]], dtype=np.float32)
    result = reference_matmul(a, np.ones((3, 3), dtype=np.float32))
    np.testing.assert_array_equal(result, [[1, 1, 1], [0, 0, 0], [0, 0, 0]])


def test_large_fp32_values_produce_finite_fp64_reference():
    a = np.full((1, 1), np.finfo(np.float32).max, dtype=np.float32)
    result = reference_matmul(a, a)
    assert result.dtype == np.float64 and np.isfinite(result).all()
    assert result[0, 0] == float(a[0, 0]) ** 2


@pytest.mark.parametrize("invalid", [None, [[1]]])
@pytest.mark.parametrize("side", ["left", "right"])
def test_non_arrays_are_rejected(invalid, side):
    valid = np.ones((2, 2), dtype=np.float32)
    a, b = (invalid, valid) if side == "left" else (valid, invalid)
    with pytest.raises(TypeError, match="NumPy arrays"):
        reference_matmul(a, b)


@pytest.mark.parametrize("shape", [(), (2,), (0, 0), (2, 3), (2, 2, 2)])
def test_invalid_shapes_are_rejected(shape):
    a = np.zeros(shape, dtype=np.float32)
    with pytest.raises(ValueError, match="non-empty square"):
        reference_matmul(a, a)


def test_mismatched_shapes_are_rejected():
    with pytest.raises(ValueError, match="matching shapes"):
        reference_matmul(np.zeros((2, 2), dtype=np.float32), np.zeros((4, 4), dtype=np.float32))


@pytest.mark.parametrize("dtype", [np.int64, np.uint8, np.float64, object])
@pytest.mark.parametrize("side", ["left", "right"])
def test_non_fp32_arrays_are_rejected(dtype, side):
    valid = np.ones((2, 2), dtype=np.float32)
    invalid = valid.astype(dtype)
    a, b = (invalid, valid) if side == "left" else (valid, invalid)
    with pytest.raises(TypeError, match="dtype float32"):
        reference_matmul(a, b)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("side", ["left", "right"])
def test_non_finite_inputs_are_rejected(value, side):
    valid = np.ones((2, 2), dtype=np.float32)
    invalid = valid.copy()
    invalid[0, 0] = value
    a, b = (invalid, valid) if side == "left" else (valid, invalid)
    with pytest.raises(ValueError, match="finite"):
        reference_matmul(a, b)
