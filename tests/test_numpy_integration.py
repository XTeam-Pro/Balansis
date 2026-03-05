import numpy as np
from balansis.core.absolute import AbsoluteValue
from balansis.numpy_integration import to_numpy, from_numpy, ufunc_add, ufunc_log

def test_to_from_numpy_roundtrip():
    values = [AbsoluteValue.from_float(1.0), AbsoluteValue.from_float(-2.0)]
    arr = to_numpy(values)
    back = from_numpy(arr)
    assert back[0] == values[0]
    assert back[1] == values[1]

def test_numpy_ufunc_add():
    a = np.array([AbsoluteValue.from_float(1.0), AbsoluteValue.from_float(2.0)], dtype=object)
    b = np.array([AbsoluteValue.from_float(3.0), AbsoluteValue.from_float(-1.0)], dtype=object)
    c = ufunc_add(a, b)
    assert c[0].to_float() == 4.0
    assert c[1].to_float() == 1.0

def test_numpy_ufunc_log():
    a = np.array([AbsoluteValue.from_float(1.0), AbsoluteValue.from_float(4.0)], dtype=object)
    l = ufunc_log(a)
    assert abs(l[0] - 0.0) < 1e-12
    assert abs(l[1] - np.log(4.0)) < 1e-12


def test_compensated_array_add():
    from balansis.numpy_integration import compensated_array_add
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, -1.0, 0.5])
    result = compensated_array_add(a, b)
    expected = np.array([5.0, 1.0, 3.5])
    np.testing.assert_allclose(result, expected, atol=1e-10)


def test_compensated_array_add_preserves_shape():
    from balansis.numpy_integration import compensated_array_add
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    b = np.array([[5.0, 6.0], [7.0, 8.0]])
    result = compensated_array_add(a, b)
    assert result.shape == (2, 2)
    np.testing.assert_allclose(result, np.array([[6.0, 8.0], [10.0, 12.0]]), atol=1e-10)


def test_compensated_array_multiply():
    from balansis.numpy_integration import compensated_array_multiply
    a = np.array([2.0, 3.0, -1.0])
    b = np.array([4.0, -2.0, 5.0])
    result = compensated_array_multiply(a, b)
    expected = np.array([8.0, -6.0, -5.0])
    np.testing.assert_allclose(result, expected, atol=1e-10)


def test_compensated_dot_product():
    from balansis.numpy_integration import compensated_dot_product
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0, 6.0])
    result = compensated_dot_product(a, b)
    assert abs(result - 32.0) < 1e-10


def test_compensated_dot_product_orthogonal():
    from balansis.numpy_integration import compensated_dot_product
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    result = compensated_dot_product(a, b)
    assert abs(result) < 1e-10


def test_compensated_outer_product():
    from balansis.numpy_integration import compensated_outer_product
    a = np.array([1.0, 2.0])
    b = np.array([3.0, 4.0])
    result = compensated_outer_product(a, b)
    expected = np.array([[3.0, 4.0], [6.0, 8.0]])
    np.testing.assert_allclose(result, expected, atol=1e-10)
    assert result.shape == (2, 2)


def test_compensated_softmax():
    from balansis.numpy_integration import compensated_softmax
    logits = np.array([1.0, 2.0, 3.0])
    result = compensated_softmax(logits)
    # Compare with standard softmax
    e = np.exp(logits - np.max(logits))
    expected = e / e.sum()
    np.testing.assert_allclose(result, expected, atol=1e-6)
    # Probabilities must sum to 1
    assert abs(result.sum() - 1.0) < 1e-6


def test_compensated_softmax_uniform():
    from balansis.numpy_integration import compensated_softmax
    logits = np.array([0.0, 0.0, 0.0])
    result = compensated_softmax(logits)
    np.testing.assert_allclose(result, np.array([1/3, 1/3, 1/3]), atol=1e-6)
