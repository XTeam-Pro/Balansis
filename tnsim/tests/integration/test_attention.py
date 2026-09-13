import numpy as np
import pytest

torch = pytest.importorskip("torch")
from tnsim.integrations.balansis_integration import (
    BalansisCompensator,
    ZeroSumAttention,
    ZeroSumTransformerBlock,
)


def test_numpy_compensation():
    x = np.array([1.0, 2.0, 3.0])
    result, metrics = BalansisCompensator().compensated_sum(x)
    assert np.array_equal(result, [-1.0, 0.0, 1.0])
    assert metrics.compensation_error == 0


def test_attention_forward_backward():
    torch.manual_seed(7)
    x = torch.randn(2, 3, 8, requires_grad=True)
    attention = ZeroSumAttention(8, 2, dropout=0)
    y, metadata = attention(x, x, x)
    assert y.shape == x.shape and torch.isfinite(y).all()
    y.square().sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert torch.isfinite(metadata["attention_weights"]).all()


def test_transformer_block():
    block = ZeroSumTransformerBlock(8, 2, d_ff=16, dropout=0)
    x = torch.randn(2, 3, 8, requires_grad=True)
    y, _ = block(x)
    assert y.shape == x.shape
    y.sum().backward()
    assert x.grad is not None
