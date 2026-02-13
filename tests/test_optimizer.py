"""Tests for ml/optimizer.py - EternalOptimizer with momentum and weight decay."""

import pytest

from balansis.ml.optimizer import EternalOptimizer

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class TestEternalOptimizerCreation:
    """Test EternalOptimizer creation and validation."""

    def test_creation_default_params(self):
        """Test optimizer creation with default parameters."""
        opt = EternalOptimizer([], lr=1e-3)
        assert opt.lr == 1e-3
        assert opt.momentum == 0.0
        assert opt.weight_decay == 0.0

    def test_creation_with_momentum(self):
        """Test optimizer creation with momentum."""
        opt = EternalOptimizer([], lr=1e-3, momentum=0.9)
        assert opt.momentum == 0.9

    def test_creation_with_weight_decay(self):
        """Test optimizer creation with weight decay."""
        opt = EternalOptimizer([], lr=1e-3, weight_decay=0.01)
        assert opt.weight_decay == 0.01

    def test_creation_all_params(self):
        """Test optimizer creation with all custom parameters."""
        opt = EternalOptimizer([], lr=0.01, momentum=0.95, weight_decay=1e-4)
        assert opt.lr == 0.01
        assert opt.momentum == 0.95
        assert opt.weight_decay == 1e-4

    def test_invalid_lr_raises_error(self):
        """Test negative learning rate raises ValueError."""
        with pytest.raises(ValueError, match="Invalid learning rate"):
            EternalOptimizer([], lr=-0.1)

    def test_invalid_momentum_negative(self):
        """Test negative momentum raises ValueError."""
        with pytest.raises(ValueError, match="Invalid momentum"):
            EternalOptimizer([], lr=1e-3, momentum=-0.1)

    def test_invalid_momentum_ge_one(self):
        """Test momentum >= 1.0 raises ValueError."""
        with pytest.raises(ValueError, match="Invalid momentum"):
            EternalOptimizer([], lr=1e-3, momentum=1.0)

    def test_invalid_weight_decay_raises_error(self):
        """Test negative weight decay raises ValueError."""
        with pytest.raises(ValueError, match="Invalid weight_decay"):
            EternalOptimizer([], lr=1e-3, weight_decay=-0.1)

    def test_state_initialization(self):
        """Test state dict is initially empty."""
        opt = EternalOptimizer([], lr=1e-3)
        assert opt.state == {}

    def test_get_state_initializes(self):
        """Test _get_state creates initial state."""
        opt = EternalOptimizer([], lr=1e-3)
        state = opt._get_state(42)
        assert state["momentum_buffer"] is None
        assert state["step"] == 0

    def test_get_state_returns_existing(self):
        """Test _get_state returns existing state."""
        opt = EternalOptimizer([], lr=1e-3)
        state1 = opt._get_state(42)
        state1["step"] = 5
        state2 = opt._get_state(42)
        assert state2["step"] == 5

    def test_step_without_torch(self):
        """Test step is safe when no parameters have gradients."""
        opt = EternalOptimizer([], lr=1e-3)
        opt.step()  # Should not raise


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestEternalOptimizerWithTorch:
    """Test EternalOptimizer with PyTorch tensors."""

    def test_step_updates_parameters(self):
        """Test that step updates parameter values."""
        param = torch.nn.Parameter(torch.tensor([1.0, 2.0, 3.0]))
        param.grad = torch.tensor([0.1, 0.2, 0.3])
        original = param.data.clone()

        opt = EternalOptimizer([param], lr=1e-2)
        opt.step()

        assert not torch.equal(param.data, original)

    def test_step_with_momentum(self):
        """Test step with momentum accumulation."""
        param = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
        param.grad = torch.tensor([0.5, 0.5])

        opt = EternalOptimizer([param], lr=1e-2, momentum=0.9)
        opt.step()
        opt.step()

        state = opt._get_state(id(param))
        assert state["momentum_buffer"] is not None
        assert state["step"] == 2

    def test_step_with_weight_decay(self):
        """Test step with weight decay reduces parameter magnitude."""
        param = torch.nn.Parameter(torch.tensor([10.0, 10.0]))
        param.grad = torch.tensor([0.0, 0.0])

        opt = EternalOptimizer([param], lr=1e-2, weight_decay=0.1)
        original_norm = torch.linalg.norm(param.data).item()
        opt.step()
        new_norm = torch.linalg.norm(param.data).item()

        assert new_norm < original_norm

    def test_step_skips_none_grad(self):
        """Test step skips parameters without gradients."""
        param1 = torch.nn.Parameter(torch.tensor([1.0]))
        param1.grad = None
        param2 = torch.nn.Parameter(torch.tensor([2.0]))
        param2.grad = torch.tensor([0.1])

        original1 = param1.data.clone()
        opt = EternalOptimizer([param1, param2], lr=1e-2)
        opt.step()

        assert torch.equal(param1.data, original1)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestEternalTorchOptimizer:
    """Test EternalTorchOptimizer PyTorch integration."""

    def test_creation(self):
        """Test EternalTorchOptimizer creation."""
        from balansis.ml.optimizer import EternalTorchOptimizer
        param = torch.nn.Parameter(torch.randn(3))
        opt = EternalTorchOptimizer([param], lr=1e-3)
        assert len(opt.param_groups) == 1

    def test_step_basic(self):
        """Test basic step updates parameters."""
        from balansis.ml.optimizer import EternalTorchOptimizer
        param = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
        param.grad = torch.tensor([0.1, 0.1])
        opt = EternalTorchOptimizer([param], lr=1e-2)
        original = param.data.clone()
        opt.step()
        assert not torch.equal(param.data, original)

    def test_step_with_closure(self):
        """Test step with closure returns loss."""
        from balansis.ml.optimizer import EternalTorchOptimizer
        param = torch.nn.Parameter(torch.tensor([1.0]))
        param.grad = torch.tensor([0.1])
        opt = EternalTorchOptimizer([param], lr=1e-2)

        def closure():
            return torch.tensor(0.5)

        loss = opt.step(closure=closure)
        assert loss is not None

    def test_step_with_momentum_and_weight_decay(self):
        """Test step with momentum and weight decay."""
        from balansis.ml.optimizer import EternalTorchOptimizer
        param = torch.nn.Parameter(torch.tensor([5.0, 5.0]))
        param.grad = torch.tensor([1.0, 1.0])
        opt = EternalTorchOptimizer(
            [param], lr=1e-2, momentum=0.9, weight_decay=0.01
        )
        opt.step()
        opt.step()
        state = opt.state[param]
        assert state["momentum_buffer"] is not None
        assert state["step"] == 2

    def test_creation_with_all_params(self):
        """Test EternalTorchOptimizer with all parameters."""
        from balansis.ml.optimizer import EternalTorchOptimizer
        param = torch.nn.Parameter(torch.randn(3))
        opt = EternalTorchOptimizer(
            [param], lr=0.01, momentum=0.9, weight_decay=1e-4
        )
        group = opt.param_groups[0]
        assert group["lr"] == 0.01
        assert group["momentum"] == 0.9
        assert group["weight_decay"] == 1e-4
