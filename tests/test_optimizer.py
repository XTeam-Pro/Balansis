"""Tests for ml/optimizer.py - EternalOptimizer with momentum and weight decay."""

import pytest

from balansis.ml.optimizer import EternalOptimizer, AdaptiveEternalOptimizer

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


class TestAdaptiveEternalOptimizerCreation:
    """Test AdaptiveEternalOptimizer creation and validation."""

    def test_creation_default_params(self):
        """Test optimizer creation with default parameters."""
        opt = AdaptiveEternalOptimizer([], lr=1e-3)
        assert len(opt.param_groups) == 1
        assert opt.param_groups[0]["lr"] == 1e-3
        assert opt.param_groups[0]["betas"] == (0.9, 0.999)
        assert opt._global_step == 0

    def test_creation_custom_params(self):
        """Test optimizer creation with custom parameters."""
        opt = AdaptiveEternalOptimizer(
            [], lr=0.01, betas=(0.8, 0.99), eps=1e-7,
            weight_decay=0.01, max_grad_norm=5.0,
            warmup_steps=100, total_steps=1000,
        )
        assert opt.param_groups[0]["lr"] == 0.01
        assert opt.param_groups[0]["betas"] == (0.8, 0.99)
        assert opt.warmup_steps == 100
        assert opt.total_steps == 1000

    def test_invalid_lr(self):
        """Test negative learning rate raises ValueError."""
        with pytest.raises(ValueError, match="Invalid learning rate"):
            AdaptiveEternalOptimizer([], lr=-0.1)

    def test_invalid_beta1(self):
        """Test invalid beta1 raises ValueError."""
        with pytest.raises(ValueError, match="Invalid beta1"):
            AdaptiveEternalOptimizer([], betas=(1.0, 0.999))

    def test_invalid_beta2(self):
        """Test invalid beta2 raises ValueError."""
        with pytest.raises(ValueError, match="Invalid beta2"):
            AdaptiveEternalOptimizer([], betas=(0.9, -0.1))

    def test_invalid_weight_decay(self):
        """Test negative weight decay raises ValueError."""
        with pytest.raises(ValueError, match="Invalid weight_decay"):
            AdaptiveEternalOptimizer([], weight_decay=-1.0)

    def test_invalid_max_grad_norm(self):
        """Test negative max_grad_norm raises ValueError."""
        with pytest.raises(ValueError, match="Invalid max_grad_norm"):
            AdaptiveEternalOptimizer([], max_grad_norm=-1.0)

    def test_invalid_warmup_steps(self):
        """Test negative warmup_steps raises ValueError."""
        with pytest.raises(ValueError, match="Invalid warmup_steps"):
            AdaptiveEternalOptimizer([], warmup_steps=-1)

    def test_step_without_torch_params(self):
        """Test step is safe when no parameters have gradients."""
        opt = AdaptiveEternalOptimizer([], lr=1e-3)
        opt.step()  # Should not raise


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestAdaptiveEternalOptimizerWithTorch:
    """Test AdaptiveEternalOptimizer with PyTorch tensors."""

    def test_basic_step(self):
        """Test that a basic step updates parameter values."""
        param = torch.nn.Parameter(torch.tensor([1.0, 2.0, 3.0]))
        param.grad = torch.tensor([0.1, 0.2, 0.3])
        original = param.data.clone()

        opt = AdaptiveEternalOptimizer([param], lr=1e-2, max_grad_norm=0.0)
        opt.step()

        assert not torch.equal(param.data, original)
        assert opt._global_step == 1
        assert id(param) in opt.state

    def test_convergence_quadratic(self):
        """Test convergence on simple quadratic f(x) = ||x||^2."""
        x = torch.nn.Parameter(torch.tensor([5.0, 5.0]))
        opt = AdaptiveEternalOptimizer(
            [x], lr=0.1, max_grad_norm=0.0
        )

        for _ in range(300):
            x.grad = 2.0 * x.data.clone()
            opt.step()

        loss = (x.data ** 2).sum().item()
        assert loss < 0.01, f"Failed to converge: loss={loss}"

    def test_gradient_clipping(self):
        """Test gradient clipping via ACT normalization."""
        param = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
        opt = AdaptiveEternalOptimizer([param], lr=0.01, max_grad_norm=2.0)

        # Gradient with norm 5.0 should be clipped to norm 2.0
        g = torch.tensor([3.0, 4.0])
        clipped = opt._clip_grad_by_norm(g, 2.0)
        clipped_norm = torch.linalg.norm(clipped).item()
        assert abs(clipped_norm - 2.0) < 1e-5

        # Gradient with norm < max should not be clipped
        small_g = torch.tensor([0.5, 0.5])
        not_clipped = opt._clip_grad_by_norm(small_g, 2.0)
        assert torch.equal(not_clipped, small_g)

        # max_grad_norm=0 disables clipping
        no_clip = opt._clip_grad_by_norm(g, 0.0)
        assert torch.equal(no_clip, g)

    def test_warmup_schedule(self):
        """Test learning rate warmup schedule."""
        opt = AdaptiveEternalOptimizer(
            [torch.nn.Parameter(torch.tensor([1.0]))],
            lr=0.1, warmup_steps=10,
        )

        # During warmup: linear scale from 0.1 to 1.0
        for i in range(10):
            expected = (i + 1) / 10.0
            actual = opt._get_lr_scale(i)
            assert abs(actual - expected) < 1e-10, (
                f"step={i}: expected={expected}, actual={actual}"
            )

        # After warmup with no decay: scale should be 1.0
        for i in range(10, 15):
            assert abs(opt._get_lr_scale(i) - 1.0) < 1e-10

    def test_cosine_decay_schedule(self):
        """Test cosine decay after warmup."""
        opt = AdaptiveEternalOptimizer(
            [torch.nn.Parameter(torch.tensor([1.0]))],
            lr=0.1, warmup_steps=10, total_steps=110,
        )

        # At start of decay (step 10): scale ~ 1.0
        scale_start = opt._get_lr_scale(10)
        assert abs(scale_start - 1.0) < 0.01

        # At end of decay (step 109): scale ~ 0.0
        scale_end = opt._get_lr_scale(109)
        assert scale_end < 0.05

        # At midpoint: scale ~ 0.5
        scale_mid = opt._get_lr_scale(60)
        assert 0.3 < scale_mid < 0.7

    def test_parameter_groups(self):
        """Test parameter groups with different settings."""
        p1 = torch.nn.Parameter(torch.tensor([5.0]))
        p2 = torch.nn.Parameter(torch.tensor([5.0]))

        opt = AdaptiveEternalOptimizer([
            {"params": [p1], "lr": 0.1, "weight_decay": 0.0},
            {"params": [p2], "lr": 0.01, "weight_decay": 0.0},
        ], max_grad_norm=0.0)

        assert len(opt.param_groups) == 2
        assert opt.param_groups[0]["lr"] == 0.1
        assert opt.param_groups[1]["lr"] == 0.01

        for _ in range(100):
            p1.grad = 2.0 * p1.data.clone()
            p2.grad = 2.0 * p2.data.clone()
            opt.step()

        # p1 with higher lr should converge faster (closer to 0)
        assert abs(p1.data.item()) < abs(p2.data.item())

    def test_comparison_with_eternal_optimizer(self):
        """AdaptiveEternalOptimizer should converge at least as well."""
        x1 = torch.nn.Parameter(torch.tensor([3.0, 3.0]))
        x2 = torch.nn.Parameter(torch.tensor([3.0, 3.0]))

        opt1 = EternalOptimizer([x1], lr=0.1)
        opt2 = AdaptiveEternalOptimizer(
            [x2], lr=0.1, max_grad_norm=0.0
        )

        for _ in range(300):
            x1.grad = 2.0 * x1.data.clone()
            x2.grad = 2.0 * x2.data.clone()
            opt1.step()
            opt2.step()

        loss1 = (x1.data ** 2).sum().item()
        loss2 = (x2.data ** 2).sum().item()
        # Both should converge; adaptive should be at least comparable
        assert loss1 < 1.0, f"EternalOptimizer failed: {loss1}"
        assert loss2 < 1.0, f"AdaptiveEternalOptimizer failed: {loss2}"
        assert loss2 <= loss1 + 0.1

    def test_weight_decay(self):
        """Test weight decay reduces parameter magnitude."""
        param = torch.nn.Parameter(torch.tensor([10.0, 10.0]))
        param.grad = torch.tensor([0.0, 0.0])

        opt = AdaptiveEternalOptimizer(
            [param], lr=1e-2, weight_decay=0.1, max_grad_norm=0.0
        )
        original_norm = torch.linalg.norm(param.data).item()
        opt.step()
        new_norm = torch.linalg.norm(param.data).item()

        assert new_norm < original_norm
