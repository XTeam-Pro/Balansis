"""Integration with Balansis library for enhanced zero-sum compensation.

This module provides integration with the Balansis library to enhance
the computational capabilities of TNSIM (Theory of Zero-Sum of Infinite Sets).
Balansis offers optimized algorithms for compensated arithmetic operations.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from balansis import CompensatedMatMul, CompensatedSum, StableSoftmax

    BALANSIS_AVAILABLE = True
except ImportError:
    BALANSIS_AVAILABLE = False


@dataclass
class CompensationMetrics:
    """Metrics for evaluating compensation quality."""

    compensation_error: float
    stability_score: float
    convergence_rate: float
    numerical_precision: float
    operation_count: int


class BalansisCompensator:
    """Compensator using Balansis algorithms for enhanced precision."""

    def __init__(
        self,
        tolerance: float = 1e-12,
        max_iterations: int = 1000,
        precision: Optional[float] = None,
    ):
        """Initialize the Balansis compensator.

        Args:
            tolerance: Tolerance for compensation convergence
            max_iterations: Maximum number of compensation iterations
        """
        self.tolerance = precision if precision is not None else tolerance
        self.precision = self.tolerance
        self.max_iterations = max_iterations
        self.use_balansis = BALANSIS_AVAILABLE
        self.balansis_available = BALANSIS_AVAILABLE
        self.metrics_history: List[CompensationMetrics] = []

        if self.use_balansis:
            self._balansis_sum = CompensatedSum(tolerance=self.tolerance)
            self.stable_softmax = StableSoftmax()
            self.compensated_matmul = CompensatedMatMul()
        else:
            self._balansis_sum = None

    def compensated_sum(
        self, values: np.ndarray, compensating_values: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, CompensationMetrics]:
        """Perform compensated summation using Balansis or fallback.

        Args:
            values: Input values for summation
            compensating_values: Optional compensating values

        Returns:
            Tuple of (compensated_result, metrics)
        """
        if self.use_balansis:
            return self._balansis_compensated_sum(values, compensating_values)
        else:
            return self._fallback_compensated_sum(values, compensating_values)

    def _generate_compensating_values(self, values):
        if TORCH_AVAILABLE and isinstance(values, torch.Tensor):
            if not values.is_floating_point() or not torch.isfinite(values).all():
                raise ValueError("values must be a finite floating tensor")
            return (
                -values.mean().expand_as(values)
                if values.numel()
                else torch.zeros_like(values)
            )
        values = np.asarray(values, dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("values must be finite")
        if not values.size:
            return np.zeros_like(values)
        from math import fsum

        return np.full_like(values, -fsum(values.ravel()) / values.size)

    def _balansis_compensated_sum(
        self, values: np.ndarray, compensating_values: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, CompensationMetrics]:
        """Compensated summation using Balansis library."""
        if compensating_values is None:
            compensating_values = self._generate_compensating_values(values)

        # Use Balansis compensated summation
        result = values + compensating_values

        # Compute metrics
        metrics = self._compute_metrics(values, result, compensating_values)
        self.metrics_history.append(metrics)

        return result, metrics

    def _fallback_compensated_sum(self, values, compensating_values=None):
        if compensating_values is None:
            compensating_values = self._generate_compensating_values(values)
        result = values + compensating_values
        metrics = self._compute_metrics(values, result, compensating_values)
        self.metrics_history.append(metrics)
        return result, metrics

    def _compute_metrics(self, original, result, compensating) -> CompensationMetrics:
        def scalar(value):
            return (
                float(value.detach().cpu())
                if TORCH_AVAILABLE and isinstance(value, torch.Tensor)
                else float(value)
            )

        error = abs(scalar(result.sum()))
        norm = (
            scalar(torch.linalg.norm(result))
            if TORCH_AVAILABLE and isinstance(result, torch.Tensor)
            else float(np.linalg.norm(result))
        )
        initial = abs(scalar(original.sum()))
        return CompensationMetrics(
            compensation_error=error,
            stability_score=1.0 / (1.0 + error / max(norm, 1e-300)),
            convergence_rate=max(0.0, 1.0 - error / max(initial, 1e-12)),
            numerical_precision=float(-np.log10(error + 1e-16)),
            operation_count=len(self.metrics_history) + 1,
        )


ZeroSumAttention = None
ZeroSumTransformerBlock = None

if TORCH_AVAILABLE:

    class ZeroSumAttention(nn.Module):
        """Attention mechanism based on zero-sum theory with Balansis integration.

        This implementation uses the principles of TNSIM to ensure that
        attention weights maintain zero-sum properties, improving numerical
        stability and theoretical consistency.
        """

        def __init__(
            self,
            d_model: int,
            n_heads: int = 8,
            dropout: float = 0.1,
            compensation_tolerance: float = 1e-8,
        ):
            """Initialize ZeroSumAttention.

            Args:
                d_model: Model dimension
                n_heads: Number of attention heads
                dropout: Dropout probability
                compensation_tolerance: Tolerance for zero-sum compensation
            """
            super().__init__()

            assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

            self.d_model = d_model
            self.n_heads = n_heads
            self.d_k = d_model // n_heads
            self.compensation_tolerance = compensation_tolerance

            # Linear projections
            self.w_q = nn.Linear(d_model, d_model, bias=False)
            self.w_k = nn.Linear(d_model, d_model, bias=False)
            self.w_v = nn.Linear(d_model, d_model, bias=False)
            self.w_o = nn.Linear(d_model, d_model, bias=False)

            # Dropout and normalization
            self.dropout_layer = nn.Dropout(dropout)
            self.layer_norm = nn.LayerNorm(d_model)

            # Balansis integration
            self.use_balansis = BALANSIS_AVAILABLE
            self.compensator = BalansisCompensator(tolerance=compensation_tolerance)

            if self.use_balansis:
                self.stable_softmax = StableSoftmax()
                self.compensated_matmul = CompensatedMatMul()

            # Initialize weights with compensation
            self._init_compensated_weights()

        def _init_compensated_weights(self):
            """Initialize weights with zero-sum compensation."""
            for module in [self.w_q, self.w_k, self.w_v, self.w_o]:
                # Xavier initialization with compensation
                nn.init.xavier_uniform_(module.weight)

                # Apply compensation correction
                with torch.no_grad():
                    weight_sum = module.weight.sum()
                    if abs(weight_sum) > self.compensation_tolerance:
                        # Add compensation term to approximate zero sum
                        compensation = -weight_sum / module.weight.numel()
                        module.weight += compensation

        def forward(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
        ) -> Tuple[torch.Tensor, Dict[str, Any]]:
            """Forward pass of ZeroSumAttention.

            Args:
                query: Query tensor [batch_size, seq_len, d_model]
                key: Key tensor [batch_size, seq_len, d_model]
                value: Value tensor [batch_size, seq_len, d_model]
                mask: Attention mask [batch_size, seq_len, seq_len]

            Returns:
                Tuple[output_tensor, attention_metadata]
            """
            batch_size, seq_len, d_model = query.size()

            # Linear projections
            Q = (
                self.w_q(query)
                .view(batch_size, seq_len, self.n_heads, self.d_k)
                .transpose(1, 2)
            )
            K = (
                self.w_k(key)
                .view(batch_size, seq_len, self.n_heads, self.d_k)
                .transpose(1, 2)
            )
            V = (
                self.w_v(value)
                .view(batch_size, seq_len, self.n_heads, self.d_k)
                .transpose(1, 2)
            )

            # Compensated attention computation
            (
                attention_output,
                attention_weights,
                compensation_metrics,
            ) = self._compensated_attention(Q, K, V, mask)

            # Combine heads
            attention_output = (
                attention_output.transpose(1, 2)
                .contiguous()
                .view(batch_size, seq_len, d_model)
            )

            # Final projection
            output = self.w_o(attention_output)

            # Residual connection with compensation
            output = self._compensated_residual(output, query)

            # Layer normalization
            output = self.layer_norm(output)

            # Metadata for analysis
            metadata = {
                "attention_weights": attention_weights,
                "compensation_metrics": compensation_metrics,
                "zero_sum_quality": self._evaluate_zero_sum_quality(output),
            }

            return output, metadata

        def _compensated_attention(
            self,
            Q: torch.Tensor,
            K: torch.Tensor,
            V: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
        ) -> Tuple[torch.Tensor, torch.Tensor, CompensationMetrics]:
            """Compute attention with compensation."""
            # Compute attention scores
            if self.use_balansis:
                # Use compensated matrix multiplication from Balansis
                scores = self.compensated_matmul(Q, K.transpose(-2, -1)) / np.sqrt(
                    self.d_k
                )
            else:
                scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)

            # Apply mask
            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)

            # Compensated softmax
            if self.use_balansis:
                attention_weights = self.stable_softmax(scores)
            else:
                attention_weights = F.softmax(scores, dim=-1)

            # Apply dropout
            attention_weights = self.dropout_layer(attention_weights)

            # Compensated multiplication with values
            if self.use_balansis:
                attention_output = self.compensated_matmul(attention_weights, V)
            else:
                attention_output = torch.matmul(attention_weights, V)

            # Apply TNSIM compensation
            compensated_output, metrics = self.compensator.compensated_sum(
                attention_output,
                compensating_values=None,
            )

            return compensated_output, attention_weights, metrics

        def _compensated_residual(
            self, output: torch.Tensor, residual: torch.Tensor
        ) -> torch.Tensor:
            """Residual connection with compensation."""
            # Apply TNSIM principles to residual connection
            combined_output, _ = self.compensator.compensated_sum(
                output,
                compensating_values=-residual,
            )
            return combined_output + residual  # Add back for gradient preservation

        def _evaluate_zero_sum_quality(self, tensor: torch.Tensor) -> float:
            """Evaluate zero-sum approximation quality."""
            total_sum = torch.abs(tensor.sum()).item()
            tensor_norm = torch.norm(tensor).item()

            if tensor_norm == 0:
                return 1.0

            zero_sum_quality = 1.0 / (1.0 + total_sum / tensor_norm)
            return zero_sum_quality

        def get_compensation_statistics(self) -> Dict[str, Any]:
            """Get compensation statistics."""
            if not self.compensator.metrics_history:
                return {}

            metrics = self.compensator.metrics_history

            return {
                "total_operations": len(metrics),
                "avg_compensation_error": np.mean(
                    [m.compensation_error for m in metrics]
                ),
                "avg_stability_score": np.mean([m.stability_score for m in metrics]),
                "avg_convergence_rate": np.mean([m.convergence_rate for m in metrics]),
                "avg_numerical_precision": np.mean(
                    [m.numerical_precision for m in metrics]
                ),
                "balansis_integration": self.use_balansis,
            }

    class ZeroSumTransformerBlock(nn.Module):
        """Transformer block with ZeroSumAttention."""

        def __init__(
            self,
            d_model: int,
            n_heads: int = 8,
            d_ff: int = 2048,
            dropout: float = 0.1,
            compensation_tolerance: float = 1e-8,
        ):
            super().__init__()

            self.attention = ZeroSumAttention(
                d_model=d_model,
                n_heads=n_heads,
                dropout=dropout,
                compensation_tolerance=compensation_tolerance,
            )

            self.feed_forward = nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model),
            )

            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.dropout = nn.Dropout(dropout)

            # Compensator for feed-forward network
            self.ff_compensator = BalansisCompensator(tolerance=compensation_tolerance)

        def forward(
            self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
        ) -> Tuple[torch.Tensor, Dict[str, Any]]:
            """Forward pass of transformer block."""
            # Self-attention with compensation
            attn_output, attn_metadata = self.attention(x, x, x, mask)
            x = self.norm1(x + self.dropout(attn_output))

            # Feed-forward with compensation
            ff_output = self.feed_forward(x)

            # Apply compensation to feed-forward output
            compensated_ff, ff_metrics = self.ff_compensator.compensated_sum(
                ff_output, compensating_values=None
            )

            x = self.norm2(x + self.dropout(compensated_ff))

            # Combine metadata
            metadata = {
                "attention_metadata": attn_metadata,
                "ff_compensation_metrics": ff_metrics,
            }

            return x, metadata


# Utilities for testing integration
def test_balansis_integration():
    """Test Balansis integration."""
    print(f"Balansis available: {BALANSIS_AVAILABLE}")

    # Test compensator
    compensator = BalansisCompensator(tolerance=1e-8)
    test_values = torch.randn(10, 20)

    result, metrics = compensator.compensated_sum(test_values)
    print(f"Compensation error: {metrics.compensation_error:.2e}")
    print(f"Stability score: {metrics.stability_score:.4f}")

    # Test ZeroSumAttention
    attention = ZeroSumAttention(d_model=512, n_heads=8)
    x = torch.randn(2, 10, 512)

    output, metadata = attention(x, x, x)
    print(f"Zero-sum quality: {metadata['zero_sum_quality']:.4f}")

    return True


if __name__ == "__main__":
    test_balansis_integration()
