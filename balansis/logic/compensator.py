"""Compensator engine for Balansis library.

This module implements the core compensation engine that maintains mathematical
stability and balance according to Absolute Compensation Theory (ACT) principles.
The Compensator analyzes mathematical operations and applies corrections to
prevent instabilities, overflows, and singularities.
"""

import collections
import math
from typing import List, Dict, Deque, Tuple, Optional, Any, Callable
from enum import Enum
from dataclasses import dataclass
from pydantic import BaseModel, Field, validator

from ..core.absolute import AbsoluteValue
from ..core.eternity import EternalRatio
from ..core.operations import Operations, CompensatedResult


class CompensationType(Enum):
    """Types of compensation that can be applied."""
    STABILITY = "stability"  # Prevents numerical instability
    OVERFLOW = "overflow"    # Prevents arithmetic overflow
    UNDERFLOW = "underflow"  # Prevents arithmetic underflow
    SINGULARITY = "singularity"  # Prevents division by zero
    BALANCE = "balance"      # Maintains mathematical balance
    CONVERGENCE = "convergence"  # Ensures iterative convergence


@dataclass
class CompensationRecord:
    """Record of a compensation operation."""
    operation_type: str
    compensation_type: CompensationType
    original_values: List[Any]
    compensated_values: List[Any]
    compensation_factor: float
    stability_metric: float
    timestamp: float
    
    def __post_init__(self):
        """Validate compensation record after initialization."""
        if self.compensation_factor < 0:
            raise ValueError("Compensation factor must be non-negative")
        if not (0.0 <= self.stability_metric <= 1.0):
            raise ValueError("Stability metric must be between 0 and 1")


class CompensationStrategy(BaseModel):
    """Configuration for compensation strategies."""
    
    stability_threshold: float = Field(
        default=1e-12,
        description="Threshold below which stability compensation is applied"
    )
    overflow_threshold: float = Field(
        default=1e100,
        description="Threshold above which overflow compensation is applied"
    )
    underflow_threshold: float = Field(
        default=1e-100,
        description="Threshold below which underflow compensation is applied"
    )
    max_iterations: int = Field(
        default=100,
        description="Maximum iterations for convergence compensation"
    )
    convergence_tolerance: float = Field(
        default=1e-10,
        description="Tolerance for convergence checks"
    )
    balance_factor: float = Field(
        default=0.5,
        description="Factor for balance compensation (0.0 to 1.0)"
    )
    max_history_size: int = Field(
        default=10000,
        gt=0,
        description="Maximum number of compensation records to keep in history"
    )

    @validator('balance_factor')
    def validate_balance_factor(cls, v: float) -> float:
        """Ensure balance factor is in valid range."""
        if not (0.0 <= v <= 1.0):
            raise ValueError('Balance factor must be between 0.0 and 1.0')
        return v
    
    class Config:
        """Pydantic configuration."""
        validate_assignment = True

    @classmethod
    def high_precision(cls) -> 'CompensationStrategy':
        """Create a high-precision compensation strategy.

        Suitable for scientific computing and financial calculations
        where maximum numerical stability is required.

        Returns:
            CompensationStrategy configured for high precision.
        """
        return cls(
            stability_threshold=1e-15,
            overflow_threshold=1e50,
            underflow_threshold=1e-50,
            max_iterations=1000,
            convergence_tolerance=1e-14,
            balance_factor=0.1,
            max_history_size=50000,
        )

    @classmethod
    def balanced(cls) -> 'CompensationStrategy':
        """Create a balanced compensation strategy.

        Suitable for general-purpose numerical computing with a
        reasonable trade-off between precision and performance.

        Returns:
            CompensationStrategy with balanced configuration.
        """
        return cls(
            stability_threshold=1e-12,
            overflow_threshold=1e100,
            underflow_threshold=1e-100,
            max_iterations=100,
            convergence_tolerance=1e-10,
            balance_factor=0.5,
            max_history_size=10000,
        )

    @classmethod
    def fast(cls) -> 'CompensationStrategy':
        """Create a fast compensation strategy.

        Suitable for real-time systems and ML training loops
        where speed is more important than extreme precision.

        Returns:
            CompensationStrategy configured for performance.
        """
        return cls(
            stability_threshold=1e-8,
            overflow_threshold=1e200,
            underflow_threshold=1e-200,
            max_iterations=10,
            convergence_tolerance=1e-6,
            balance_factor=0.9,
            max_history_size=1000,
        )


class Compensator:
    """Core compensation engine for ACT operations.
    
    The Compensator analyzes mathematical operations and applies appropriate
    compensations to maintain stability, prevent overflows, and ensure
    mathematical balance according to ACT principles.
    
    Attributes:
        strategy: CompensationStrategy configuration
        history: List of compensation records
        active_compensations: Currently active compensations
    
    Examples:
        >>> compensator = Compensator()
        >>> a = AbsoluteValue(magnitude=1e-15, direction=1)
        >>> b = AbsoluteValue(magnitude=1e-15, direction=-1)
        >>> result = compensator.compensate_addition(a, b)
        >>> result.is_absolute()  # True - compensated to prevent instability
    """
    
    def __init__(self, strategy: Optional[CompensationStrategy] = None):
        """Initialize the Compensator.
        
        Args:
            strategy: Compensation strategy configuration
        """
        self.strategy = strategy or CompensationStrategy()
        self.history: Deque[CompensationRecord] = collections.deque(
            maxlen=self.strategy.max_history_size
        )
        self.active_compensations: Dict[str, float] = {}
        self._operation_count = 0
    
    def analyze_stability(self, values: List[AbsoluteValue]) -> float:
        """Analyze the stability of a set of AbsoluteValues.
        
        Args:
            values: List of AbsoluteValues to analyze
            
        Returns:
            Stability metric between 0.0 (unstable) and 1.0 (stable)
        """
        if not values:
            return 1.0
        
        if len(values) == 1:
            value = values[0]
            if value.is_absolute():
                return 0.5  # Absolute is neutral stability
            return min(1.0, value.magnitude / self.strategy.stability_threshold)
        
        # Calculate stability based on magnitude distribution
        magnitudes = [v.magnitude for v in values if not v.is_absolute()]
        if not magnitudes:
            return 0.5  # All Absolute values
        
        # Coefficient of variation as stability metric
        mean_mag = sum(magnitudes) / len(magnitudes)
        if mean_mag == 0:
            return 0.0
        
        variance = sum((m - mean_mag) ** 2 for m in magnitudes) / len(magnitudes)
        cv = math.sqrt(variance) / mean_mag
        
        # Convert CV to stability metric (lower CV = higher stability)
        stability = 1.0 / (1.0 + cv)
        return min(1.0, max(0.0, stability))
    
    def detect_compensation_need(self, operation: str, 
                               operands: List[AbsoluteValue]) -> List[CompensationType]:
        """Detect what types of compensation are needed.
        
        Args:
            operation: Type of mathematical operation
            operands: List of operand AbsoluteValues
            
        Returns:
            List of compensation types needed
        """
        needed_compensations = []
        
        # Check for stability issues
        stability = self.analyze_stability(operands)
        if stability < 0.5:
            needed_compensations.append(CompensationType.STABILITY)
        
        # Check for overflow/underflow
        for operand in operands:
            if not operand.is_absolute():
                if operand.magnitude > self.strategy.overflow_threshold:
                    needed_compensations.append(CompensationType.OVERFLOW)
                elif operand.magnitude < self.strategy.underflow_threshold:
                    needed_compensations.append(CompensationType.UNDERFLOW)
        
        # Check for singularities (operation-specific)
        if operation in ['divide', 'invert'] and any(op.is_absolute() for op in operands[1:]):
            needed_compensations.append(CompensationType.SINGULARITY)
        
        # Check for balance issues in addition/subtraction
        if operation in ['add', 'subtract'] and len(operands) == 2:
            a, b = operands
            if (not a.is_absolute() and not b.is_absolute() and
                abs(a.magnitude - b.magnitude) < self.strategy.stability_threshold and
                a.direction != b.direction):
                needed_compensations.append(CompensationType.BALANCE)
        
        return list(set(needed_compensations))  # Remove duplicates
    
    def apply_stability_compensation(self, values: List[AbsoluteValue]) -> List[AbsoluteValue]:
        """Apply stability compensation to a list of values.
        
        Args:
            values: List of AbsoluteValues to compensate
            
        Returns:
            List of compensated AbsoluteValues
        """
        compensated = []
        
        for value in values:
            if value.is_absolute():
                compensated.append(value)
            elif value.magnitude < self.strategy.stability_threshold:
                # Compensate very small values to Absolute
                compensated.append(AbsoluteValue.absolute())
            else:
                compensated.append(value)
        
        return compensated
    
    def apply_overflow_compensation(self, values: List[AbsoluteValue]) -> List[AbsoluteValue]:
        """Apply overflow compensation to prevent arithmetic overflow.
        
        Args:
            values: List of AbsoluteValues to compensate
            
        Returns:
            List of compensated AbsoluteValues
        """
        compensated = []
        
        for value in values:
            if value.is_absolute():
                compensated.append(value)
            elif value.magnitude > self.strategy.overflow_threshold:
                # Scale down to prevent overflow
                compensated_value = AbsoluteValue(
                    magnitude=self.strategy.overflow_threshold,
                    direction=value.direction
                )
                compensated.append(compensated_value)
            else:
                compensated.append(value)
        
        return compensated
    
    def apply_balance_compensation(self, a: AbsoluteValue, b: AbsoluteValue) -> Tuple[AbsoluteValue, AbsoluteValue]:
        """Apply balance compensation for near-canceling operations.
        
        Args:
            a: First AbsoluteValue
            b: Second AbsoluteValue
            
        Returns:
            Tuple of compensated AbsoluteValues
        """
        if (not a.is_absolute() and not b.is_absolute() and
            abs(a.magnitude - b.magnitude) < self.strategy.stability_threshold and
            a.direction != b.direction):
            
            # Apply balance compensation
            avg_magnitude = (a.magnitude + b.magnitude) / 2
            compensation_factor = self.strategy.balance_factor
            
            compensated_a = AbsoluteValue(
                magnitude=avg_magnitude * (1 + compensation_factor),
                direction=a.direction
            )
            compensated_b = AbsoluteValue(
                magnitude=avg_magnitude * (1 - compensation_factor),
                direction=b.direction
            )
            
            return compensated_a, compensated_b
        
        return a, b
    
    def compensate_addition(self, a: AbsoluteValue, b: AbsoluteValue) -> AbsoluteValue:
        """Perform compensated addition.
        
        Args:
            a: First AbsoluteValue operand
            b: Second AbsoluteValue operand
            
        Returns:
            Compensated addition result
        """
        self._operation_count += 1
        
        # Detect needed compensations
        compensations = self.detect_compensation_need('add', [a, b])
        
        # Apply compensations
        comp_a, comp_b = a, b
        compensation_factor = 1.0
        
        if CompensationType.BALANCE in compensations:
            comp_a, comp_b = self.apply_balance_compensation(a, b)
            compensation_factor *= self.strategy.balance_factor
        
        if CompensationType.STABILITY in compensations:
            values = self.apply_stability_compensation([comp_a, comp_b])
            comp_a, comp_b = values[0], values[1]
        
        if CompensationType.OVERFLOW in compensations:
            values = self.apply_overflow_compensation([comp_a, comp_b])
            comp_a, comp_b = values[0], values[1]
        
        # Perform the operation
        result, op_compensation = Operations.compensated_add(comp_a, comp_b, compensation_factor)
        
        # Record the compensation
        if compensations:
            record = CompensationRecord(
                operation_type='addition',
                compensation_type=compensations[0],  # Primary compensation
                original_values=[a, b],
                compensated_values=[comp_a, comp_b],
                compensation_factor=op_compensation,
                stability_metric=self.analyze_stability([result]),
                timestamp=self._operation_count
            )
            self.history.append(record)
        
        return result
    
    def compensate_multiplication(self, a: AbsoluteValue, b: AbsoluteValue) -> AbsoluteValue:
        """Perform compensated multiplication.
        
        Args:
            a: First AbsoluteValue operand
            b: Second AbsoluteValue operand
            
        Returns:
            Compensated multiplication result
        """
        self._operation_count += 1
        
        # Detect needed compensations
        compensations = self.detect_compensation_need('multiply', [a, b])
        
        # Apply compensations
        comp_a, comp_b = a, b
        
        if CompensationType.OVERFLOW in compensations:
            values = self.apply_overflow_compensation([comp_a, comp_b])
            comp_a, comp_b = values[0], values[1]
        
        if CompensationType.UNDERFLOW in compensations:
            # For underflow, we might want to preserve the operation
            pass  # Let Operations handle underflow compensation
        
        # Perform the operation
        result, compensation_factor = Operations.compensated_multiply(comp_a, comp_b)
        
        # Record the compensation
        if compensations:
            record = CompensationRecord(
                operation_type='multiplication',
                compensation_type=compensations[0],
                original_values=[a, b],
                compensated_values=[comp_a, comp_b],
                compensation_factor=compensation_factor,
                stability_metric=self.analyze_stability([result]),
                timestamp=self._operation_count
            )
            self.history.append(record)
        
        return result
    
    def compensate_division(self, numerator: AbsoluteValue, 
                          denominator: AbsoluteValue) -> EternalRatio:
        """Perform compensated division using EternalRatio.
        
        Args:
            numerator: AbsoluteValue to divide
            denominator: AbsoluteValue to divide by
            
        Returns:
            EternalRatio representing the compensated division
        """
        self._operation_count += 1
        
        # Detect needed compensations
        compensations = self.detect_compensation_need('divide', [numerator, denominator])
        
        # Handle singularity compensation
        if CompensationType.SINGULARITY in compensations:
            if denominator.is_absolute():
                # Division by Absolute - return special EternalRatio
                # Use a very small denominator instead
                comp_denominator = AbsoluteValue(
                    magnitude=self.strategy.stability_threshold,
                    direction=1
                )
            else:
                comp_denominator = denominator
        else:
            comp_denominator = denominator
        
        # Perform the operation
        result, op_compensation = Operations.compensated_divide(numerator, comp_denominator)

        # Record the compensation
        if compensations:
            record = CompensationRecord(
                operation_type='division',
                compensation_type=compensations[0],
                original_values=[numerator, denominator],
                compensated_values=[numerator, comp_denominator],
                compensation_factor=op_compensation,
                stability_metric=1.0 if result.is_stable() else 0.0,
                timestamp=self._operation_count
            )
            self.history.append(record)
        
        return result
    
    def compensate_power(self, base: AbsoluteValue, exponent: float) -> AbsoluteValue:
        """Perform compensated exponentiation.
        
        Args:
            base: AbsoluteValue base
            exponent: Power to raise base to
            
        Returns:
            Compensated power result
        """
        self._operation_count += 1
        
        # Detect needed compensations
        compensations = self.detect_compensation_need('power', [base])
        
        # Apply compensations
        comp_base = base
        
        if CompensationType.OVERFLOW in compensations:
            values = self.apply_overflow_compensation([base])
            comp_base = values[0]
        
        # Perform the operation
        result, compensation_factor = Operations.compensated_power(comp_base, exponent)
        
        # Record the compensation
        if compensations:
            record = CompensationRecord(
                operation_type='power',
                compensation_type=compensations[0],
                original_values=[base],
                compensated_values=[comp_base],
                compensation_factor=compensation_factor,
                stability_metric=self.analyze_stability([result]),
                timestamp=self._operation_count
            )
            self.history.append(record)
        
        return result
    
    def compensate_sequence(self, operation: Callable, 
                          values: List[AbsoluteValue]) -> AbsoluteValue:
        """Apply compensation to a sequence of operations.
        
        Args:
            operation: Binary operation function
            values: List of AbsoluteValues to process
            
        Returns:
            Compensated result of the sequence
        """
        if not values:
            return AbsoluteValue.absolute()
        
        if len(values) == 1:
            return values[0]
        
        # Apply pairwise operations with compensation
        result = values[0]
        for value in values[1:]:
            if operation == Operations.compensated_add:
                result = self.compensate_addition(result, value)
            elif operation == Operations.compensated_multiply:
                result = self.compensate_multiplication(result, value)
            else:
                # Generic operation - apply basic compensation
                compensations = self.detect_compensation_need('generic', [result, value])
                if compensations:
                    comp_values = self.apply_stability_compensation([result, value])
                    result, value = comp_values[0], comp_values[1]
                
                # Apply the operation (assuming it returns CompensatedResult)
                op_result, _ = operation(result, value)
                result = op_result
        
        return result
    
    def get_compensation_summary(self) -> Dict[str, Any]:
        """Get a summary of all compensations applied.
        
        Returns:
            Dictionary containing compensation statistics
        """
        if not self.history:
            return {
                'total_operations': self._operation_count,
                'compensated_operations': 0,
                'compensation_rate': 0.0,
                'compensation_types': {},
                'average_stability': 1.0
            }
        
        compensation_types = {}
        total_stability = 0.0
        
        for record in self.history:
            comp_type = record.compensation_type.value
            compensation_types[comp_type] = compensation_types.get(comp_type, 0) + 1
            total_stability += record.stability_metric
        
        return {
            'total_operations': self._operation_count,
            'compensated_operations': len(self.history),
            'compensation_rate': len(self.history) / max(1, self._operation_count),
            'compensation_types': compensation_types,
            'average_stability': total_stability / len(self.history),
            'latest_compensations': list(self.history)[-5:]
        }
    
    def reset_history(self) -> None:
        """Reset the compensation history."""
        self.history.clear()
        self.active_compensations.clear()
        self._operation_count = 0
    
    def set_strategy(self, strategy: CompensationStrategy) -> None:
        """Update the compensation strategy.

        Args:
            strategy: New compensation strategy.
        """
        self.strategy = strategy
        # Resize history deque if max_history_size changed
        if self.history.maxlen != strategy.max_history_size:
            self.history = collections.deque(
                self.history, maxlen=strategy.max_history_size
            )
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return (f"Compensator(operations={self._operation_count}, "
                f"compensations={len(self.history)}, "
                f"rate={len(self.history)/max(1, self._operation_count):.3f})")
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        summary = self.get_compensation_summary()
        return (f"Compensator: {summary['compensated_operations']}/{summary['total_operations']} "
                f"operations compensated ({summary['compensation_rate']:.1%} rate), "
                f"avg stability: {summary['average_stability']:.3f}")

    def compensate_array(self, arr_a: List[AbsoluteValue], arr_b: List[AbsoluteValue]) -> List[AbsoluteValue]:
        out: List[AbsoluteValue] = []
        n = min(len(arr_a), len(arr_b))
        for i in range(n):
            out.append(self.compensate_addition(arr_a[i], arr_b[i]))
        for j in range(n, len(arr_a)):
            out.append(arr_a[j])
        for j in range(n, len(arr_b)):
            out.append(arr_b[j])
        return [v for v in out if not v.is_absolute()]
