"""Plotting utilities for Balansis library.

This module provides visualization capabilities for AbsoluteValue, EternalRatio,
and other mathematical structures in the Balansis library. It supports both
static plots (Matplotlib) and interactive visualizations (Plotly).
"""

from typing import List, Tuple, Optional, Dict, Any, Union, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go
import math
from enum import Enum
from pydantic import BaseModel, Field, field_validator

# Conditional imports for plotting dependencies
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.animation import FuncAnimation
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None
    patches = None
    LinearSegmentedColormap = None
    FuncAnimation = None

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.figure_factory as ff
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    go = None
    px = None
    make_subplots = None
    ff = None

from ..core.absolute import AbsoluteValue
from ..core.eternity import EternalRatio
from ..core.operations import Operations
from ..algebra.absolute_group import AbsoluteGroup, GroupElement
from ..algebra.eternity_field import EternityField, FieldElement
from ..logic.compensator import Compensator, CompensationRecord


class PlotStyle(str, Enum):
    """Available plot styles."""
    SCIENTIFIC = "scientific"
    ELEGANT = "elegant"
    MINIMAL = "minimal"
    COLORFUL = "colorful"


class PlotBackend(str, Enum):
    """Available plotting backends."""
    MATPLOTLIB = "matplotlib"
    PLOTLY = "plotly"


class PlotConfig(BaseModel):
    """Configuration for plot appearance and behavior.
    
    Attributes:
        style: Plot style theme
        backend: Plotting backend to use
        width: Figure width in pixels
        height: Figure height in pixels
        dpi: Resolution for matplotlib plots
        color_palette: Color palette for plots
        font_size: Base font size
        line_width: Default line width
        marker_size: Default marker size
        alpha: Default transparency
        grid: Whether to show grid
        legend: Whether to show legend
        title_size: Title font size
        axis_label_size: Axis label font size
        interactive: Whether to enable interactive features
    """
    
    style: PlotStyle = Field(default=PlotStyle.SCIENTIFIC, description="Plot style theme")
    backend: PlotBackend = Field(default=PlotBackend.MATPLOTLIB, description="Plotting backend")
    width: int = Field(default=800, description="Figure width in pixels")
    height: int = Field(default=600, description="Figure height in pixels")
    dpi: int = Field(default=100, description="Resolution for matplotlib")
    color_palette: List[str] = Field(
        default=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"],
        description="Color palette for plots"
    )
    font_size: int = Field(default=12, description="Base font size")
    line_width: float = Field(default=2.0, description="Default line width")
    marker_size: float = Field(default=6.0, description="Default marker size")
    alpha: float = Field(default=0.8, description="Default transparency")
    grid: bool = Field(default=True, description="Show grid")
    legend: bool = Field(default=True, description="Show legend")
    title_size: int = Field(default=14, description="Title font size")
    axis_label_size: int = Field(default=14, description="Axis label font size")
    interactive: bool = Field(default=False, description="Enable interactive features")
    save_format: str = Field(default="png", description="Default save format for plots")
    animation_duration: int = Field(default=1000, description="Animation duration in milliseconds")
    animation_frames: int = Field(default=50, description="Number of animation frames")
    
    @field_validator('width')
    @classmethod
    def validate_width(cls, v: int) -> int:
        """Ensure width is positive."""
        if v <= 0:
            raise ValueError('Width must be positive')
        return v
    
    @field_validator('height')
    @classmethod
    def validate_height(cls, v: int) -> int:
        """Ensure height is positive."""
        if v <= 0:
            raise ValueError('Height must be positive')
        return v
    
    @field_validator('dpi')
    @classmethod
    def validate_dpi(cls, v: int) -> int:
        """Ensure dpi is positive."""
        if v <= 0:
            raise ValueError('DPI must be positive')
        return v
    
    @field_validator('alpha')
    @classmethod
    def validate_alpha(cls, v: float) -> float:
        """Ensure alpha is between 0 and 1."""
        if not 0 <= v <= 1:
            raise ValueError('Alpha must be between 0 and 1')
        return v
    
    class Config:
        """Pydantic configuration."""
        validate_assignment = True


class PlotUtils:
    """Utility class for plotting mathematical structures.
    
    Provides comprehensive visualization capabilities for AbsoluteValue,
    EternalRatio, and other Balansis mathematical objects.
    
    Attributes:
        config: Plot configuration settings
        operations: Operations instance for calculations
        compensator: Compensator for stable computations
    
    Examples:
        >>> plotter = PlotUtils()
        >>> values = [AbsoluteValue(i, 1) for i in range(-5, 6)]
        >>> plotter.plot_absolute_values(values, title="AbsoluteValue Distribution")
        >>> 
        >>> ratios = [EternalRatio(AbsoluteValue(i, 1), AbsoluteValue(2, 1)) for i in range(1, 6)]
        >>> plotter.plot_eternal_ratios(ratios, title="EternalRatio Sequence")
    """
    
    def __init__(self, config: Optional[PlotConfig] = None,
                 operations: Optional[Operations] = None,
                 compensator: Optional[Compensator] = None):
        """Initialize PlotUtils.
        
        Args:
            config: Plot configuration settings
            operations: Operations instance for calculations
            compensator: Compensator for stable computations
        """
        self.config = config or PlotConfig()
        self.operations = operations or Operations()
        self.compensator = compensator or Compensator()
        
        # Validate dependencies
        self._validate_dependencies()
        
        # Set up matplotlib style
        if self.config.backend == PlotBackend.MATPLOTLIB and MATPLOTLIB_AVAILABLE:
            self._setup_matplotlib_style()
    
    def _validate_dependencies(self) -> None:
        """Validate that required plotting dependencies are available."""
        if not NUMPY_AVAILABLE:
            raise ImportError("NumPy is required for plotting functionality. Install with: pip install numpy")
        
        if self.config.backend == PlotBackend.MATPLOTLIB and not MATPLOTLIB_AVAILABLE:
            raise ImportError("Matplotlib is required for matplotlib backend. Install with: pip install matplotlib")
        
        if self.config.backend == PlotBackend.PLOTLY and not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is required for plotly backend. Install with: pip install plotly")
    
    def _setup_matplotlib_style(self) -> None:
        """Configure matplotlib style based on config."""
        if not MATPLOTLIB_AVAILABLE or plt is None:
            return
            
        plt.rcParams.update({
            'figure.figsize': (self.config.width / 100, self.config.height / 100),
            'figure.dpi': self.config.dpi,
            'font.size': self.config.font_size,
            'axes.titlesize': self.config.title_size,
            'axes.labelsize': self.config.axis_label_size,
            'lines.linewidth': self.config.line_width,
            'lines.markersize': self.config.marker_size,
            'axes.grid': self.config.grid,
            'legend.fontsize': self.config.font_size - 2,
        })
        
        # Apply style theme with fallback for missing styles
        try:
            if self.config.style == PlotStyle.SCIENTIFIC:
                plt.style.use('seaborn-v0_8-whitegrid')
            elif self.config.style == PlotStyle.ELEGANT:
                plt.style.use('seaborn-v0_8-darkgrid')
            elif self.config.style == PlotStyle.MINIMAL:
                plt.style.use('seaborn-v0_8-white')
            elif self.config.style == PlotStyle.COLORFUL:
                plt.style.use('seaborn-v0_8-bright')
        except OSError:
            # Fallback to default style if seaborn styles are not available
            plt.style.use('default')
    
    def plot_absolute_values(self, values: List[AbsoluteValue],
                           title: str = "AbsoluteValue Distribution",
                           xlabel: str = "Index",
                           ylabel: str = "Magnitude",
                           show_directions: bool = True,
                           save_path: Optional[str] = None) -> Any:
        """Plot a sequence of AbsoluteValue objects.
        
        Args:
            values: List of AbsoluteValue objects to plot
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            show_directions: Whether to show direction indicators
            save_path: Path to save the plot
            
        Returns:
            Figure object (matplotlib or plotly)
        """
        if self.config.backend == PlotBackend.MATPLOTLIB:
            return self._plot_absolute_values_matplotlib(
                values, title, xlabel, ylabel, show_directions, save_path
            )
        else:
            return self._plot_absolute_values_plotly(
                values, title, xlabel, ylabel, show_directions, save_path
            )
    
    def _plot_absolute_values_matplotlib(self, values: List[AbsoluteValue],
                                       title: str, xlabel: str, ylabel: str,
                                       show_directions: bool, save_path: Optional[str]) -> Any:
        """Plot AbsoluteValues using matplotlib."""
        fig, ax = plt.subplots(figsize=(self.config.width / 100, self.config.height / 100))
        
        # Extract data
        indices = list(range(len(values)))
        magnitudes = [v.magnitude for v in values]
        directions = [v.direction for v in values]
        
        # Create scatter plot with direction-based colors
        colors = [self.config.color_palette[0] if d > 0 else self.config.color_palette[1] for d in directions]
        
        scatter = ax.scatter(indices, magnitudes, c=colors, s=self.config.marker_size**2,
                           alpha=self.config.alpha, edgecolors='black', linewidth=0.5)
        
        # Add direction indicators if requested
        if show_directions:
            for i, (mag, direction) in enumerate(zip(magnitudes, directions)):
                if direction > 0:
                    ax.annotate('↑', (i, mag), xytext=(0, 10), textcoords='offset points',
                              ha='center', va='bottom', fontsize=self.config.font_size)
                elif direction < 0:
                    ax.annotate('↓', (i, mag), xytext=(0, -15), textcoords='offset points',
                              ha='center', va='top', fontsize=self.config.font_size)
                else:
                    ax.annotate('○', (i, mag), xytext=(0, 0), textcoords='offset points',
                              ha='center', va='center', fontsize=self.config.font_size)
        
        # Customize plot
        ax.set_title(title, fontsize=self.config.title_size, pad=20)
        ax.set_xlabel(xlabel, fontsize=self.config.axis_label_size)
        ax.set_ylabel(ylabel, fontsize=self.config.axis_label_size)
        
        if self.config.legend:
            # Create custom legend
            positive_patch = patches.Patch(color=self.config.color_palette[0], label='Positive Direction')
            negative_patch = patches.Patch(color=self.config.color_palette[1], label='Negative Direction')
            ax.legend(handles=[positive_patch, negative_patch], loc='upper right')
        
        if self.config.grid:
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        
        return fig
    
    def _plot_absolute_values_plotly(self, values: List[AbsoluteValue],
                                   title: str, xlabel: str, ylabel: str,
                                   show_directions: bool, save_path: Optional[str]) -> Any:
        """Plot AbsoluteValues using plotly."""
        # Extract data
        indices = list(range(len(values)))
        magnitudes = [v.magnitude for v in values]
        directions = [v.direction for v in values]
        
        # Create traces for different directions
        positive_indices = [i for i, d in enumerate(directions) if d > 0]
        negative_indices = [i for i, d in enumerate(directions) if d < 0]
        absolute_indices = [i for i, d in enumerate(directions) if d == 0]
        
        fig = go.Figure()
        
        # Add positive direction points
        if positive_indices:
            fig.add_trace(go.Scatter(
                x=[indices[i] for i in positive_indices],
                y=[magnitudes[i] for i in positive_indices],
                mode='markers',
                name='Positive Direction',
                marker=dict(
                    color=self.config.color_palette[0],
                    size=self.config.marker_size,
                    opacity=self.config.alpha,
                    line=dict(width=1, color='black')
                )
            ))
        
        # Add negative direction points
        if negative_indices:
            fig.add_trace(go.Scatter(
                x=[indices[i] for i in negative_indices],
                y=[magnitudes[i] for i in negative_indices],
                mode='markers',
                name='Negative Direction',
                marker=dict(
                    color=self.config.color_palette[1],
                    size=self.config.marker_size,
                    opacity=self.config.alpha,
                    line=dict(width=1, color='black')
                )
            ))
        
        # Add absolute (zero direction) points
        if absolute_indices:
            fig.add_trace(go.Scatter(
                x=[indices[i] for i in absolute_indices],
                y=[magnitudes[i] for i in absolute_indices],
                mode='markers',
                name='Absolute (Zero Direction)',
                marker=dict(
                    color=self.config.color_palette[2],
                    size=self.config.marker_size,
                    opacity=self.config.alpha,
                    line=dict(width=1, color='black'),
                    symbol='circle-open'
                )
            ))
        
        # Update layout
        fig.update_layout(
            title=dict(text=title, font=dict(size=self.config.title_size)),
            xaxis_title=dict(text=xlabel, font=dict(size=self.config.axis_label_size)),
            yaxis_title=dict(text=ylabel, font=dict(size=self.config.axis_label_size)),
            showlegend=self.config.legend,
            width=self.config.width,
            height=self.config.height,
            font=dict(size=self.config.font_size)
        )
        
        if self.config.grid:
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def plot_eternal_ratios(self, ratios: List[EternalRatio],
                          title: str = "EternalRatio Sequence",
                          xlabel: str = "Index",
                          ylabel: str = "Ratio Value",
                          show_stability: bool = True,
                          save_path: Optional[str] = None) -> Any:
        """Plot a sequence of EternalRatio objects.
        
        Args:
            ratios: List of EternalRatio objects to plot
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            show_stability: Whether to highlight stable ratios
            save_path: Path to save the plot
            
        Returns:
            Figure object (matplotlib or plotly)
        """
        if self.config.backend == PlotBackend.MATPLOTLIB:
            return self._plot_eternal_ratios_matplotlib(
                ratios, title, xlabel, ylabel, show_stability, save_path
            )
        else:
            return self._plot_eternal_ratios_plotly(
                ratios, title, xlabel, ylabel, show_stability, save_path
            )
    
    def _plot_eternal_ratios_matplotlib(self, ratios: List[EternalRatio],
                                      title: str, xlabel: str, ylabel: str,
                                      show_stability: bool, save_path: Optional[str]) -> Any:
        """Plot EternalRatios using matplotlib."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(self.config.width / 100, self.config.height / 100 * 1.2))
        
        # Extract data
        indices = list(range(len(ratios)))
        numerical_values = [r.numerical_value() for r in ratios]
        signed_values = [r.signed_value() for r in ratios]
        stability = [r.is_stable() for r in ratios]
        
        # Plot numerical values
        colors = [self.config.color_palette[0] if stable else self.config.color_palette[3] for stable in stability]
        
        ax1.scatter(indices, numerical_values, c=colors, s=self.config.marker_size**2,
                   alpha=self.config.alpha, edgecolors='black', linewidth=0.5)
        ax1.plot(indices, numerical_values, color=self.config.color_palette[0],
                alpha=0.5, linewidth=self.config.line_width/2)
        
        ax1.set_title(f"{title} - Numerical Values", fontsize=self.config.title_size)
        ax1.set_xlabel(xlabel, fontsize=self.config.axis_label_size)
        ax1.set_ylabel("Numerical Value", fontsize=self.config.axis_label_size)
        
        if show_stability and self.config.legend:
            stable_patch = patches.Patch(color=self.config.color_palette[0], label='Stable')
            unstable_patch = patches.Patch(color=self.config.color_palette[3], label='Unstable')
            ax1.legend(handles=[stable_patch, unstable_patch], loc='upper right')
        
        # Plot signed values
        ax2.scatter(indices, signed_values, c=colors, s=self.config.marker_size**2,
                   alpha=self.config.alpha, edgecolors='black', linewidth=0.5)
        ax2.plot(indices, signed_values, color=self.config.color_palette[1],
                alpha=0.5, linewidth=self.config.line_width/2)
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        ax2.set_title(f"{title} - Signed Values", fontsize=self.config.title_size)
        ax2.set_xlabel(xlabel, fontsize=self.config.axis_label_size)
        ax2.set_ylabel("Signed Value", fontsize=self.config.axis_label_size)
        
        if self.config.grid:
            ax1.grid(True, alpha=0.3)
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        
        return fig
    
    def _plot_eternal_ratios_plotly(self, ratios: List[EternalRatio],
                                  title: str, xlabel: str, ylabel: str,
                                  show_stability: bool, save_path: Optional[str]) -> Any:
        """Plot EternalRatios using plotly."""
        # Extract data
        indices = list(range(len(ratios)))
        numerical_values = [r.numerical_value() for r in ratios]
        signed_values = [r.signed_value() for r in ratios]
        stability = [r.is_stable() for r in ratios]
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=[f"{title} - Numerical Values", f"{title} - Signed Values"],
            vertical_spacing=0.1
        )
        
        # Separate stable and unstable ratios
        stable_indices = [i for i, stable in enumerate(stability) if stable]
        unstable_indices = [i for i, stable in enumerate(stability) if not stable]
        
        # Add numerical values - stable
        if stable_indices:
            fig.add_trace(go.Scatter(
                x=[indices[i] for i in stable_indices],
                y=[numerical_values[i] for i in stable_indices],
                mode='markers+lines',
                name='Stable (Numerical)',
                marker=dict(
                    color=self.config.color_palette[0],
                    size=self.config.marker_size,
                    opacity=self.config.alpha
                ),
                line=dict(color=self.config.color_palette[0], width=self.config.line_width/2)
            ), row=1, col=1)
        
        # Add numerical values - unstable
        if unstable_indices:
            fig.add_trace(go.Scatter(
                x=[indices[i] for i in unstable_indices],
                y=[numerical_values[i] for i in unstable_indices],
                mode='markers',
                name='Unstable (Numerical)',
                marker=dict(
                    color=self.config.color_palette[3],
                    size=self.config.marker_size,
                    opacity=self.config.alpha
                )
            ), row=1, col=1)
        
        # Add signed values - stable
        if stable_indices:
            fig.add_trace(go.Scatter(
                x=[indices[i] for i in stable_indices],
                y=[signed_values[i] for i in stable_indices],
                mode='markers+lines',
                name='Stable (Signed)',
                marker=dict(
                    color=self.config.color_palette[1],
                    size=self.config.marker_size,
                    opacity=self.config.alpha
                ),
                line=dict(color=self.config.color_palette[1], width=self.config.line_width/2)
            ), row=2, col=1)
        
        # Add signed values - unstable
        if unstable_indices:
            fig.add_trace(go.Scatter(
                x=[indices[i] for i in unstable_indices],
                y=[signed_values[i] for i in unstable_indices],
                mode='markers',
                name='Unstable (Signed)',
                marker=dict(
                    color=self.config.color_palette[3],
                    size=self.config.marker_size,
                    opacity=self.config.alpha
                )
            ), row=2, col=1)
        
        # Add zero line for signed values
        fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5, row=2, col=1)
        
        # Update layout
        fig.update_layout(
            title=dict(text=title, font=dict(size=self.config.title_size)),
            showlegend=self.config.legend and show_stability,
            width=self.config.width,
            height=self.config.height * 1.2,
            font=dict(size=self.config.font_size)
        )
        
        fig.update_xaxes(title_text=xlabel, title_font=dict(size=self.config.axis_label_size))
        fig.update_yaxes(title_text="Numerical Value", title_font=dict(size=self.config.axis_label_size), row=1, col=1)
        fig.update_yaxes(title_text="Signed Value", title_font=dict(size=self.config.axis_label_size), row=2, col=1)
        
        if self.config.grid:
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def plot_compensation_analysis(self, records: List[CompensationRecord],
                                 title: str = "Compensation Analysis",
                                 save_path: Optional[str] = None) -> Any:
        """Plot compensation analysis from CompensationRecord objects.
        
        Args:
            records: List of CompensationRecord objects
            title: Plot title
            save_path: Path to save the plot
            
        Returns:
            Figure object (matplotlib or plotly)
        """
        if not records:
            raise ValueError("No compensation records provided")
        
        if self.config.backend == PlotBackend.MATPLOTLIB:
            return self._plot_compensation_matplotlib(records, title, save_path)
        else:
            return self._plot_compensation_plotly(records, title, save_path)
    
    def _plot_compensation_matplotlib(self, records: List[CompensationRecord],
                                    title: str, save_path: Optional[str]) -> Any:
        """Plot compensation analysis using matplotlib."""
        fig, axes = plt.subplots(2, 2, figsize=(self.config.width / 100 * 1.5, self.config.height / 100 * 1.2))
        ax1, ax2 = axes[0]
        ax3, ax4 = axes[1]
        
        # Extract data
        compensation_types = [record.compensation_type.value for record in records]
        original_values = [record.original_values[0].magnitude if record.original_values else 0 for record in records]
        compensated_values = [record.compensated_values[0].magnitude if record.compensated_values else 0 for record in records]
        factors = [record.compensation_factor for record in records]
        
        # Plot 1: Compensation types distribution
        type_counts = {}
        for comp_type in compensation_types:
            type_counts[comp_type] = type_counts.get(comp_type, 0) + 1
        
        ax1.bar(type_counts.keys(), type_counts.values(), 
               color=self.config.color_palette[:len(type_counts)], alpha=self.config.alpha)
        ax1.set_title("Compensation Types Distribution", fontsize=self.config.title_size)
        ax1.set_ylabel("Count", fontsize=self.config.axis_label_size)
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot 2: Original vs Compensated values
        ax2.scatter(original_values, compensated_values, 
                   color=self.config.color_palette[0], alpha=self.config.alpha,
                   s=self.config.marker_size**2)
        ax2.plot([min(original_values), max(original_values)], 
                [min(original_values), max(original_values)], 
                'r--', alpha=0.5, label='y=x')
        ax2.set_title("Original vs Compensated Values", fontsize=self.config.title_size)
        ax2.set_xlabel("Original Value", fontsize=self.config.axis_label_size)
        ax2.set_ylabel("Compensated Value", fontsize=self.config.axis_label_size)
        if self.config.legend:
            ax2.legend()
        
        # Plot 3: Compensation factors over time
        ax3.plot(range(len(factors)), factors, 
                color=self.config.color_palette[1], linewidth=self.config.line_width,
                marker='o', markersize=self.config.marker_size/2, alpha=self.config.alpha)
        ax3.set_title("Compensation Factors Over Time", fontsize=self.config.title_size)
        ax3.set_xlabel("Record Index", fontsize=self.config.axis_label_size)
        ax3.set_ylabel("Compensation Factor", fontsize=self.config.axis_label_size)
        
        # Plot 4: Compensation effectiveness
        effectiveness = [abs(comp - orig) / max(abs(orig), 1e-10) 
                        for orig, comp in zip(original_values, compensated_values)]
        ax4.hist(effectiveness, bins=20, color=self.config.color_palette[2], 
                alpha=self.config.alpha, edgecolor='black')
        ax4.set_title("Compensation Effectiveness Distribution", fontsize=self.config.title_size)
        ax4.set_xlabel("Relative Change", fontsize=self.config.axis_label_size)
        ax4.set_ylabel("Frequency", fontsize=self.config.axis_label_size)
        
        if self.config.grid:
            for ax in [ax1, ax2, ax3, ax4]:
                ax.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=self.config.title_size + 2)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        
        return fig
    
    def _plot_compensation_plotly(self, records: List[CompensationRecord],
                                title: str, save_path: Optional[str]) -> Any:
        """Plot compensation analysis using plotly."""
        # Extract data
        compensation_types = [record.compensation_type.value for record in records]
        original_values = [record.original_values[0].magnitude if record.original_values else 0 for record in records]
        compensated_values = [record.compensated_values[0].magnitude if record.compensated_values else 0 for record in records]
        factors = [record.compensation_factor for record in records]
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "Compensation Types Distribution",
                "Original vs Compensated Values",
                "Compensation Factors Over Time",
                "Compensation Effectiveness Distribution"
            ],
            specs=[[{"type": "bar"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "histogram"}]]
        )
        
        # Plot 1: Compensation types distribution
        type_counts = {}
        for comp_type in compensation_types:
            type_counts[comp_type] = type_counts.get(comp_type, 0) + 1
        
        fig.add_trace(go.Bar(
            x=list(type_counts.keys()),
            y=list(type_counts.values()),
            marker_color=self.config.color_palette[0],
            opacity=self.config.alpha,
            name="Type Counts"
        ), row=1, col=1)
        
        # Plot 2: Original vs Compensated values
        fig.add_trace(go.Scatter(
            x=original_values,
            y=compensated_values,
            mode='markers',
            marker=dict(
                color=self.config.color_palette[1],
                size=self.config.marker_size,
                opacity=self.config.alpha
            ),
            name="Data Points"
        ), row=1, col=2)
        
        # Add y=x line
        min_val, max_val = min(original_values), max(original_values)
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            line=dict(color='red', dash='dash'),
            name="y=x"
        ), row=1, col=2)
        
        # Plot 3: Compensation factors over time
        fig.add_trace(go.Scatter(
            x=list(range(len(factors))),
            y=factors,
            mode='lines+markers',
            line=dict(color=self.config.color_palette[2], width=self.config.line_width),
            marker=dict(size=self.config.marker_size/2),
            name="Factors"
        ), row=2, col=1)
        
        # Plot 4: Compensation effectiveness
        effectiveness = [abs(comp - orig) / max(abs(orig), 1e-10) 
                        for orig, comp in zip(original_values, compensated_values)]
        fig.add_trace(go.Histogram(
            x=effectiveness,
            nbinsx=20,
            marker_color=self.config.color_palette[3],
            opacity=self.config.alpha,
            name="Effectiveness"
        ), row=2, col=2)
        
        # Update layout
        fig.update_layout(
            title=dict(text=title, font=dict(size=self.config.title_size)),
            showlegend=False,
            width=self.config.width * 1.5,
            height=self.config.height * 1.2,
            font=dict(size=self.config.font_size)
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Compensation Type", row=1, col=1)
        fig.update_yaxes(title_text="Count", row=1, col=1)
        fig.update_xaxes(title_text="Original Value", row=1, col=2)
        fig.update_yaxes(title_text="Compensated Value", row=1, col=2)
        fig.update_xaxes(title_text="Record Index", row=2, col=1)
        fig.update_yaxes(title_text="Compensation Factor", row=2, col=1)
        fig.update_xaxes(title_text="Relative Change", row=2, col=2)
        fig.update_yaxes(title_text="Frequency", row=2, col=2)
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def plot_act_phase_space(self, values: List[AbsoluteValue],
                           title: str = "ACT Phase Space",
                           save_path: Optional[str] = None) -> Any:
        """Plot AbsoluteValue objects in ACT phase space (magnitude vs direction).
        
        Args:
            values: List of AbsoluteValue objects
            title: Plot title
            save_path: Path to save the plot
            
        Returns:
            Figure object (matplotlib or plotly)
        """
        if self.config.backend == PlotBackend.MATPLOTLIB:
            return self._plot_phase_space_matplotlib(values, title, save_path)
        else:
            return self._plot_phase_space_plotly(values, title, save_path)
    
    def _plot_phase_space_matplotlib(self, values: List[AbsoluteValue],
                                   title: str, save_path: Optional[str]) -> Any:
        """Plot phase space using matplotlib."""
        fig, ax = plt.subplots(figsize=(self.config.width / 100, self.config.height / 100))
        
        # Extract data
        magnitudes = [v.magnitude for v in values]
        directions = [v.direction for v in values]
        
        # Create scatter plot
        scatter = ax.scatter(magnitudes, directions, 
                           c=range(len(values)), cmap='viridis',
                           s=self.config.marker_size**2, alpha=self.config.alpha,
                           edgecolors='black', linewidth=0.5)
        
        # Add special regions
        # Absolute region (magnitude = 0)
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Absolute Line')
        
        # Direction boundaries
        ax.axhline(y=0, color='blue', linestyle='--', alpha=0.7, label='Direction Boundary')
        
        # Customize plot
        ax.set_title(title, fontsize=self.config.title_size, pad=20)
        ax.set_xlabel("Magnitude", fontsize=self.config.axis_label_size)
        ax.set_ylabel("Direction", fontsize=self.config.axis_label_size)
        
        if self.config.legend:
            ax.legend(loc='upper right')
        
        if self.config.grid:
            ax.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Sequence Index', fontsize=self.config.axis_label_size)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        
        return fig
    
    def _plot_phase_space_plotly(self, values: List[AbsoluteValue],
                               title: str, save_path: Optional[str]) -> Any:
        """Plot phase space using plotly."""
        # Extract data
        magnitudes = [v.magnitude for v in values]
        directions = [v.direction for v in values]
        
        fig = go.Figure()
        
        # Add scatter plot
        fig.add_trace(go.Scatter(
            x=magnitudes,
            y=directions,
            mode='markers',
            marker=dict(
                color=list(range(len(values))),
                colorscale='Viridis',
                size=self.config.marker_size,
                opacity=self.config.alpha,
                line=dict(width=1, color='black'),
                colorbar=dict(title="Sequence Index")
            ),
            name="AbsoluteValues"
        ))
        
        # Add special lines
        fig.add_vline(x=0, line_dash="dash", line_color="red", opacity=0.7,
                     annotation_text="Absolute Line")
        fig.add_hline(y=0, line_dash="dash", line_color="blue", opacity=0.7,
                     annotation_text="Direction Boundary")
        
        # Update layout
        fig.update_layout(
            title=dict(text=title, font=dict(size=self.config.title_size)),
            xaxis_title=dict(text="Magnitude", font=dict(size=self.config.axis_label_size)),
            yaxis_title=dict(text="Direction", font=dict(size=self.config.axis_label_size)),
            showlegend=self.config.legend,
            width=self.config.width,
            height=self.config.height,
            font=dict(size=self.config.font_size)
        )
        
        if self.config.grid:
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def create_interactive_dashboard(self, values: List[AbsoluteValue],
                                   ratios: List[EternalRatio],
                                   records: List[CompensationRecord],
                                   title: str = "Balansis Interactive Dashboard",
                                   save_path: Optional[str] = None) -> Any:
        """Create an interactive dashboard combining multiple visualizations.
        
        Args:
            values: List of AbsoluteValue objects
            ratios: List of EternalRatio objects
            records: List of CompensationRecord objects
            title: Dashboard title
            save_path: Path to save the dashboard
            
        Returns:
            Plotly figure with interactive dashboard
        """
        if self.config.backend != PlotBackend.PLOTLY:
            raise ValueError("Interactive dashboard requires Plotly backend")
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                "AbsoluteValue Distribution",
                "EternalRatio Sequence",
                "ACT Phase Space",
                "Compensation Analysis",
                "Stability Analysis",
                "Mathematical Properties"
            ],
            specs=[
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "bar"}],
                [{"type": "scatter"}, {"type": "scatter"}]
            ],
            vertical_spacing=0.08,
            horizontal_spacing=0.1
        )
        
        # Plot 1: AbsoluteValue distribution
        indices = list(range(len(values)))
        magnitudes = [v.magnitude for v in values]
        directions = [v.direction for v in values]
        
        fig.add_trace(go.Scatter(
            x=indices,
            y=magnitudes,
            mode='markers+lines',
            marker=dict(
                color=[self.config.color_palette[0] if d > 0 else self.config.color_palette[1] for d in directions],
                size=self.config.marker_size,
                opacity=self.config.alpha
            ),
            name="AbsoluteValues"
        ), row=1, col=1)
        
        # Plot 2: EternalRatio sequence
        ratio_indices = list(range(len(ratios)))
        numerical_values = [r.numerical_value() for r in ratios]
        
        fig.add_trace(go.Scatter(
            x=ratio_indices,
            y=numerical_values,
            mode='markers+lines',
            marker=dict(
                color=self.config.color_palette[2],
                size=self.config.marker_size,
                opacity=self.config.alpha
            ),
            name="EternalRatios"
        ), row=1, col=2)
        
        # Plot 3: Phase space
        fig.add_trace(go.Scatter(
            x=magnitudes,
            y=directions,
            mode='markers',
            marker=dict(
                color=indices,
                colorscale='Viridis',
                size=self.config.marker_size,
                opacity=self.config.alpha
            ),
            name="Phase Space"
        ), row=2, col=1)
        
        # Plot 4: Compensation types
        if records:
            compensation_types = [record.compensation_type.value for record in records]
            type_counts = {}
            for comp_type in compensation_types:
                type_counts[comp_type] = type_counts.get(comp_type, 0) + 1
            
            fig.add_trace(go.Bar(
                x=list(type_counts.keys()),
                y=list(type_counts.values()),
                marker_color=self.config.color_palette[3],
                opacity=self.config.alpha,
                name="Compensation Types"
            ), row=2, col=2)
        
        # Plot 5: Stability analysis
        stability_values = [1 if r.is_stable() else 0 for r in ratios]
        fig.add_trace(go.Scatter(
            x=ratio_indices,
            y=stability_values,
            mode='markers+lines',
            marker=dict(
                color=self.config.color_palette[4],
                size=self.config.marker_size,
                opacity=self.config.alpha
            ),
            name="Stability"
        ), row=3, col=1)
        
        # Plot 6: Mathematical properties summary
        properties = {
            'Absolute Count': sum(1 for v in values if v.is_absolute()),
            'Positive Count': sum(1 for v in values if v.is_positive()),
            'Negative Count': sum(1 for v in values if v.is_negative()),
            'Stable Ratios': sum(1 for r in ratios if r.is_stable())
        }
        
        fig.add_trace(go.Bar(
            x=list(properties.keys()),
            y=list(properties.values()),
            marker_color=self.config.color_palette[4],
            opacity=self.config.alpha,
            name="Properties"
        ), row=3, col=2)
        
        # Update layout
        fig.update_layout(
            title=dict(text=title, font=dict(size=self.config.title_size + 4)),
            showlegend=False,
            width=self.config.width * 2,
            height=self.config.height * 1.8,
            font=dict(size=self.config.font_size)
        )
        
        # Update axes
        fig.update_xaxes(title_text="Index", row=1, col=1)
        fig.update_yaxes(title_text="Magnitude", row=1, col=1)
        fig.update_xaxes(title_text="Index", row=1, col=2)
        fig.update_yaxes(title_text="Ratio Value", row=1, col=2)
        fig.update_xaxes(title_text="Magnitude", row=2, col=1)
        fig.update_yaxes(title_text="Direction", row=2, col=1)
        fig.update_xaxes(title_text="Compensation Type", row=2, col=2)
        fig.update_yaxes(title_text="Count", row=2, col=2)
        fig.update_xaxes(title_text="Index", row=3, col=1)
        fig.update_yaxes(title_text="Stable (1) / Unstable (0)", row=3, col=1)
        fig.update_xaxes(title_text="Property", row=3, col=2)
        fig.update_yaxes(title_text="Count", row=3, col=2)
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def animate_sequence_evolution(self, sequences: List[List[AbsoluteValue]],
                                 title: str = "AbsoluteValue Sequence Evolution",
                                 save_path: Optional[str] = None) -> Any:
        """Create an animated plot showing evolution of AbsoluteValue sequences.
        
        Args:
            sequences: List of AbsoluteValue sequences (each representing a time step)
            title: Animation title
            save_path: Path to save the animation
            
        Returns:
            Animation object (matplotlib or plotly)
        """
        if self.config.backend == PlotBackend.MATPLOTLIB:
            return self._animate_matplotlib(sequences, title, save_path)
        else:
            return self._animate_plotly(sequences, title, save_path)
    
    def _animate_matplotlib(self, sequences: List[List[AbsoluteValue]],
                          title: str, save_path: Optional[str]) -> FuncAnimation:
        """Create matplotlib animation."""
        fig, ax = plt.subplots(figsize=(self.config.width / 100, self.config.height / 100))
        
        # Set up the plot
        max_magnitude = max(max(v.magnitude for v in seq) for seq in sequences if seq)
        max_index = max(len(seq) for seq in sequences)
        
        ax.set_xlim(0, max_index)
        ax.set_ylim(0, max_magnitude * 1.1)
        ax.set_title(title, fontsize=self.config.title_size)
        ax.set_xlabel("Index", fontsize=self.config.axis_label_size)
        ax.set_ylabel("Magnitude", fontsize=self.config.axis_label_size)
        
        if self.config.grid:
            ax.grid(True, alpha=0.3)
        
        # Initialize empty plot elements
        scatter = ax.scatter([], [], s=self.config.marker_size**2, alpha=self.config.alpha)
        line, = ax.plot([], [], linewidth=self.config.line_width, alpha=0.7)
        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=self.config.font_size)
        
        def animate(frame):
            """Animation function."""
            if frame < len(sequences):
                sequence = sequences[frame]
                indices = list(range(len(sequence)))
                magnitudes = [v.magnitude for v in sequence]
                directions = [v.direction for v in sequence]
                
                # Update scatter plot
                colors = [self.config.color_palette[0] if d > 0 else self.config.color_palette[1] for d in directions]
                scatter.set_offsets(list(zip(indices, magnitudes)))
                scatter.set_color(colors)
                
                # Update line plot
                line.set_data(indices, magnitudes)
                
                # Update time text
                time_text.set_text(f'Time Step: {frame}')
            
            return scatter, line, time_text
        
        # Create animation
        anim = FuncAnimation(fig, animate, frames=len(sequences), 
                           interval=200, blit=True, repeat=True)
        
        if save_path:
            anim.save(save_path, writer='pillow', fps=5)
        
        return anim
    
    def _animate_plotly(self, sequences: List[List[AbsoluteValue]],
                      title: str, save_path: Optional[str]) -> Any:
        """Create plotly animation."""
        frames = []
        
        for i, sequence in enumerate(sequences):
            if not sequence:
                continue
                
            indices = list(range(len(sequence)))
            magnitudes = [v.magnitude for v in sequence]
            directions = [v.direction for v in sequence]
            
            frame = go.Frame(
                data=[
                    go.Scatter(
                        x=indices,
                        y=magnitudes,
                        mode='markers+lines',
                        marker=dict(
                            color=[self.config.color_palette[0] if d > 0 else self.config.color_palette[1] for d in directions],
                            size=self.config.marker_size,
                            opacity=self.config.alpha
                        ),
                        line=dict(width=self.config.line_width, color=self.config.color_palette[0]),
                        name=f"Step {i}"
                    )
                ],
                name=str(i)
            )
            frames.append(frame)
        
        # Create initial figure
        if sequences and sequences[0]:
            initial_sequence = sequences[0]
            indices = list(range(len(initial_sequence)))
            magnitudes = [v.magnitude for v in initial_sequence]
            directions = [v.direction for v in initial_sequence]
            
            fig = go.Figure(
                data=[
                    go.Scatter(
                        x=indices,
                        y=magnitudes,
                        mode='markers+lines',
                        marker=dict(
                            color=[self.config.color_palette[0] if d > 0 else self.config.color_palette[1] for d in directions],
                            size=self.config.marker_size,
                            opacity=self.config.alpha
                        ),
                        line=dict(width=self.config.line_width, color=self.config.color_palette[0]),
                        name="AbsoluteValues"
                    )
                ],
                frames=frames
            )
        else:
            fig = go.Figure(frames=frames)
        
        # Add animation controls
        fig.update_layout(
            title=dict(text=title, font=dict(size=self.config.title_size)),
            xaxis_title=dict(text="Index", font=dict(size=self.config.axis_label_size)),
            yaxis_title=dict(text="Magnitude", font=dict(size=self.config.axis_label_size)),
            width=self.config.width,
            height=self.config.height,
            font=dict(size=self.config.font_size),
            updatemenus=[
                {
                    "buttons": [
                        {
                            "args": [None, {"frame": {"duration": 500, "redraw": True},
                                           "fromcurrent": True, "transition": {"duration": 300}}],
                            "label": "Play",
                            "method": "animate"
                        },
                        {
                            "args": [[None], {"frame": {"duration": 0, "redraw": True},
                                             "mode": "immediate", "transition": {"duration": 0}}],
                            "label": "Pause",
                            "method": "animate"
                        }
                    ],
                    "direction": "left",
                    "pad": {"r": 10, "t": 87},
                    "showactive": False,
                    "type": "buttons",
                    "x": 0.1,
                    "xanchor": "right",
                    "y": 0,
                    "yanchor": "top"
                }
            ],
            sliders=[
                {
                    "active": 0,
                    "yanchor": "top",
                    "xanchor": "left",
                    "currentvalue": {
                        "font": {"size": 20},
                        "prefix": "Time Step:",
                        "visible": True,
                        "xanchor": "right"
                    },
                    "transition": {"duration": 300, "easing": "cubic-in-out"},
                    "pad": {"b": 10, "t": 50},
                    "len": 0.9,
                    "x": 0.1,
                    "y": 0,
                    "steps": [
                        {
                            "args": [
                                [str(k)],
                                {
                                    "frame": {"duration": 300, "redraw": True},
                                    "mode": "immediate",
                                    "transition": {"duration": 300}
                                }
                            ],
                            "label": str(k),
                            "method": "animate"
                        }
                        for k in range(len(sequences))
                    ]
                }
            ]
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def export_plot_data(self, 
                        absolute_values: Optional[List[AbsoluteValue]] = None,
                        eternal_ratios: Optional[List[EternalRatio]] = None,
                        compensation_records: Optional[List[CompensationRecord]] = None,
                        format: str = 'csv',
                        filename: str = 'export.csv') -> str:
        """Export plot data to file.
        
        Args:
            absolute_values: List of AbsoluteValue objects
            eternal_ratios: List of EternalRatio objects
            compensation_records: List of CompensationRecord objects
            format: Export format ('csv', 'json', 'excel', 'xlsx')
            filename: Name of the output file
            
        Returns:
            str: The filename of the exported file
        """
        import pandas as pd
        import json
        
        # Check if any data is provided
        if not any([absolute_values, eternal_ratios, compensation_records]):
            raise ValueError("No data provided for export")
        
        # Prepare data dictionary
        data = {}
        
        # Add absolute values data
        if absolute_values:
            data.update({
                'absolute_magnitudes': [v.magnitude for v in absolute_values],
                'absolute_directions': [v.direction for v in absolute_values],
                'absolute_is_absolute': [v.is_absolute() for v in absolute_values]
            })
        
        # Add eternal ratios data
        if eternal_ratios:
            data.update({
                'ratio_numerical': [r.numerical_value() for r in eternal_ratios],
                'ratio_signed': [r.signed_value() for r in eternal_ratios],
                'ratio_stable': [r.is_stable() for r in eternal_ratios]
            })
        
        # Add compensation records data
        if compensation_records:
            data.update({
                'compensation_type': [r.compensation_type.value for r in compensation_records],
                'original_value': [r.original_value.magnitude for r in compensation_records],
                'compensated_value': [r.compensated_value.magnitude for r in compensation_records],
                'compensation_factor': [r.compensation_factor for r in compensation_records],
                'stability_gain': [r.stability_gain for r in compensation_records]
            })
        
        # Export based on format
        format_lower = format.lower()
        if format_lower == 'csv':
            df = pd.DataFrame(data)
            df.to_csv(filename, index=False)
        elif format_lower == 'json':
            df = pd.DataFrame(data)
            df.to_json(filename, orient="records", indent=2)
        elif format_lower in ['excel', 'xlsx']:
            df = pd.DataFrame(data)
            df.to_excel(filename, index=False)
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        return filename
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"PlotUtils(backend={self.config.backend.value}, style={self.config.style.value})"
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        return f"PlotUtils with {self.config.backend.value} backend and {self.config.style.value} style"