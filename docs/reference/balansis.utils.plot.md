# balansis.utils.plot

[Source](../../balansis/utils/plot.py) · [Reference index](index.md)

```python
class PlotStyle(str, Enum):
    ...
```

```python
class PlotBackend(str, Enum):
    ...
```

```python
class PlotConfig(BaseModel):
    style: PlotStyle = Field(default=PlotStyle.SCIENTIFIC, description='Plot style theme')

    backend: PlotBackend = Field(default=PlotBackend.MATPLOTLIB, description='Plotting backend')

    width: int = Field(default=800, description='Figure width in pixels')

    height: int = Field(default=600, description='Figure height in pixels')

    dpi: int = Field(default=100, description='Resolution for matplotlib')

    color_palette: List[str] = Field(default=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'], description='Color palette for plots')

    font_size: int = Field(default=12, description='Base font size')

    line_width: float = Field(default=2.0, description='Default line width')

    marker_size: float = Field(default=6.0, description='Default marker size')

    alpha: float = Field(default=0.8, description='Default transparency')

    grid: bool = Field(default=True, description='Show grid')

    legend: bool = Field(default=True, description='Show legend')

    title_size: int = Field(default=14, description='Title font size')

    axis_label_size: int = Field(default=14, description='Axis label font size')

    interactive: bool = Field(default=False, description='Enable interactive features')

    save_format: str = Field(default='png', description='Default save format for plots')

    animation_duration: int = Field(default=1000, description='Animation duration in milliseconds')

    animation_frames: int = Field(default=50, description='Number of animation frames')

    @classmethod
    def validate_width(cls, v: int) -> int:
        ...

    @classmethod
    def validate_height(cls, v: int) -> int:
        ...

    @classmethod
    def validate_dpi(cls, v: int) -> int:
        ...

    @classmethod
    def validate_alpha(cls, v: float) -> float:
        ...
```

```python
class PlotUtils:
    def __init__(self, config: Optional[PlotConfig]=None, operations: Optional[Operations]=None, compensator: Optional[Compensator]=None):
        ...

    def plot_absolute_values(self, values: List[AbsoluteValue], title: str='AbsoluteValue Distribution', xlabel: str='Index', ylabel: str='Magnitude', show_directions: bool=True, save_path: Optional[str]=None) -> Any:
        ...

    def plot_eternal_ratios(self, ratios: List[EternalRatio], title: str='EternalRatio Sequence', xlabel: str='Index', ylabel: str='Ratio Value', show_stability: bool=True, save_path: Optional[str]=None) -> Any:
        ...

    def plot_compensation_analysis(self, records: List[CompensationRecord], title: str='Compensation Analysis', save_path: Optional[str]=None) -> Any:
        ...

    def plot_act_phase_space(self, values: List[AbsoluteValue], title: str='ACT Phase Space', save_path: Optional[str]=None) -> Any:
        ...

    def create_interactive_dashboard(self, values: List[AbsoluteValue], ratios: List[EternalRatio], records: List[CompensationRecord], title: str='Balansis Interactive Dashboard', save_path: Optional[str]=None) -> Any:
        ...

    def animate_sequence_evolution(self, sequences: List[List[AbsoluteValue]], title: str='AbsoluteValue Sequence Evolution', save_path: Optional[str]=None) -> Any:
        ...

    def export_plot_data(self, absolute_values: Optional[List[AbsoluteValue]]=None, eternal_ratios: Optional[List[EternalRatio]]=None, compensation_records: Optional[List[CompensationRecord]]=None, format: str='csv', filename: str='export.csv') -> str:
        ...
```
