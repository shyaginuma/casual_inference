import pandas as pd
import plotly.graph_objs as go
from sklearn.preprocessing import StandardScaler
from typing_extensions import Self

from .base import BaseEvaluator

class LinearRegressionEvaluator(BaseEvaluator):
    """Evaluate treatment impact by Linear Regression
    """

    def __init__(self) -> None:
        super().__init__()

    def evaluate(self, data: pd.DataFrame, unit_col: str, metrics: list[str]) -> Self:
        return super().evaluate(data, unit_col, metrics)

    def summary_table(self) -> pd.DataFrame:
        return super().summary_table()

    def summary_plot(self) -> go.Figure:
        return super().summary_plot()
