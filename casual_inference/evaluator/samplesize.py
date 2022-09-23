import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go

from .base import BaseEvaluator


class SampleSizeEvaluator(BaseEvaluator):
    def __init__(self) -> None:
        super().__init__()

    def evaluate(self, data: pd.DataFrame, unit_col: str, metrics: list[str], n_variant: int = 2) -> None:
        self._validate_passed_data(data, unit_col, metrics)
        metrics_stats = dict()
        stats = pd.DataFrame()
        stats["threshold"] = np.arange(0.01, 1, 0.01)
        for metric in metrics:
            metrics_stats[metric]["mean"] = data[metric].mean()
            metrics_stats[metric]["var"] = data[metric].var()
        self.stats = stats

    def summary_table(self) -> pd.DataFrame:
        self._validate_evaluate_executed()
        return super().summary_table()

    def summary_plot(self) -> go.Figure:
        self._validate_evaluate_executed()
        return super().summary_plot()
