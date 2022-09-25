import numpy as np
import pandas as pd
import plotly.graph_objs as go

from .base import BaseEvaluator


class SampleSizeEvaluator(BaseEvaluator):
    def __init__(self) -> None:
        super().__init__()

    def evaluate(self, data: pd.DataFrame, unit_col: str, metrics: list[str], n_variant: int = 2) -> None:
        self._validate_passed_data(data, unit_col, metrics)
        stats = pd.DataFrame()
        for metric in metrics:
            stats_partial = pd.DataFrame()
            stats_partial["threshold"] = np.arange(0.01, 1.01, 0.01)
            stats_partial["metric"] = metric
            stats_partial["mean"] = data[metric].mean()
            stats_partial["var"] = data[metric].var()
            stats_partial["count"] = data[metric].count()
            stats = pd.concat([stats, stats_partial])

        stats["sample_size"] = stats["threshold"] * stats["count"] / n_variant
        stats["mde_abs"] = 4 * np.sqrt(stats["var"] / stats["sample_size"])
        stats["mde_rel"] = stats["mde_abs"] / stats["mean"]
        self.stats = stats

    def summary_table(self) -> pd.DataFrame:
        self._validate_evaluate_executed()
        return super().summary_table()

    def summary_plot(self) -> go.Figure:
        self._validate_evaluate_executed()
        return super().summary_plot()
