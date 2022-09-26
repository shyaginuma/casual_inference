from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go

from .base import BaseEvaluator


class SampleSizeEvaluator(BaseEvaluator):
    def __init__(self) -> None:
        super().__init__()

    def evaluate(self, data: pd.DataFrame, unit_col: str, metrics: list[str], n_variant: int = 2) -> None:
        """Calculate statistics of metrics and mde with simulating A/B test threshold.

        Parameters
        ----------
        data : pd.DataFrame
            Dataframe has randomization unit column, and metrics columns. The data should have been aggregated by the randomization unit.
        unit_col : str
            A column name stores the randomization unit. something like user_id, session_id, ...
        metrics : list[str]
            Columns stores metrics you want to evaluate.
        n_variant : int, optional
            The number of variant planned in the A/B test, by default 2
        """
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

    def summary_table(self, target_mde: Optional[float] = None) -> pd.DataFrame:
        """Find threshold suffices the provided target MDE.

        Parameters
        ----------
        target_mde : Optional[float], optional
            The relative Minimum Detectable Effect you want to set in the A/B test.

        Returns
        -------
        pd.DataFrame
            stats summary
        """
        self._validate_evaluate_executed()
        _validate_target_mde(target_mde)
        stats = self.stats.copy(deep=True)
        if target_mde:
            return stats.loc[stats["mde_rel"] <= target_mde].groupby("metric").head(1)
        return stats

    def summary_plot(self, target_mde: Optional[float] = None) -> go.Figure:
        """Plot threshold vs MDE curve

        Parameters
        ----------
        target_mde : Optional[float], optional
            The relative Minimum Detectable Effect you want to set in the A/B test.

        Returns
        -------
        go.Figure
        """
        self._validate_evaluate_executed()
        _validate_target_mde(target_mde)

        g = px.line(data_frame=self.stats, x="threshold", y="mde_rel", color="metric", markers=True)
        if target_mde:
            g.add_hline(y=target_mde, line_dash="dot", line_width=2)
        return g


def _validate_target_mde(target_mde: Optional[float] = None) -> None:
    if target_mde is None or (0 < target_mde < 1):
        return
    raise ValueError("The target MDE should be (0, 1).")
