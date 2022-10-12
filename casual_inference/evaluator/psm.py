
import pandas as pd
import plotly.graph_objs as go
from typing_extensions import Self
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression

from .base import BaseEvaluator


class PSMEvaluator(BaseEvaluator):
    """Evaluate treatment impact by Propensity Score Matching based on the Nearest Neighbor approach.

    Although it automatically fit the model for propensity score prediction, you also can pass your custom model. (Assuming sklearn like model)
    """

    def __init__(self) -> None:
        super().__init__()
        self.ps_model: BaseEstimator = None

    def evaluate(
        self,
        data: pd.DataFrame,
        unit_col: str,
        metrics: list[str],
        variant_col: str = "variant",
        covariates: list[str] = [],
        custom_ps_model: BaseEstimator = None
    ) -> Self:
        return super().evaluate(data, unit_col, metrics)

    def summary_table(self) -> pd.DataFrame:
        return super().summary_table()

    def summary_plot(self) -> go.Figure:
        return super().summary_plot()

    def summary_ps_model(self):
        pass
