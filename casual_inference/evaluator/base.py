from abc import ABC, abstractmethod

import pandas as pd
import plotly.graph_objs as go


class BaseEvaluator(ABC):
    def __init__(self) -> None:
        self.stats: pd.DataFrame = pd.DataFrame()

    @abstractmethod
    def evaluate(self, data: pd.DataFrame, unit_col: str, metrics: list[str]) -> None:
        pass

    @abstractmethod
    def summary_table(self) -> pd.DataFrame:
        return pd.DataFrame()

    @abstractmethod
    def summary_plot(self) -> go.Figure:
        return go.Figure()

    def _validate_evaluate_executed(self) -> None:
        if self.stats.shape[0] == 0:
            raise ValueError("Evaluated statistics haven't been calculated. Please call evaluate() in advance.")

    def _validate_passed_data(self, data: pd.DataFrame, unit_col: str, metrics: list[str]) -> None:
        if data.shape[0] != data[unit_col].nunique():
            raise ValueError("passed dataframe hasn't been aggregated by the randomization unit.")
        if len(metrics) == 0:
            raise ValueError("metrics hasn't been specified.")
