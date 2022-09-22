from abc import ABC, abstractmethod

import pandas as pd
import plotly.graph_objs as go

from .aatest import AATestEvaluator
from .abtest import ABTestEvaluator
from .samplesize import SampleSizeEvaluator

__all__ = ["ABTestEvaluator", "AATestEvaluator", "SampleSizeEvaluator"]


class BaseEvaluator(ABC):
    @abstractmethod
    def evaluate(self, data: pd.DataFrame, unit_col: str, metrics: list[str]) -> None:
        pass

    @abstractmethod
    def summary_table(self) -> pd.DataFrame:
        return pd.DataFrame()

    @abstractmethod
    def summary_plot(self) -> go.Figure:
        return go.Figure()
