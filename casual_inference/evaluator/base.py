from abc import ABC, abstractmethod
from typing import Union

import pandas as pd
import plotly.graph_objs as go
from typing_extensions import Self

from ..model import CustomMetric


class BaseEvaluator(ABC):
    def __init__(self) -> None:
        self.stats: pd.DataFrame = pd.DataFrame()

    @abstractmethod
    # ignore mypy error temporary, because the "Self" type support on mypy is ongoing. https://github.com/python/mypy/pull/11666
    def evaluate(self, data: pd.DataFrame, unit_col: str, metrics: list[Union[str, CustomMetric]]) -> Self:  # type: ignore
        return self

    @abstractmethod
    def summary_table(self) -> pd.DataFrame:
        return pd.DataFrame()

    @abstractmethod
    def summary_plot(self) -> go.Figure:
        return go.Figure()

    def _validate_evaluate_executed(self) -> None:
        if self.stats.shape[0] == 0:
            raise ValueError("Evaluated statistics haven't been calculated. Please call evaluate() in advance.")

    def _validate_passed_data(self, data: pd.DataFrame, unit_col: str, metrics: list[Union[str, CustomMetric]]) -> None:
        if data.shape[0] != data[unit_col].nunique():
            raise ValueError("passed dataframe hasn't been aggregated by the randomization unit.")
        if len(metrics) == 0:
            raise ValueError("metrics hasn't been specified.")
