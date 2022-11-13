import pandas as pd
import plotly.graph_objs as go
import statsmodels.formula.api as smf
from typing_extensions import Self

from .base import BaseEvaluator


class LinearRegressionEvaluator(BaseEvaluator):
    """Evaluate treatment impact by Linear Regression"""

    def __init__(self) -> None:
        super().__init__()
        self.models: dict[str, smf.ols] = dict()

    # ignore mypy error temporary, because the "Self" type support on mypy is ongoing. https://github.com/python/mypy/pull/11666
    def evaluate(
        self,
        data: pd.DataFrame,
        unit_col: str,
        metrics: list[str],
        treatment_col: str = "treatment",
        covariates: list[str] = [],
    ) -> Self:  # type: ignore
        """Evaluate impact by Linear Regression model

        Parameters
        ----------
        data : pd.DataFrame
            Dataframe has randomization unit column, and metrics columns. The data should have been aggregated by the randomization unit.
        unit_col : str
            A column name stores the randomization unit. something like user_id, session_id, ...
        metrics : list[str]
            Columns stores metrics you want to evaluate.
        treatment_col : str
            A column name stores information if the unit received treatment or not.
            Stored value should be binary:
                - 0: did not receive treatment
                - 1: received treatment
        covariates : list[str], optional
            column names of covariate variables you want to use for the analysis, by default []

        Returns
        -------
        Self
            Evaluator object has statistics calculated
        """
        self._validate_passed_data(data, unit_col, metrics)
        if data[treatment_col].unique() != [0, 1]:
            raise ValueError("The treatment value should be binary.")

        covariates_str = "+ ".join(covariates)

        for metric in metrics:
            model = smf.ols(formula=f"{metric} ~ {treatment_col} + {covariates_str}", data=data).fit()
            self.models[metric] = model

        return self

    def summary_table(self) -> pd.DataFrame:
        return super().summary_table()

    def summary_plot(self) -> go.Figure:
        return super().summary_plot()
