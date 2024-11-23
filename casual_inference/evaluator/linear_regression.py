import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import statsmodels.formula.api as smf
from typing_extensions import Self

from ..model import CustomMetric
from .base import BaseEvaluator


class LinearRegressionEvaluator(BaseEvaluator):
    """Evaluate treatment impact by Linear Regression

    Attributes
    ----------
    models : dict[str, smf.ols]
        linear regression models built to evaluate the impact of each metric.
    """

    def __init__(self) -> None:
        super().__init__()
        self.models: dict[str | CustomMetric, smf.ols] = dict()

    # ignore mypy error temporary, because the "Self" type support on mypy is ongoing. https://github.com/python/mypy/pull/11666
    def evaluate(
        self,
        data: pd.DataFrame,
        unit_col: str,
        metrics: list[str | CustomMetric],
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
        if set(data[treatment_col].unique()) != {0, 1}:
            raise ValueError("The treatment value should be binary.")

        # to avoid errors in later steps
        data[treatment_col] = data[treatment_col].astype(int)
        for metric_col in metrics:
            data[metric_col] = data[metric_col].astype(np.float64)

        covariates_str = ""
        if len(covariates) > 0:
            covariates_str = "+ " + "+ ".join(covariates)

        stats = self.stats.copy(deep=True)
        for metric in metrics:
            model = smf.ols(formula=f"{metric} ~ {treatment_col} {covariates_str}", data=data).fit()
            self.models[metric] = model

            # convert statsmodels summary table to pandas dataframe
            stats_partial = pd.read_html(model.summary().tables[1].as_html(), header=0, index_col=0)[0].reset_index()
            stats_partial["metric"] = metric
            stats = pd.concat([stats, stats_partial])

        self.stats = stats.loc[stats["index"] == treatment_col].reset_index(drop=True).drop("index", axis=1)
        return self

    def summary_table(self) -> pd.DataFrame:
        self._validate_evaluate_executed()
        stats = self.stats.copy(deep=True)
        return stats

    def summary_plot(self, p_threshold: float = 0.05, display_ci: bool = True) -> go.Figure:
        """Plot evaluated impact and confidence interval for each metric.
        Currently it only supports the absolute difference visualization.

        Parameters
        ----------
        p_threshold : float, optional
            significance level, by default 0.05
        display_ci : bool, optional
            Whether to display confidence interval, by default True

        Returns
        -------
        go.Figure
        """
        stats = self.stats.copy(deep=True)
        self._validate_evaluate_executed()
        stats["abs_ci_width"] = stats["coef"] - stats["[0.025"]
        stats["significant"] = stats.apply(
            lambda x: (
                "up"
                if x["P>|t|"] <= p_threshold and x["t"] > 0
                else "down" if x["P>|t|"] <= p_threshold and x["t"] < 0 else "unclear"
            ),
            axis=1,
        )
        stats.rename(columns={"coef": "impact_abs"}, inplace=True)

        viz_options = {
            "data_frame": stats,
            "x": "impact_abs",
            "y": "metric",
            "color": "significant",
            "color_discrete_map": {"up": "#54A24B", "down": "#E45756", "unclear": "silver"},
        }
        if display_ci:
            viz_options["error_x"] = "abs_ci_width"

        g = px.bar(**viz_options)
        return g
