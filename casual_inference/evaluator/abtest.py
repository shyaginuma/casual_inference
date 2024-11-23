import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import pandas.api.types as pd_types
import plotly.express as px
import plotly.graph_objs as go
from scipy.stats import chisquare
from typing_extensions import Self

from ..statistical_testing import eval_ttest_significance, t_test
from .base import BaseEvaluator


@dataclass
class SRMCheckResult:
    variant: int
    sample_size: int
    sample_size_c: int
    chi_square: float
    p_value: float
    significant: bool
    segment: Optional[str] = None

    def __repr__(self) -> str:
        if self.segment:
            return f"variant: {self.variant}, segment: {self.segment} control:treatment = {self.sample_size_c}:{self.sample_size}, p_value = {self.p_value}"
        else:
            return f"variant: {self.variant} control:treatment = {self.sample_size_c}:{self.sample_size}, p_value = {self.p_value}"


class ABTestEvaluator(BaseEvaluator):
    def __init__(self) -> None:
        super().__init__()
        self.variant_col: str = ""
        self.segment_col: str = ""

    # ignore mypy error temporary, because the "Self" type support on mypy is ongoing. https://github.com/python/mypy/pull/11666
    def evaluate(
        self,
        data: pd.DataFrame,
        unit_col: str,
        metrics: list[str],
        variant_col: str = "variant",
        segment_col: Optional[str] = None,
    ) -> Self:  # type: ignore
        """calculate stats of A/B test and cache it into the class variable.
        At first, it only assumes metrics can handle by Welch's t-test.

        Parameters
        ----------
        data : pd.DataFrame
            Dataframe has randomization unit column, variant assignment column, and metrics columns.
            The data should have been aggregated by the randomization unit.
        unit_col : str
            A column name stores the randomization unit. something like user_id, session_id, ...
        metrics : list[str]
            Columns stores metrics you want to evaluate.
        variant_col : str
            A column name stores the variant assignment.
            The control variant should have value 1.
        segment_col : Optional[str]
            A column name stores 'segment' you want to break down in the analysis.
            e.g., light users, heavy users, new registers, ...
            When the specified column in dataframe stores numerical values, it automatically binning the values.

        Returns
        -------
        self : object
            Evaluator storing statistics calculated.
        """
        self._validate_passed_data(data, unit_col, metrics)

        # to avoid errors in later steps
        data[variant_col] = data[variant_col].astype(int)
        for metric_col in metrics:
            data[metric_col] = data[metric_col].astype(np.float64)

        if segment_col:
            segment = data[segment_col]
            if pd_types.is_numeric_dtype(segment) and not pd_types.is_bool_dtype(segment):
                segment = pd.qcut(x=segment, q=5, duplicates="drop")

            stats = pd.DataFrame()
            for s in segment.unique():
                partial_stats = t_test(data.loc[segment == s], unit_col, variant_col, metrics)
                partial_stats[segment_col] = s
                stats = pd.concat([stats, partial_stats])
            self.stats = stats
            self.segment_col = segment_col
        else:
            self.stats = t_test(data, unit_col, variant_col, metrics)
        self.variant_col = variant_col
        return self

    def summary_table(self, p_threshold: float = 0.05) -> pd.DataFrame:
        """return statistics summary.

        Parameters
        ----------
        p_threshold : float, optional
            significance level, by default 0.05

        Returns
        -------
        pd.DataFrame
            stats summary
        """
        self._validate_evaluate_executed()

        stats = self.stats.copy(deep=True)
        significance, abs_ci_width, rel_ci_width = eval_ttest_significance(self.stats, p_threshold)
        stats["significance"] = significance
        stats["abs_ci_width"] = abs_ci_width
        stats["rel_ci_width"] = rel_ci_width
        for diff_type in ["abs", "rel"]:
            stats[f"ci_{diff_type}_diff"] = stats.apply(
                lambda x: (
                    x[f"{diff_type}_diff_mean"] - x[f"{diff_type}_ci_width"],
                    x[f"{diff_type}_diff_mean"] + x[f"{diff_type}_ci_width"],
                ),
                axis=1,
            )
            del stats[f"{diff_type}_ci_width"]

        srm_check_results = self._diagnose_srm()
        for result in srm_check_results:
            if result.significant:
                print(f"[Warning] SRM was detected, pay attention to interpret the result. {result}")

        return_cols = [col for col in stats.columns if col[-2:] != "_c"]
        return stats.loc[:, return_cols]

    def summary_plot(self, p_threshold: float = 0.05, diff_type: str = "rel", display_ci: bool = True) -> go.Figure:
        """plot impact and confidence interval for each metric

        Parameters
        ----------
        p_threshold : float, optional
            significance level, by default 0.05
        diff_type : str, optional
            The type of difference you want to plot, by default 'rel'
        display_ci : bool, optional
            Whether to display confidence interval, by default True

        Returns
        -------
        go.Figure
        """
        self._validate_evaluate_executed()
        if diff_type not in ["rel", "abs"]:
            raise ValueError("Specified diff type is invalid.")

        stats = self.stats.copy(deep=True)
        significance, abs_ci_width, rel_ci_width = eval_ttest_significance(self.stats, p_threshold)
        stats["significant"] = significance
        stats["abs_ci_width"] = abs_ci_width
        stats["rel_ci_width"] = rel_ci_width

        viz_options = {
            "data_frame": stats.loc[stats[self.variant_col] > 1],  # not display control group
            "x": f"{diff_type}_diff_mean",
            "y": "metric",
            "facet_col": self.variant_col,
            "color": "significant",
            "color_discrete_map": {"up": "#54A24B", "down": "#E45756", "unclear": "silver"},
        }
        if display_ci:
            viz_options["error_x"] = f"{diff_type}_ci_width"
        if len(self.segment_col) > 0:
            viz_options["facet_row"] = self.segment_col
            viz_options["height"] = stats[self.segment_col].nunique() * 200

        srm_check_results = self._diagnose_srm()
        for result in srm_check_results:
            if result.significant:
                print(f"[Warning] SRM was detected, pay attention to interpret the result. {result}")

        g = px.bar(**viz_options)
        return g

    def summary_barplot(self, p_threshold: float = 0.05, diff_type: str = "rel", display_ci: bool = True) -> go.Figure:
        """plot impact and confidence interval for each metric

        Parameters
        ----------
        p_threshold : float, optional
            significance level, by default 0.05
        diff_type : str, optional
            The type of difference you want to plot, by default 'rel'
        display_ci : bool, optional
            Whether to display confidence interval, by default True

        Returns
        -------
        go.Figure
        """
        warnings.warn(
            "The summary_barplot() is deprecated and will be removed in the future release."
            "Please use summary_plot() instead.",
            FutureWarning,
        )
        return self.summary_plot(p_threshold=p_threshold, diff_type=diff_type, display_ci=display_ci)

    def _diagnose_srm(self) -> list[SRMCheckResult]:
        """Diagnosing Sample Ratio Mismatch by applying chi-square goodness of fit test.
        Assuming each variant has the same sample ratio.
        See this paper to understand what the SRM is: https://dl.acm.org/doi/10.1145/3292500.3330722

        Returns
        -------
        list[SRMCheckResult]
        """
        self._validate_evaluate_executed()
        if self.segment_col:
            stats_subset = self.stats[[self.variant_col, self.segment_col, "count", "count_c"]].drop_duplicates()
        else:
            stats_subset = self.stats[[self.variant_col, "count", "count_c"]].drop_duplicates()
        chi2q, p_value = chisquare(f_obs=stats_subset[["count", "count_c"]], axis=1)
        stats_subset["chi_square"] = chi2q
        stats_subset["p_value"] = p_value
        stats_subset["significant"] = p_value < 0.05
        stats_subset = stats_subset.loc[stats_subset[self.variant_col] > 1]

        results = []
        for _, row in stats_subset.iterrows():
            result = SRMCheckResult(
                variant=row[self.variant_col],
                sample_size=row["count"],
                sample_size_c=row["count_c"],
                chi_square=row["chi_square"],
                p_value=row["p_value"],
                significant=row["significant"],
            )
            if self.segment_col:
                result.segment = row[self.segment_col]
            results.append(result)
        return results
