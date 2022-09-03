import numpy as np
import pandas as pd
import plotly
import plotly.express as px
from scipy.stats import t, ttest_ind_from_stats


class ABTestEvaluator:
    def __init__(self) -> None:
        self.stats: pd.DataFrame = None

    def evaluate(self, data: pd.DataFrame, unit_col: str, variant_col: str, metrics: list[str]) -> None:
        """calculate stats of A/B test and cache it into the class variable.
        At first, it only assumes metrics can handle by Welch's t-test.

        Parameters
        ----------
        data : pd.DataFrame
            Dataframe has randomization unit column, variant assignment column, and metrics columns.
            The data should have been aggregated by the randmization unit.
        unit_col : str
            A column name stores the randomization unit. something like user_id, session_id, ...
        variant_col : str
            A column name stores the variant assignment.
            The control variant should have value 1.
        metrics : list[str]
            Columns stores metrics you want to evaluate.
        """
        if data.shape[0] != data[unit_col].nunique():
            raise ValueError("passed dataframe hasn't been aggregated by the randomization unit.")

        if data[variant_col].min() != 1:
            raise ValueError("the control variant seems not to exist.")

        if len(metrics) == 0:
            raise ValueError("metrics hasn't been specified.")

        means = (
            data.groupby(variant_col)[metrics]
            .mean()
            .stack()
            .reset_index()
            .rename(columns={"level_1": "metric", 0: "mean"})
        )
        vars = (
            data.groupby(variant_col)[metrics]
            .var()
            .stack()
            .reset_index()
            .rename(columns={"level_1": "metric", 0: "var"})
        )
        counts = (
            data.groupby(variant_col)[metrics]
            .count()
            .stack()
            .reset_index()
            .rename(columns={"level_1": "metric", 0: "count"})
        )
        stats = means.merge(vars, on=[variant_col, "metric"]).merge(counts, on=[variant_col, "metric"])
        stats["std"] = np.sqrt(stats["var"])
        stats["stderr"] = np.sqrt(stats["var"] / stats["count"])
        stats = stats.merge(stats.query(f"{variant_col} == 1"), on="metric", suffixes=["", "_c"]).drop(
            "variant_c", axis=1
        )

        stats["abs_diff_mean"] = stats["mean"] - stats["mean_c"]
        stats["abs_diff_std"] = np.sqrt(stats["stderr"] ** 2 + stats["stderr_c"] ** 2)
        stats["rel_diff_mean"] = stats["mean"] / stats["mean_c"] - 1

        # see: https://arxiv.org/pdf/1803.06336.pdf
        stats["rel_diff_std"] = (
            np.sqrt((stats["stderr"] ** 2 + (stats["mean"] ** 2 / stats["mean_c"] ** 2) * stats["stderr_c"] ** 2))
            / stats["mean_c"]
        )
        stats["t_value"] = stats["abs_diff_mean"] / stats["abs_diff_std"]
        stats["dof"] = stats["abs_diff_std"] ** 4 / (
            stats["stderr"] ** 4 / (stats["count"] - 1) + stats["stderr_c"] ** 4 / (stats["count_c"] - 1)
        )
        stats["p_value"] = stats.apply(
            lambda x: ttest_ind_from_stats(
                mean1=x["mean"],
                std1=x["std"],
                nobs1=x["count"],
                mean2=x["mean_c"],
                std2=x["std_c"],
                nobs2=x["count_c"],
                equal_var=False,
            ).pvalue,
            axis=1,
        )
        self.stats = stats

    def summary_table(self, p_threshold: float = 0.05) -> pd.DataFrame:
        """return statistics summary.

        In the future, styling dataframe or add another visualization can be considered.

        Parameters
        ----------
        p_threshold : float, optional
            significance level, by default 0.05

        Returns
        -------
        pd.DataFrame
            stats summary
        """
        if self.stats is None:
            raise ValueError("A/B test statistics haven't been calculated. Please call evaluate() in advance.")

        stats = self.stats.copy(deep=True)
        significance, abs_ci_width, rel_ci_width = self._eval_significance(p_threshold)
        stats["significance"] = significance
        stats["abs_ci_width"] = abs_ci_width
        stats["rel_ci_width"] = rel_ci_width
        for diff_type in ["abs", "rel"]:
            stats[f"ci_{diff_type}_diff"] = stats.apply(lambda x: (x[f"{diff_type}_diff_mean"] - x[f"{diff_type}_ci_width"], x[f"{diff_type}_diff_mean"] + x[f"{diff_type}_ci_width"]), axis=1)
            del stats[f"{diff_type}_ci_width"]
        return_cols = [col for col in stats.columns if col[-2:] != "_c"]
        return stats.loc[:, return_cols]

    def summary_barplot(
        self, p_threshold: float = 0.05, diff_type: str = "rel", display_ci: bool = True
    ) -> plotly.graph_objs.Figure:
        """plot impact and confidence interval for each metric

        Parameters
        ----------
        p_threshold : float, optional
            significance level, by default 0.05

        Returns
        -------
        plotly.graph_objs.Figure
        """
        if self.stats is None:
            raise ValueError("A/B test statistics haven't been calculated. Please call evaluate() in advance.")
        if diff_type not in ["rel", "abs"]:
            raise ValueError("Specified diff type is invalid.")

        stats = self.stats.copy(deep=True)
        significance, abs_ci_width, rel_ci_width = self._eval_significance(p_threshold)
        stats["significant"] = significance
        stats["abs_ci_width"] = abs_ci_width
        stats["rel_ci_width"] = rel_ci_width

        viz_options = {
            "data_frame": stats,
            "x": f"{diff_type}_diff_mean",
            "y": "metric",
            "facet_col": "variant",
            "color": "significant",
            "color_discrete_map": {"up": "#54A24B", "down": "#E45756", "unclear": "silver"},
        }
        if display_ci:
            viz_options["error_x"] = f"{diff_type}_ci_width"

        g = px.bar(**viz_options)
        g.show()
        return g

    def _eval_significance(self, p_threshold: float = 0.05) -> tuple[pd.Series]:
        """evaluate statistical significance & return information related to that.

        Parameters
        ----------
        p_threshold : float, optional
            _description_, by default 0.05

        Returns
        -------
        tuple[pd.Series]
            includes three pd.Series as follows:
            - statistical significance
            - confidence interval width of absolute difference
            - confidence interval width of relative difference
        """

        significance = self.stats.apply(
            lambda x: "up"
            if x["p_value"] <= p_threshold and x["t_value"] > 0
            else "down"
            if x["p_value"] <= p_threshold and x["t_value"] < 0
            else "unclear",
            axis=1,
        )

        abs_ci_width = t.ppf(1 - p_threshold / 2, self.stats["dof"]) * self.stats[f"abs_diff_std"]
        rel_ci_width = t.ppf(1 - p_threshold / 2, self.stats["dof"]) * self.stats[f"rel_diff_std"]

        return significance, abs_ci_width, rel_ci_width
