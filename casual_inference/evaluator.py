import numpy as np
import pandas as pd
from scipy.stats import ttest_ind_from_stats


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
        stats = stats.merge(stats.query(f"{variant_col} == 1"), on="metric", suffixes=["", "_c"]).drop(
            "variant_c", axis=1
        )

        stats["abs_diff_mean"] = stats["mean"] - stats["mean_c"]
        stats["abs_diff_std"] = np.sqrt(
            np.power(stats["std"], 2) / stats["count"] + np.power(stats["std_c"], 2) / stats["count_c"]
        )
        stats["rel_diff_mean"] = stats["mean"] / stats["mean_c"] - 1

        # see: https://arxiv.org/pdf/1803.06336.pdf
        stats["rel_diff_std"] = np.sqrt(
            (
                stats["var"] / stats["count"]
                + (stats["mean"] ** 2 / stats["mean_c"] ** 2) * stats["var_c"] / stats["count_c"]
            )
            / stats["mean_c"]
        )
        stats["t_value"] = stats["abs_diff_mean"] / stats["abs_diff_std"]
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

    def summary(self) -> None:
        if self.stats is None:
            raise ValueError("A/B test statistics haven't been calculated. Please call evaluate() in advance.")
        pass
