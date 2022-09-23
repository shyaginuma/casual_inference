import numpy as np
import pandas as pd
from scipy.stats import t, ttest_ind_from_stats


def t_test(data: pd.DataFrame, unit_col: str, variant_col: str, metrics: list[str]) -> pd.DataFrame:
    """_summary_

    Parameters
    ----------
    data : pd.DataFrame
        A DataFrame has randomization unit column, variant assignment column, and metrics columns.
        The data should have been aggregated by the randomization unit.
    unit_col : str
        A column name stores the randomization unit. something like user_id, session_id, ...
    variant_col : str
        A column name stores the variant assignment.
        The control variant should have value 1.
    metrics : list[str]
        Columns stores metrics you want to evaluate.

    Returns
    -------
    pd.DataFrame
        A DataFrame stores basic statistics of each variant (mean, variance, ...) and impact compared with control group.
    """
    if data.shape[0] != data[unit_col].nunique():
        raise ValueError("passed dataframe hasn't been aggregated by the randomization unit.")
    if len(metrics) == 0:
        raise ValueError("metrics hasn't been specified.")
    if data[variant_col].min() != 1:
        raise ValueError("the control variant seems not to exist.")
    means = (
        data.groupby(variant_col)[metrics].mean().stack().reset_index().rename(columns={"level_1": "metric", 0: "mean"})
    )
    vars = (
        data.groupby(variant_col)[metrics].var().stack().reset_index().rename(columns={"level_1": "metric", 0: "var"})
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
    stats = stats.merge(stats.query(f"{variant_col} == 1"), on="metric", suffixes=["", "_c"]).drop("variant_c", axis=1)

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
    return stats


def eval_ttest_significance(
    ttest_stats: pd.DataFrame, p_threshold: float = 0.05
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """evaluate statistical significance & return information related to that.

    Parameters
    ----------
    ttest_stats : pd.DataFrame
        calculated statistics in t_test() function.
    p_threshold : float, optional
        significant level, by default 0.05

    Returns
    -------
    tuple[pd.Series]
        includes three pd.Series as follows:
        - statistical significance
        - confidence interval width of absolute difference
        - confidence interval width of relative difference
    """
    necessary_cols = ["p_value", "t_value", "dof", "abs_diff_std", "rel_diff_std"]
    for col in necessary_cols:
        if col in ttest_stats.columns:
            continue
        raise ValueError(f"Necessary column does not exist: {col}")

    significance = ttest_stats.apply(
        lambda x: "up"
        if x["p_value"] <= p_threshold and x["t_value"] > 0
        else "down"
        if x["p_value"] <= p_threshold and x["t_value"] < 0
        else "unclear",
        axis=1,
    )

    abs_ci_width = t.ppf(1 - p_threshold / 2, ttest_stats["dof"]) * ttest_stats["abs_diff_std"]
    rel_ci_width = t.ppf(1 - p_threshold / 2, ttest_stats["dof"]) * ttest_stats["rel_diff_std"]

    return significance, abs_ci_width, rel_ci_width
