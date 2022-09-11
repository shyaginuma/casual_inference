import numpy as np
import pandas as pd
import pytest

from casual_inference.dataset import create_sample_ab_result
from casual_inference.statistical_testing import eval_ttest_significance, t_test


@pytest.fixture
def prepare_sample_data() -> pd.DataFrame:
    return create_sample_ab_result(n_variant=4, sample_size=1000000, simulated_lift=[0.01, 0.05, -0.05])


def test_t_test(prepare_sample_data):
    data: pd.DataFrame = prepare_sample_data
    ttest_stats = t_test(data=data, unit_col="rand_unit", variant_col="variant", metrics=["metric_bin", "metric_cont"])

    stats_cols = [
        "variant",
        "metric",
        "mean",
        "var",
        "count",
        "std",
        "stderr",
        "mean_c",
        "var_c",
        "count_c",
        "std_c",
        "stderr_c",
        "abs_diff_mean",
        "abs_diff_std",
        "rel_diff_mean",
        "rel_diff_std",
        "t_value",
        "dof",
        "p_value",
    ]

    assert (ttest_stats.columns == stats_cols).all()
    assert ttest_stats["p_value"].min() > 0.0
    assert ttest_stats["p_value"].max() <= 1.0
    assert (ttest_stats.query("variant == 1")["mean"] == ttest_stats.query("variant == 1")["mean_c"]).all()
    assert (ttest_stats["var"] >= 0).all()
    assert (ttest_stats["count"] >= 0).all()
    assert (ttest_stats["std"] >= 0).all()
    assert (ttest_stats["std"] >= ttest_stats["stderr"]).all()
    assert (ttest_stats["abs_diff_std"] >= 0).all()
    assert (ttest_stats["abs_diff_std"] >= ttest_stats["stderr"]).all()
    assert (ttest_stats["rel_diff_std"] >= 0).all()
    assert (np.sign(ttest_stats["abs_diff_mean"]) == np.sign(ttest_stats["rel_diff_mean"])).all()
    assert (np.sign(ttest_stats["abs_diff_mean"]) == np.sign(ttest_stats["t_value"])).all()


@pytest.mark.parametrize("p_threshold", (0.001, 0.01, 0.05, 0.1))
def test_eval_ttest_significance(prepare_sample_data, p_threshold):
    data: pd.DataFrame = prepare_sample_data
    ttest_stats = t_test(data=data, unit_col="rand_unit", variant_col="variant", metrics=["metric_bin", "metric_cont"])
    significance, abs_ci_width, rel_ci_width = eval_ttest_significance(ttest_stats=ttest_stats, p_threshold=p_threshold)
    ttest_stats["significance"] = significance
    ttest_stats["abs_ci_upper"] = ttest_stats["abs_diff_mean"] + abs_ci_width
    ttest_stats["abs_ci_lower"] = ttest_stats["abs_diff_mean"] - abs_ci_width
    ttest_stats["rel_ci_upper"] = ttest_stats["rel_diff_mean"] + rel_ci_width
    ttest_stats["rel_ci_lower"] = ttest_stats["rel_diff_mean"] - rel_ci_width

    assert significance.isin(["up", "down", "unclear"]).all()
    assert (ttest_stats.query("significance == 'up' or significance == 'down'")["p_value"] <= p_threshold).all()
    assert (ttest_stats.query("significance == 'up'")["abs_ci_lower"] > 0).all()
    assert (ttest_stats.query("significance == 'up'")["rel_ci_lower"] > 0).all()
    assert (ttest_stats.query("significance == 'down'")["abs_ci_upper"] < 0).all()
    assert (ttest_stats.query("significance == 'down'")["rel_ci_upper"] < 0).all()
