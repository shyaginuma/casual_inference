import numpy as np
import pandas as pd
import pytest

from casual_inference.dataset import sample_abtest
from casual_inference.evaluator import ABTestEvaluator


@pytest.fixture
def generate_sample_data() -> pd.DataFrame:
    return sample_abtest.create_sample_ab_result(n_variant=4, sample_size=1000000, simulated_lift=[0.01, 0.05, -0.05])


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


class TestABTestEvaluator:
    def prepare_evaluator(self):
        return ABTestEvaluator()

    def test_evaluate(self, generate_sample_data):
        evaluator = self.prepare_evaluator()
        evaluator.evaluate(
            generate_sample_data, unit_col="rand_unit", variant_col="variant", metrics=["metric_bin", "metric_cont"]
        )
        assert (evaluator.stats.columns == stats_cols).all()
        assert evaluator.stats["p_value"].min() > 0.0
        assert evaluator.stats["p_value"].max() <= 1.0

        assert (evaluator.stats.query("variant == 1")["mean"] == evaluator.stats.query("variant == 1")["mean_c"]).all()
        assert (evaluator.stats["var"] >= 0).all()
        assert (evaluator.stats["count"] >= 0).all()
        assert (evaluator.stats["std"] >= 0).all()
        assert (evaluator.stats["std"] >= evaluator.stats["stderr"]).all()
        assert (evaluator.stats["abs_diff_std"] >= 0).all()
        assert (evaluator.stats["abs_diff_std"] >= evaluator.stats["stderr"]).all()
        assert (evaluator.stats["rel_diff_std"] >= 0).all()
        assert (np.sign(evaluator.stats["abs_diff_mean"]) == np.sign(evaluator.stats["rel_diff_mean"])).all()
        assert (np.sign(evaluator.stats["abs_diff_mean"]) == np.sign(evaluator.stats["t_value"])).all()
