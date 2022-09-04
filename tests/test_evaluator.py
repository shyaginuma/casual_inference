import numpy as np
import pytest

from casual_inference.dataset import sample_abtest
from casual_inference.evaluator import ABTestEvaluator


@pytest.fixture
def prepare_evaluator() -> ABTestEvaluator:
    sample_data = sample_abtest.create_sample_ab_result(
        n_variant=4, sample_size=1000000, simulated_lift=[0.01, 0.05, -0.05]
    )
    evaluator = ABTestEvaluator()
    evaluator.evaluate(sample_data, unit_col="rand_unit", variant_col="variant", metrics=["metric_bin", "metric_cont"])
    return evaluator


class TestABTestEvaluator:
    def prepare_evaluator(self):
        return ABTestEvaluator()

    def test_evaluate(self, prepare_evaluator):
        evaluator = prepare_evaluator

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

    @pytest.mark.parametrize("p_threshold", (0.01, 0.05, 0.1))
    def test_summary_table(self, p_threshold, prepare_evaluator):
        evaluator: ABTestEvaluator = prepare_evaluator
        summary = evaluator.summary_table(p_threshold=p_threshold)

        assert "significance" in summary.columns
        assert "ci_abs_diff" in summary.columns
        assert "ci_rel_diff" in summary.columns
        assert summary.query(f"p_value <= {p_threshold}")["significance"].isin(["up", "down"]).all()
        assert summary.query(f"p_value <= {p_threshold}")["ci_abs_diff"].map(lambda x: True if x[1] < 0 or x[0] > 0 else False).all()
        assert summary.query(f"p_value <= {p_threshold}")["ci_rel_diff"].map(lambda x: True if x[1] < 0 or x[0] > 0 else False).all()


    @pytest.mark.parametrize("diff_type", ("rel", "abs"))
    def test_summary_barplot(self, diff_type, prepare_evaluator):
        evaluator: ABTestEvaluator = prepare_evaluator
        evaluator.summary_barplot(diff_type=diff_type)
