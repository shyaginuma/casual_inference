import numpy as np
import pytest

from casual_inference.dataset import sample_abtest
from casual_inference.evaluator import ABTestEvaluator
from casual_inference.evaluator.aatest import AATestEvaluator


@pytest.fixture
def prepare_abtest_evaluator() -> ABTestEvaluator:
    sample_data = sample_abtest.create_sample_ab_result(
        n_variant=4, sample_size=1000000, simulated_lift=[0.01, 0.05, -0.05]
    )
    evaluator = ABTestEvaluator()
    evaluator.evaluate(sample_data, unit_col="rand_unit", variant_col="variant", metrics=["metric_bin", "metric_cont"])
    return evaluator


@pytest.fixture
def prepare_aatest_evaluator() -> AATestEvaluator:
    sample_data = sample_abtest.create_sample_ab_result(
        n_variant=4, sample_size=1000000, simulated_lift=[0.01, 0.05, -0.05]
    )
    evaluator = AATestEvaluator(n_simulation=10, n_variant=4)
    evaluator.evaluate(sample_data, unit_col="rand_unit", metrics=["metric_bin", "metric_cont"])
    return evaluator


class TestABTestEvaluator:
    def test_evaluate(self, prepare_abtest_evaluator):
        evaluator: ABTestEvaluator = prepare_abtest_evaluator
        assert evaluator.stats is not None

    @pytest.mark.parametrize("p_threshold", (0.01, 0.05, 0.1))
    def test_summary_table(self, p_threshold, prepare_abtest_evaluator):
        evaluator: ABTestEvaluator = prepare_abtest_evaluator
        summary = evaluator.summary_table(p_threshold=p_threshold)

        assert "significance" in summary.columns
        assert "ci_abs_diff" in summary.columns
        assert "ci_rel_diff" in summary.columns
        assert summary.query(f"p_value <= {p_threshold}")["significance"].isin(["up", "down"]).all()
        assert (
            summary.query(f"p_value <= {p_threshold}")["ci_abs_diff"]
            .map(lambda x: True if x[1] < 0 or x[0] > 0 else False)
            .all()
        )
        assert (
            summary.query(f"p_value <= {p_threshold}")["ci_rel_diff"]
            .map(lambda x: True if x[1] < 0 or x[0] > 0 else False)
            .all()
        )

    @pytest.mark.parametrize("diff_type", ("rel", "abs"))
    def test_summary_barplot(self, diff_type, prepare_abtest_evaluator):
        evaluator: ABTestEvaluator = prepare_abtest_evaluator
        evaluator.summary_barplot(diff_type=diff_type)


class TestAATestEvaluator:
    def test_evaluate(self, prepare_aatest_evaluator):
        evaluator: AATestEvaluator = prepare_aatest_evaluator
        print(evaluator.stats)
