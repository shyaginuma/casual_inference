import pytest

from casual_inference.dataset import create_sample_ab_result
from casual_inference.evaluator import (
    AATestEvaluator,
    ABTestEvaluator,
    SampleSizeEvaluator,
)


@pytest.fixture
def prepare_abtest_evaluator() -> ABTestEvaluator:
    sample_data = create_sample_ab_result(n_variant=4, sample_size=1000000, simulated_lift=[0.01, 0.05, -0.05])
    evaluator = ABTestEvaluator()
    evaluator.evaluate(sample_data, unit_col="rand_unit", variant_col="variant", metrics=["metric_bin", "metric_cont"])
    return evaluator


@pytest.fixture
def prepare_aatest_evaluator() -> AATestEvaluator:
    sample_data = create_sample_ab_result(n_variant=2, sample_size=1000000, simulated_lift=[0.0])
    evaluator = AATestEvaluator(n_simulation=10)
    evaluator.evaluate(sample_data, unit_col="rand_unit", metrics=["metric_bin", "metric_cont"])
    return evaluator


@pytest.fixture
def prepare_samplesize_evaluator() -> SampleSizeEvaluator:
    sample_data = create_sample_ab_result(n_variant=2, sample_size=1000000, simulated_lift=[0.0])
    evaluator = SampleSizeEvaluator()
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
    def test_summary_plot(self, diff_type, prepare_abtest_evaluator):
        evaluator: ABTestEvaluator = prepare_abtest_evaluator
        g = evaluator.summary_plot(diff_type=diff_type)
        g.show()


class TestAATestEvaluator:
    def test_evaluate(self, prepare_aatest_evaluator):
        evaluator: AATestEvaluator = prepare_aatest_evaluator
        assert evaluator.stats is not None

    def test_summary_table(self, prepare_aatest_evaluator):
        evaluator: AATestEvaluator = prepare_aatest_evaluator
        summary = evaluator.summary_table()
        assert "metric" in summary.columns
        assert "abs_diff_mean" in summary.columns
        assert "rel_diff_mean" in summary.columns
        assert "p_value" in summary.columns
        assert "ksstat" in summary.columns
        assert "ks_pvalue" in summary.columns
        assert "significance" in summary.columns
        assert (summary["ks_pvalue"] > 0).all()
        assert (summary["ksstat"] > 0).all()

    def test_summary_plot(self, prepare_aatest_evaluator):
        evaluator: AATestEvaluator = prepare_aatest_evaluator
        g = evaluator.summary_plot()
        g.show()


class TestSampleSizeEvaluator:
    def test_evaluate(self, prepare_samplesize_evaluator):
        evaluator: SampleSizeEvaluator = prepare_samplesize_evaluator
        stats = evaluator.stats
        assert (
            stats.columns == ["threshold", "metric", "mean", "var", "count", "sample_size", "mde_abs", "mde_rel"]
        ).all()
        assert stats["threshold"].max() == 1.0
        assert stats["threshold"].min() == 0.01
        assert (stats["metric"].unique() == ["metric_bin", "metric_cont"]).all()
        assert stats["mean"].min() > 0
        assert stats["var"].min() > 0
        assert stats["count"].min() > 0
        assert stats["sample_size"].min() > 0
        assert stats["mde_abs"].min() > 0
        assert stats["mde_rel"].max() <= 1.0 or stats["mde_rel"].min() > 0.0
