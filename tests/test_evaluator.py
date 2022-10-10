import pandas as pd
import pytest

from casual_inference.dataset import create_sample_ab_result
from casual_inference.evaluator import (
    AATestEvaluator,
    SampleSizeEvaluator,
)


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

    @pytest.mark.parametrize("target_mde", (0.03, 0.05, 0.1))
    def test_summary_table(self, target_mde, prepare_samplesize_evaluator):
        evaluator: SampleSizeEvaluator = prepare_samplesize_evaluator
        summary = evaluator.summary_table(target_mde=target_mde)
        assert summary.shape[0] == evaluator.stats["metric"].nunique()
        assert (summary["mde_rel"] < target_mde).all()

    def test_summary_plot(self, prepare_samplesize_evaluator):
        evaluator: SampleSizeEvaluator = prepare_samplesize_evaluator
        g = evaluator.summary_plot(target_mde=0.03)
        g.show()
