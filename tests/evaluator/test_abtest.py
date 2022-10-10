from multiprocessing.spawn import prepare
import pandas as pd
import pytest

from casual_inference.dataset import create_sample_ab_result
from casual_inference.evaluator import ABTestEvaluator

@pytest.fixture
def prepare_sample_data() -> pd.DataFrame:
    return create_sample_ab_result(n_variant=4, sample_size=1000000, simulated_lift=[0.01, 0.05, -0.05])


@pytest.fixture
def prepare_sample_data_extream() -> pd.DataFrame:
    sample_data = create_sample_ab_result(n_variant=4, sample_size=1000000)

    # double sample size of control group
    additional_data = sample_data.query("variant == 1")
    additional_data["rand_unit"] = additional_data["rand_unit"] + 1000000
    sample_data = pd.concat([sample_data, additional_data])
    return sample_data


@pytest.fixture
def prepare_abtest_evaluator(prepare_sample_data) -> ABTestEvaluator:
    sample_data = prepare_sample_data
    return ABTestEvaluator().evaluate(sample_data, unit_col="rand_unit", variant_col="variant", metrics=["metric_bin", "metric_cont"])


@pytest.fixture
def prepare_abtest_evaluator_with_extream_data(prepare_sample_data_extream) -> ABTestEvaluator:
    sample_data = prepare_sample_data_extream
    return ABTestEvaluator().evaluate(sample_data, unit_col="rand_unit", variant_col="variant", metrics=["metric_bin", "metric_cont"])


class TestABTestEvaluator:
    def test_evaluate(self, prepare_abtest_evaluator):
        evaluator: ABTestEvaluator = prepare_abtest_evaluator
        assert evaluator.stats is not None

    @pytest.mark.parametrize("segment", ("segment_str", "segment_numer"))
    def test_evaluate_segment(self, prepare_sample_data, segment):
        sample_data: pd.DataFrame = prepare_sample_data
        evaluator = ABTestEvaluator().evaluate(sample_data, unit_col="rand_unit", variant_col="variant", metrics=["metric_bin", "metric_cont"], segment_col=segment)
        stats = evaluator.stats

        assert segment in stats.columns
        assert stats["count"].sum() == sample_data.shape[0]

        if segment == "segment_str":
            assert sample_data[segment].nunique() == stats[segment].nunique()
        else:
            assert stats[segment].nunique() == 5

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

    def test_diagnose_srm(self, prepare_abtest_evaluator_with_extream_data):
        evaluator: ABTestEvaluator = prepare_abtest_evaluator_with_extream_data
        results = evaluator._diagnose_srm()

        assert len(results) == evaluator.stats["variant"].nunique() - 1
        assert len([result for result in results if result.significant]) == evaluator.stats["variant"].nunique() - 1
