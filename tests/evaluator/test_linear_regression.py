import pandas as pd
import pytest

from casual_inference.dataset import create_sample_biased
from casual_inference.evaluator import LinearRegressionEvaluator


@pytest.fixture
def prepare_sample_data() -> pd.DataFrame:
    return create_sample_biased(sample_size=1000000)


@pytest.fixture
def prepare_evaluator(prepare_sample_data) -> LinearRegressionEvaluator:
    sample_data = prepare_sample_data
    return LinearRegressionEvaluator().evaluate(
        sample_data,
        unit_col="unit",
        treatment_col="treatment",
        metrics=["target"],
        covariates=["covar_numer1", "covar_numer2", "covar_cat"],
    )


class TestABTestEvaluator:
    def test_evaluate(self, prepare_evaluator):
        evaluator: LinearRegressionEvaluator = prepare_evaluator
        assert evaluator.stats is not None
        assert len(evaluator.models) > 0

    @pytest.mark.parametrize("p_threshold", (0.01, 0.05, 0.1))
    def test_summary_table(self, p_threshold, prepare_evaluator):
        evaluator: LinearRegressionEvaluator = prepare_evaluator
        summary = evaluator.summary_table(p_threshold=p_threshold)

        expected_columns = {
            "coef",
            "std err",
            "t",
            "P>|t|",
            "[0.025",
            "0.975]",
            "metric",
        }
        assert expected_columns == set(summary.columns)
        assert set(evaluator.models.keys()) == set(summary["metric"].unique())
        assert (summary["std err"] > 0).all()
        assert (summary["P>|t|"] > 0).all()

    def test_summary_plot(self, prepare_evaluator):
        evaluator: LinearRegressionEvaluator = prepare_evaluator
        g = evaluator.summary_plot()
        g.show()
