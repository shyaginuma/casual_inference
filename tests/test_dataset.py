import pandas.api.types as pd_types

from casual_inference.dataset import create_sample_ab_result, create_sample_biased


def test_create_sample_ab_result():
    n_variant = 4
    sample_size = 1000

    sample_data = create_sample_ab_result(
        n_variant=n_variant, sample_size=sample_size, metric_base=0.01, simulated_lift=[0.1, 0.2, 0.3]
    )

    assert sample_data["variant"].nunique() == n_variant
    assert sample_data.shape[0] == sample_size
    assert sample_data["metric_bin"].min() == 0
    assert sample_data["metric_bin"].max() == 1
    assert sample_data["metric_cont"].min() == 0
    assert sample_data["metric_cont"].max() >= 1
    assert pd_types.is_string_dtype(sample_data["segment_str"])
    assert pd_types.is_numeric_dtype(sample_data["segment_numer"])


def test_create_sample_biased():
    sample_size = 1000

    sample_data = create_sample_biased(sample_size=sample_size)
    print(sample_data.head())

    assert sample_data.shape[0] == sample_size
    assert set(sample_data["treatment"].unique()) == set([0, 1])
    assert pd_types.is_string_dtype(sample_data["covar_cat"])
    assert pd_types.is_numeric_dtype(sample_data["covar_numer1"])
    assert pd_types.is_numeric_dtype(sample_data["covar_numer2"])
    assert pd_types.is_numeric_dtype(sample_data["target"])
