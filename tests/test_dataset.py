from casual_inference.dataset import sample_abtest


def test_create_sample_ab_result():
    n_variant = 4
    sample_size = 1000
    sample_data = sample_abtest.create_sample_ab_result(
        n_variant=n_variant, sample_size=sample_size, metric_base=0.01, simulated_lift=[0.1, 0.2, 0.3]
    )

    assert sample_data["variant"].nunique() == n_variant
    assert sample_data.shape[0] == sample_size
    assert sample_data["metric_bin"].min() == 0
    assert sample_data["metric_bin"].max() == 1
    assert sample_data["metric_cont"].min() == 0
    assert sample_data["metric_cont"].max() >= 1
