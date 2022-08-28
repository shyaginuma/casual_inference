import numpy as np
import pandas as pd


def create_sample_ab_result(
    n_variant: int = 2, sample_size: int = 1000, metric_base: float = 0.1, simulated_lift: list[float] = []
) -> pd.DataFrame:
    """
    Returns
    -------
    pd.DataFrame
        Sample A/B test result contains columns as follows
            - rand_unit: randomization unit. something like user_id, session_id, ...
            - variant: which pattern the unit was assigned. assuming 1 is the control group.
            - metric_bin: a binary metric assuming the target metric of the A/B test. e.g., Click, Purchase, ...
            - metric_cont: a continuous metric assuming the target metric of the A/B test. e.g., Clicks, Purchases, ...
    """
    if n_variant <= 1:
        raise ValueError("n_variant should be more than or equal 2.")
    if len(simulated_lift) >= 1 and len(simulated_lift) != n_variant - 1:
        raise ValueError("The length of simulated_lift doesn't equal to the number of treatment variants.")

    ab_result = pd.DataFrame()
    ab_result["rand_unit"] = [i for i in range(sample_size)]
    ab_result["variant"] = np.random.choice(a=[i + 1 for i in range(n_variant)], size=sample_size)
    ab_result["metric_bin"] = 0
    ab_result["metric_cont"] = 0

    if len(simulated_lift) == 0:
        simulated_lift = [0.05 * i for i in range(n_variant) if i != 0]
    simulated_lift.insert(0, 0)  # insert lift of control variant for convenience

    for i in range(n_variant):
        p = metric_base * (1 + simulated_lift[i])
        size = ab_result.loc[ab_result["variant"] == i + 1].shape[0]
        ab_result.loc[ab_result["variant"] == i + 1, "metric_bin"] = np.random.binomial(n=1, p=p, size=size)
        ab_result.loc[ab_result["variant"] == i + 1, "metric_cont"] = np.random.poisson(lam=p * 10, size=size)
    return ab_result
