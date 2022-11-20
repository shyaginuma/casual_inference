import numpy as np
import pandas as pd


def create_sample_biased(sample_size: int = 1000) -> pd.DataFrame:
    """Generate sample biased dataset for testing advanced causal inference approach. (e.g., linear regression, propensity score matching, ...)

    Parameters
    ----------
    sample_size : int, optional
        number of samples you want, by default 1000

    Returns
    -------
    pd.DataFrame
        Sample biased experiment result contains columns as follows
        - unit: something like user_id, session_id, ...
        - covar_numer1: dummy covariate variable (numerical)
        - covar_numer2: dummy covariate variable (numerical)
        - covar_cat: dummy covariate variable (categorical)
        - treatment: indicator variable that each unit received treatment or not
        - target: dummy target variable, interested in the treatment impact on
    """
    result = pd.DataFrame()
    result["unit"] = [i for i in range(sample_size)]
    result["covar_numer1"] = np.random.normal(size=sample_size)
    result["covar_numer2"] = np.random.exponential(size=sample_size)
    result["covar_cat"] = np.random.choice(a=[str(i + 1) for i in range(5)], size=sample_size)

    # generate probability of receiving treatment by subjective relationships
    relationship = (
        np.random.normal(loc=-1.0, scale=2.0, size=sample_size)
        + np.random.normal(loc=-2.0, scale=1.0, size=sample_size) * result["covar_numer1"]
        + np.random.normal(loc=5.0, scale=5.0, size=sample_size) * result["covar_numer2"]
    )
    treatment_prob = 1 / (1 + np.exp(-1.0 * relationship))
    result["treatment"] = (treatment_prob >= 0.5).astype("int")

    # generate target by subjective relationships
    result["target"] = (
        np.random.normal(loc=3.0, scale=5.0, size=sample_size)
        + np.random.normal(loc=1.0, scale=1.0, size=sample_size) * result["treatment"]
        + np.random.normal(loc=-0.5, scale=1.0, size=sample_size) * result["covar_numer1"]
        + np.random.normal(loc=2.0, scale=2.0, size=sample_size) * result["covar_numer2"]
    )

    return result
