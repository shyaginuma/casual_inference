import numpy as np
import pandas as pd
import plotly
import plotly.express as px

from casual_inference.statistical_testing import t_test


class AATestEvaluator:
    def __init__(self, n_simulation: int = 1000, n_variant: int = 2) -> None:
        """initialize parameters affect result of evaluation.

        Parameters
        ----------
        n_simulation : float, optional
            How many times you want to do simulated A/A test, by default 1000
        n_variant : int, optional
            How many variants you want to simulate, by default 2
        """
        self.n_simulation = n_simulation
        self.n_variant = n_variant
        self.stats: pd.DataFrame = None

    def evaluate(self, data: pd.DataFrame, unit_col: str, metrics: list[str]) -> None:
        """split data n times, and calculate statistics n times, then store it into a table

        Parameters
        ----------
        data : pd.DataFrame
            Dataframe has randomization unit column, and metrics columns. The data should have been aggregated by the randmization unit.
        unit_col : str
            A column name stores the randomization unit. something like user_id, session_id, ...
        metrics : list[str]
            Columns stores metrics you want to evaluate.
        """
        result = pd.DataFrame()
        for i in range(self.n_simulation):
            sampled_df = data.sample(frac=2 / self.n_variant, random_state=i, ignore_index=True)
            sampled_df["variant"] = np.random.randint(low=1, high=2, size=sampled_df.shape[0])
            stats = t_test(sampled_df, unit_col, "variant", metrics)
            stats["idx"] = i
            result = pd.concat([result, stats])
        return_cols = [
            "idx",
            "metric",
            "mean",
            "count",
            "stderr",
            "abs_diff_mean",
            "abs_diff_std",
            "rel_diff_mean",
            "p_value",
        ]
        result = result.query("variant == 2").loc[:, return_cols].reset_index(drop=True)
        self.stats = result

    def summary(self) -> plotly.graph_objs.Figure:
        if self.stats is None:
            raise ValueError("A/B test statistics haven't been calculated. Please call evaluate() in advance.")

        g = px.histogram(data_frame=self.stats, x="p_value", facet_col="metric", nbins=100)
        return g
