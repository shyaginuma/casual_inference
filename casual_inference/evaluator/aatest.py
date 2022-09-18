import numpy as np
import pandas as pd
import plotly
import plotly.express as px
from scipy.stats import kstest

from casual_inference.statistical_testing import t_test


class AATestEvaluator:
    def __init__(self, n_simulation: int = 1000, sample_rate: float = 1.0) -> None:
        """initialize parameters affect result of evaluation.

        Parameters
        ----------
        n_simulation : float, optional
            How many times you want to do simulated A/A test, by default 1000
        sample_rate : float, optional
            How much fraction you want to sample from the dataframe, by default 1.0
        """
        if n_simulation <= 0:
            raise ValueError("The number of simulation should be positive number.")
        if not (sample_rate > 0.0 and sample_rate <= 1.0):
            raise ValueError("The sample rate should be in (0, 1]")

        self.n_simulation = n_simulation
        self.sample_rate = sample_rate
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
        sampled_df = data.copy(deep=True)
        for i in range(self.n_simulation):
            if self.sample_rate < 1.0:
                sampled_df = data.sample(frac=self.sample_rate, random_state=i, ignore_index=True)
            sampled_df["variant"] = np.random.randint(
                low=1, high=3, size=sampled_df.shape[0]
            )  # The upper bound is exclusive
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

    def summary_table(self) -> pd.DataFrame:
        if self.stats is None:
            raise ValueError("A/B test statistics haven't been calculated. Please call evaluate() in advance.")

        stats_agg = (
            self.stats.groupby("metric")[["abs_diff_mean", "rel_diff_mean", "p_value"]]
            .agg(["mean", "std"])
            .reset_index()
        )
        stats_agg["ksstat"] = 0
        stats_agg["ks_pvalue"] = 0
        for metric in self.stats["metric"].unique():
            result = kstest(self.stats.query(f"metric == '{metric}'")["p_value"], "uniform")
            stats_agg.loc[stats_agg["metric"] == metric, "ksstat"] = result.statistic
            stats_agg.loc[stats_agg["metric"] == metric, "ks_pvalue"] = result.pvalue
        stats_agg["significance"] = stats_agg["ks_pvalue"].map(lambda x: True if x < 0.05 else False)
        return stats_agg

    def summary_plot(self) -> plotly.graph_objs.Figure:
        if self.stats is None:
            raise ValueError("A/B test statistics haven't been calculated. Please call evaluate() in advance.")

        stats = self.stats.copy(deep=True)
        stats["ks_pvalue"] = 0
        for metric in stats["metric"].unique():
            result = kstest(stats.query(f"metric == '{metric}'")["p_value"], "uniform")
            stats.loc[stats["metric"] == metric, "ks_pvalue"] = result.pvalue
        stats["significance"] = stats["ks_pvalue"].map(lambda x: True if x < 0.05 else False)

        g = px.histogram(
            data_frame=stats,
            x="p_value",
            facet_col="metric",
            nbins=100,
            histnorm="probability",
            color="significance",
            color_discrete_map={False: "#54A24B", True: "#E45756"},
        )
        g.add_hline(y=0.01, line_dash="dot", line_width=2)
        return g
