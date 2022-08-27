from multiprocessing.sharedctypes import Value
import pandas as pd


class ABTestEvaluator:
    def __init__(self) -> None:
        self.stats: pd.DataFrame = None

    def evaluate(self, data: pd.DataFrame, unit_col: str, variant_col: str, metrics: list[str]) -> None:
        """calculate stats of A/B test and cache it into the class variable.

        Parameters
        ----------
        data : pd.DataFrame
            Dataframe has randomization unit column, variant assignment column, and metrics columns.
            The data should have been aggregated by the randmization unit.
        unit_col : str
            A column name stores the randomization unit. something like user_id, session_id, ...
        variant_col : str
            A column name stores the variant assignment.
            The control variant should have value 1.
        metrics : list[str]
            Columns stores metrics you want to evaluate.
        """
        if data.shape[0] != data[unit_col].nunique():
            raise ValueError("passed dataframe hasn't been aggregated by the randomization unit.")

        if data[variant_col].min() != 1:
            raise ValueError("the control variant seems not to exist.")

        if len(metrics) == 0:
            raise ValueError("metrics hasn't been specified.")

        stats = data.groupby(variant_col)[metrics].agg(["mean", "var"])
        stats["sample_size"] = data.groupby(variant_col)[unit_col].count()
        stats = stats.merge(stats.query(f"{variant_col} == 1"), suffixes=["", "_c"], how="cross")

        for metric in metrics:
            stats[f"abs_diff_{metric}", "mean"] = stats[metric]["mean"] - stats[f"{metric}_c"]["mean"]
            stats[f"rel_diff_{metric}", "mean"] = stats[metric]["mean"]/stats[f"{metric}_c"]["mean"]
        self.stats = stats

    def summary(self) -> None:
        if self.stats == None:
            raise ValueError("A/B test statistics haven't been calculated. Please call evaluate() in advance.")
        pass
