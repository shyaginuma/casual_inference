import pandas as pd
import plotly.graph_objs as go
from typing_extensions import Self
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .base import BaseEvaluator


class PSMEvaluator(BaseEvaluator):
    """Evaluate treatment impact by Propensity Score Matching based on the Nearest Neighbor approach.

    Although it automatically fit the model for propensity score prediction, you also can pass your custom model. (Assuming sklearn like model)
    """

    def __init__(self) -> None:
        super().__init__()
        self.ps_model: BaseEstimator = None

    def evaluate(
        self,
        data: pd.DataFrame,
        unit_col: str,
        metrics: list[str],
        variant_col: str = "variant",
        covariates: list[str] = [],
        custom_ps_model: BaseEstimator = None,
        train_ps_model: bool = True,
    ) -> Self:
        """Predict propensity score and do matching, then store matched dataset into the evaluator.
        It used the Logistic Regression as the propensity score model, but you can also specify your custom model.

        Parameters
        ----------
        variant_col : str, optional
            stores binary values indicate if the sample received treatment or not, by default "variant"
        covariates : list[str], optional
            column names of covariate variables you want to use for the analysis, by default []
        custom_ps_model : BaseEstimator, optional
            If you want to use a model other than the default one for propensity score prediction, you can pass the model here.
            The model should have scikit-learn like interface. (e.g., has fit(), predict_proba(), ...)
        train_ps_model : bool, optional
            Whether to do training during the call of this method. If you pass the model that have already been trained, you can set this flag False, by default True

        Returns
        -------
        self : object
            Evaluator storing statistics calculated.
        """
        self._validate_passed_data(data, unit_col, metrics)
        if data[variant_col].min() != 0 or data[variant_col].max() != 1:
            raise ValueError("value in variant_col should be binary")

        # build propensity score model and make prediction
        if custom_ps_model:
            self.ps_model = custom_ps_model
        else:
            self.ps_model = LogisticRegression()

        if train_ps_model:
            covar_scaled = StandardScaler().fit_transform(data[covariates])
            self.ps_model.fit(covar_scaled, data[variant_col])
        ps = self.ps_model.predict_proba(covar_scaled)

        # do matching
        # TODO

        return self

    def summary_table(self) -> pd.DataFrame:
        return super().summary_table()

    def summary_plot(self) -> go.Figure:
        return super().summary_plot()

    def summary_ps_model(self):
        pass
