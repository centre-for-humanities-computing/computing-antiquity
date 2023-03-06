from typing import List

import pymc as pm
from pymc.sampling_jax import sample_blackjax_nuts
import arviz as az
import pandas as pd
import numpy as np


def broadcast(a: np.ndarray, n_levels: int):
    return np.repeat(
        np.atleast_2d(a),
        n_levels,
        axis=0,
    ).T


class BayesianLogisticRegression:
    def __init__(
        self, data: pd.DataFrame, slope_prior_sd=10, intercept_prior_sd=10
    ):
        self.feature_names = [
            column for column in data.columns if column != "outcome"
        ]
        outcome_values, self.outcome_levels = pd.factorize(data["outcome"])
        coords = {
            "observations": np.arange(len(data.index)),
            "outcome_levels": self.outcome_levels,
        }
        with pm.Model(coords=coords) as model:
            beta_0 = pm.Normal(
                "intercept",
                mu=0,
                sigma=intercept_prior_sd,
                dims="outcome_levels",
            )
            features = []
            betas = []
            for feature_name in self.feature_names:
                broadcasted_feature = broadcast(
                    data[feature_name], self.outcome_levels.shape[0]
                )
                feature = pm.MutableData(feature_name, broadcasted_feature)
                beta = pm.Normal(
                    f"slope_{feature_name}",
                    mu=0,
                    sigma=slope_prior_sd,
                    dims="outcome_levels",
                )
                features.append(feature)
                betas.append(beta)
            log_odds = beta_0
            for feature, beta in zip(features, betas):
                log_odds += beta * feature
            pm.Categorical(
                "outcome",
                logit_p=log_odds,
                observed=outcome_values,
                shape=feature.shape[0],
            )
        self.model = model

    def prior_pred_check(self):
        with self.model:
            prior_pred = pm.sample_prior_predictive()
            az.plot_ppc(prior_pred, group="prior")

    def sample(self):
        with self.model:
            self.trace = pm.sample()
            self.trace.extend(pm.sample_prior_predictive())
            self.trace.extend(pm.sample_posterior_predictive(self.trace))

    def predict(self, data: pd.DataFrame):
        data_dict = data.to_dict("list")
        data_dict.pop("outcome")
        n_levels = self.outcome_levels.shape[0]
        data_dict = {
            feature_name: broadcast(data_dict[feature_name], n_levels)
            for feature_name in data_dict
        }
        with self.model:
            pm.set_data(data_dict)
            res = pm.sample_posterior_predictive(self.trace)
        return res
