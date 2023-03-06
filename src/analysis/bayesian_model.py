# %%
import os
import subprocess

import arviz as az
import datapane as dp
import numpy as np
import pandas as pd
import plotly.express as px
import pymc as pm
from IPython import get_ipython
from scipy.stats import zscore
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from utils.bayesian import BayesianLogisticRegression, broadcast
from utils.metadata import fetch_metadata
from utils.files import prepare_stylistic_features

# %% Setting up autoreload
try:
    ipython = get_ipython()
    ipython.magic("load_ext autoreload")
    ipython.magic("autoreload 2")
except Exception:
    print("Could not load Ipython kernel, continuing without autoreload")

# %% Datapane login
subprocess.call(["bash", "datapane_login.sh"])
datapane_elements = []
project_name = "Bayesian Logistic Regression"

# %%
data = prepare_stylistic_features()
data

# %%
train_data = data[~data.outcome.isin(["Mark", "NT Narratives"])]
train_data.outcome.unique()

# %%
model = BayesianLogisticRegression(train_data)

# %% Prior predictive check
model.prior_pred_check()
# %% Model fitting
model.sample()

# %% Save trace
model.trace.to_netcdf(
    "/work/data_wrangling/dat/greek/results/bayesian_trace_wo_nt.netcdf"
)

# %% Load trace
model = BayesianLogisticRegression(train_data)
model.trace = az.InferenceData.from_netcdf(
    "/work/data_wrangling/dat/greek/results/bayesian_trace_wo_nt.netcdf"
)

# %% Posterior predictive check
fig = az.plot_ppc(model.trace, group="posterior")
# datapane_elements.append(dp.Text("## Posterior predictive check"))
# datapane_elements.append(dp.Plot(fig))


# %% Traceplot
fig = az.plot_trace(model.trace)
# datapane_elements.append(dp.Text("## Trace plot"))
# datapane_elements.append(dp.Plot(fig))

# %% Model summary
summary = az.summary(model.trace, var_names="slope", filter_vars="like")
significant_effects = summary[
    ~((summary["hdi_3%"] < 0) & (summary["hdi_97%"] > 0))
]
significant_effects
# datapane_elements.append(dp.Text("## Model summary"))
# datapane_elements.append(dp.DataTable(summary))
# %% Posterior predictive for Mark
post_pred = model.predict(data[data.outcome == "Mark"])
predictions = post_pred.posterior_predictive.outcome.to_numpy().ravel()
s = pd.Series(model.outcome_levels)
fig = px.histogram(pd.Series(predictions).map(s))
fig.update_layout(showlegend=False)
fig.update_xaxes(title="Predicted genre")
fig.update_yaxes(title="Count")
fig.show()
# datapane_elements.append(dp.Text("## Posterior predictive for Mark"))
# datapane_elements.append(dp.Plot(fig))

# %%
def wrangle_summary(summary: pd.DataFrame) -> pd.DataFrame:
    summary = summary.assign(
        error_minus=summary["mean"] - summary["hdi_3%"],
        error=summary["hdi_97%"] - summary["mean"],
    )
    summary = summary[["mean", "error", "error_minus"]].reset_index(
        names="_parameter"
    )
    summary[["Parameter", "Outcome Level"]] = (
        summary["_parameter"].str.rstrip("]").str.split("[").tolist()
    )
    summary = summary.drop(columns="_parameter").rename(
        columns={"mean": "Effect Size"}
    )
    return summary


# %% Effect sizes
fig = px.scatter(
    wrangle_summary(summary),
    facet_col="Outcome Level",
    facet_col_wrap=2,
    color="Parameter",
    x="Parameter",
    y="Effect Size",
    error_y="error",
    error_y_minus="error_minus",
    height=1200,
)
fig.update_xaxes(showticklabels=False)
fig.update_layout(legend=dict(yanchor="middle", y=0.5))
fig.show()
# datapane_elements.append(dp.Text("## Effect sizes"))
# datapane_elements.append(dp.Plot(fig))

# %% Significant effects
fig = px.scatter(
    wrangle_summary(significant_effects),
    y="Outcome Level",
    color="Parameter",
    x="Effect Size",
    error_x="error",
    error_x_minus="error_minus",
    height=600,
)
fig.update_layout(legend=dict(yanchor="middle", y=0.5))
fig.show()
# datapane_elements.append(dp.Text("## Significant effects"))
# datapane_elements.append(dp.Plot(fig))

# %%
dp.Report(*datapane_elements).upload(project_name, publicly_visible=True)

# %%
