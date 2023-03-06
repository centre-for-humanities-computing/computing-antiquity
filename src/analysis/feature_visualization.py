# %%
import subprocess

import datapane as dp
import numpy as np
import pandas as pd
import plotly.express as px
from IPython import get_ipython
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.pipeline import Pipeline

from utils.files import prepare_stylistic_features
from utils.metadata import fetch_metadata

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
project_name = "Feature set in 3D"

# %%
data = prepare_stylistic_features()
md = fetch_metadata().set_index("id_nummer")[["værk", "forfatter"]]
data = data.join(md, how="inner")
data

# %%
feature_matrix = data.iloc[:, 1:-2].values
# dim_red = TSNE(n_components=3).fit(feature_matrix)
# print("Explained variance: ", np.sum(pca.explained_variance_ratio_))

# %%
x, y, z = PCA(n_components=3).fit_transform(feature_matrix).T
# %%
data = data.assign(x=x, y=y, z=z)
data = data.assign(title=data.værk + " - " + data.forfatter)
data = data.rename(columns={"outcome": "Genre"})

# %%
fig = px.scatter_3d(
    data,
    x="x",
    y="y",
    z="z",
    color="Genre",
    height=800,
    hover_data={
        "værk": True,
        "forfatter": True,
        "x": False,
        "y": False,
        "z": False,
    },
)
datapane_elements.append(dp.Plot(fig))

# %%
dp.Report(*datapane_elements).upload(project_name, publicly_visible=True)

# %%
data[data.Genre=="Mark"]

# %%
