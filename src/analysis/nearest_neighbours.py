# %% Importing stuff
from scipy.spatial import distance_matrix
import plotly.express as px
from arviz.stats import hdi

from utils.files import prepare_stylistic_features
from utils.metadata import fetch_metadata

# %% Setting up autoreload
try:
    ipython = get_ipython()
    ipython.magic("load_ext autoreload")
    ipython.magic("autoreload 2")
except Exception:
    print("Could not load Ipython kernel, continuing without autoreload")

# %%
data = prepare_stylistic_features()
md = fetch_metadata().set_index("id_nummer")[["værk", "forfatter"]]
data = data.join(md, how="inner")
feature_matrix = data.iloc[:, 1:-2].values
data = data.reset_index()
mark_index, = data[data.outcome=="Mark"].index
# %%
delta = distance_matrix(feature_matrix, feature_matrix)
distance_from_mark = delta[mark_index]
data["distance_from_mark"] = distance_from_mark

# %%
def confidence(a):
    lower, upper = hdi(a.to_numpy())
    return (lower, upper)
summary = data.groupby("outcome").agg(median_dist=("distance_from_mark", "median"), hdi=("distance_from_mark", confidence))
summary[["lower", "upper"]] = summary.hdi.tolist()
summary["error_y"] = summary.upper - summary.median_dist
summary["error_y_minus"] = summary.median_dist - summary.lower
summary = summary.reset_index()
summary = summary[summary.outcome != "Mark"]
summary = summary.sort_values("median_dist")
summary
# %%
fig = px.bar(summary, x="median_dist", y="outcome", error_x="error_y", error_x_minus="error_y_minus")
fig.update_layout(yaxis=dict(autorange="reversed"))
fig
# %%
