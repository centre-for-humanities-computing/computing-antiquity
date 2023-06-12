import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE
from sklearn.pipeline import make_pipeline

from src.ctfidf.ctfidf import CTFIDFVectorizer

SHEET_URL = "https://docs.google.com/spreadsheets/d/15WIzk2aV3vCQLnDihdnNCLxMbDmJZiZKmuiM_xRKbwk/edit#gid=282554525"


def fetch_metadata(sheet_url: str) -> pd.DataFrame:
    """Fetches metadata in for of a Pandas Dataframe
    from the supplied Google Sheets URL.
    """
    sheet_url = sheet_url.replace("/edit#gid=", "/export?format=csv&gid=")
    metadata = pd.read_csv(sheet_url)
    return metadata


data = pd.read_csv("dat/greek/clean_data/with_stopwords.csv")
metadata = fetch_metadata(SHEET_URL)
data = data.merge(metadata, on="document_id")

data = data.dropna(subset="group")
groups = data.groupby("group").agg({"text": " ".join})

count_vectorizer = CountVectorizer()
class_vectorizer = CTFIDFVectorizer()
class_term_matrix = count_vectorizer.fit_transform(groups.text)
weighted_matrix = class_vectorizer.fit_transform(
    class_term_matrix, n_samples=len(data.index)
)
vect_pipe = make_pipeline(count_vectorizer, class_vectorizer)

dtm = vect_pipe.transform(data.text)

tsne = TSNE(metric="cosine", init="random")
data["x"], data["y"] = tsne.fit_transform(dtm).T

data["name"] = data["author"] + " - " + data["work"]

fig = px.scatter(
    data,
    x="x",
    y="y",
    color="group",
    text="author",
    hover_data=["author", "work", "group"],
)

fig.write_html("ctfidf.html")

np.argsort(weighted_matrix, axis=1)
