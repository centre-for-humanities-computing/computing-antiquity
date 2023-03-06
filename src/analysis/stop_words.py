from typing import List

import datapane as dp
import pandas as pd
import plotly.express as px

from utils.analysis import Analysis, DatapaneElement
from utils.files import load_sentences_w_stopwords
from utils.stops import STOPS


def calculate_stopword_freqs(sentences: pd.DataFrame) -> pd.DataFrame:
    """Calculates relative frequency of stop words for each work."""
    tokens = (
        sentences
        # Exploding documents -> sentences
        .explode("sentences")
        .dropna()
        # Exploding sentences -> tokens
        .explode("sentences")
        .dropna()
        # Renaming column for clarity
        .rename(columns={"sentences": "tokens"})
    )
    # Counting up all tokens for each text
    lengths = (
        tokens.groupby("id_nummer")
        .count()
        .rename(columns={"tokens": "n_tokens"})
    )
    # Counting up all stop words for each text
    stop_counts = (
        tokens[tokens.tokens.isin(STOPS)]
        .groupby("id_nummer")
        .count()
        .rename(columns={"tokens": "n_stopwords"})
    )
    stop_counts = stop_counts.join(lengths)
    stop_counts = stop_counts.assign(
        freq=stop_counts.n_stopwords / stop_counts.n_tokens
    )
    return stop_counts


def stop_plots(
    stop_counts: pd.DataFrame,
) -> List[DatapaneElement]:
    """Produce plots for the stopword analysis."""
    scatter_plot = px.scatter(
        stop_counts,
        x="n_tokens",
        y="n_stopwords",
        color="group",
        # hover_data=["forfatter", "værk"],
        # trendline="ols",
        trendline_options=dict(log_x=True, log_y=True),
    ).update_layout(
        height=800,
        xaxis_title="Total number of tokens",
        yaxis_title="Number of stop words",
    )
    box_plot = px.box(
        stop_counts,
        x="group",
        y="freq",
        color="group",
        points="all",
        hover_data=["værk", "forfatter"],
    ).update_layout(
        xaxis_title="",
        yaxis_title="Relative frequency",
        title="Relative frequency of stopwords for each group",
        showlegend=False,
        height=600,
    )
    return [dp.Plot(scatter_plot), dp.Plot(box_plot)]


stop_analysis = Analysis(
    short_name="stop_words",
    datapane_name="Stop Ord",
    load_input=load_sentences_w_stopwords,
    conduct_analysis=calculate_stopword_freqs,
    produce_elements=stop_plots,
)

if __name__ == "__main__":
    stop_analysis.run()
