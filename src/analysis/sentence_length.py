from typing import List

import datapane as dp
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from utils.analysis import Analysis, DatapaneElement
from utils.files import load_sentences_w_stopwords
from utils.mark import n_chapter_sentences


def calculate_sentence_lengths(sentences_df: pd.DataFrame) -> pd.DataFrame:
    """Calculates the mean length of sentences
    for each work in the supplied dataframe.

    Parameters
    ----------
    sentences_df: DataFrame
        Dataframe containing an id number and a sentences column
        containing the list of sentences in each work.

    Returns
    -------
    DataFrame
        Dataframe containing an id number and a mean sentence length value.
    """
    sentences_df = sentences_df.explode("sentences").dropna()
    sentences_df = sentences_df.assign(length=sentences_df.sentences.map(len))
    lengths = sentences_df[["id_nummer", "length"]]
    lengths = lengths.groupby("id_nummer").agg(
        # Imploding sentence lengths into a list
        lengths=("length", lambda s: s.tolist()),
        # Calculating mean
        mean_length=("length", "mean"),
    )
    return lengths


def sentence_length_plots(results: pd.DataFrame) -> List[DatapaneElement]:
    """Creates plots for the sentence length analysis."""
    box_plot = px.box(
        results,
        x="group",
        y="mean_length",
        color="group",
        points="all",
        hover_data=["værk", "forfatter"],
    ).update_layout(
        xaxis_title="",
        yaxis_title="Mean sentence length",
        title="Mean sentence length of different groups in the corpus",
        showlegend=False,
        height=600,
    )
    mark_sent_len = pd.Series(results.set_index("id_nummer").lengths.loc[783])
    mark_dat = pd.DataFrame(
        {
            "lengths": mark_sent_len,
            "i_sentence": range(len(mark_sent_len)),
            "rolling_average": mark_sent_len.rolling(50).mean(),
        }
    )
    mark_plot = go.Figure()
    mark_plot.add_scatter(
        x=mark_dat.i_sentence,
        y=mark_dat.lengths,
        line=dict(color="blue", width=1),
        opacity=0.2,
        name="Original data",
    )
    mark_plot.add_scatter(
        x=mark_dat.i_sentence,
        y=mark_dat.rolling_average,
        line=dict(color="red", width=3),
        opacity=0.8,
        name="Rolling average (50)",
    )
    mark_plot.update_layout(
        xaxis_title="Sentence",
        yaxis_title="Sentence length",
        title="Sentence length over Mark's gospel",
        height=600,
    )
    chapter_boundaries = np.cumsum(n_chapter_sentences)
    chapter_boundaries = [0] + chapter_boundaries.tolist()
    for boundary in chapter_boundaries:
        mark_plot.add_vline(
            x=boundary, line_width=2, line_color="black", opacity=0.8
        )
    return [dp.Plot(box_plot), dp.Plot(mark_plot)]


sentence_length_analysis = Analysis(
    short_name="sentence_lengths",
    datapane_name="Sætningslængde",
    load_input=load_sentences_w_stopwords,
    conduct_analysis=calculate_sentence_lengths,
    produce_elements=sentence_length_plots,
)

if __name__ == "__main__":
    sentence_length_analysis.run()
