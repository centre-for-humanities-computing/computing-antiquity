from typing import List

import datapane as dp
import numpy as np
import pandas as pd
import plotly.express as px

from utils.analysis import Analysis, DatapaneElement
from utils.files import load_sentences_w_stopwords


def count_ands(sentences_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each work, counts how many of the sentences begin with
    'καί (and)' as well as how many sentences there are, and
    the relative frequency of sentences starting with 'and'.

    Parameters
    ----------
    sentences_df: DataFrame
        Data frame containing all sentences in the corpus.

    Returns
    -------
    DataFrame
        id_nummer - ID of the text
        first_and - Number of sentences starting with 'and'
        n_sent - Total number of sentences
        freq - Relative frequency of sentences starting with 'and'
    """
    # Exploding lists of sentences to one sentence per row
    sentences_df = sentences_df.explode("sentences")
    sentences_df = sentences_df.dropna()
    # Exctracting all tokens
    tokens = sentences_df.explode("sentences").rename(
        columns={"sentences": "tokens"}
    )
    # Getting work lengths
    lengths = (
        tokens.groupby("id_nummer")
        .count()
        .rename(columns={"tokens": "n_tokens"})
    )
    # Counting up 'and' in each work
    and_counts = tokens.groupby("id_nummer").value_counts().loc[:, "καί"]
    # Extracting first words of the sentence
    sentences_df = sentences_df.assign(
        first_word=(
            sentences_df.sentences.map(lambda l: l[0] if l else np.nan)
        )
    )
    # Counting up how many times each first word comes up
    first_counts = (
        sentences_df[["id_nummer", "first_word"]]
        .groupby("id_nummer")
        .value_counts()
    )
    # Selecting 'and'
    first_ands = first_counts.loc[:, "καί"]
    # Counting up how many sentences there are in total in each text
    n_sentences = sentences_df.groupby("id_nummer").sentences.count()
    # Joining the data together
    counts_df = (
        pd.DataFrame({"n_sent": n_sentences, "n_tokens": lengths.n_tokens})
        .join(first_ands.rename("first_and"), how="right")
        .join(and_counts.rename("and_count"))
    )
    # Adding a relative frequency column
    counts_df = counts_df.assign(
        first_freq=counts_df.first_and / counts_df.n_sent,
        and_freq=counts_df.and_count / counts_df.n_tokens,
    )
    counts_df = counts_df.reset_index()
    return counts_df


def and_plots(results: pd.DataFrame) -> List[DatapaneElement]:
    """Produce plots for the And Analysis"""
    first_scatter = px.scatter(
        results,
        x="n_sent",
        y="first_and",
        color="group",
        hover_data=["forfatter", "værk"],
    ).update_layout(
        height=800,
        yaxis_title="Number of sentences beginning with 'and'",
        xaxis_title="Number of sentences",
    )
    overall_scatter = px.scatter(
        results,
        x="n_tokens",
        y="and_count",
        color="group",
        hover_data=["forfatter", "værk"],
    ).update_layout(
        height=800,
        yaxis_title="Total number of occurances of 'and'",
        xaxis_title="Total number of tokens",
    )
    first_box = px.box(
        results,
        x="group",
        y="first_freq",
        color="group",
        points="all",
        hover_data=["værk", "forfatter"],
    ).update_layout(
        xaxis_title="",
        yaxis_title="Frequency",
        title="Frequency of sentences starting with 'and'"
        "over different genres",
        showlegend=False,
        height=600,
    )
    overall_box = px.box(
        results,
        x="group",
        y="and_freq",
        color="group",
        points="all",
        hover_data=["værk", "forfatter"],
    ).update_layout(
        xaxis_title="",
        yaxis_title="Frequency",
        title="Relative frequency of 'and' in different groups",
        showlegend=False,
        height=600,
    )
    return [
        dp.Plot(overall_scatter),
        dp.Plot(overall_box),
        dp.Plot(first_scatter),
        dp.Plot(first_box),
    ]


and_analysis = Analysis(
    short_name="and_analysis",
    datapane_name="Og Analyse",
    load_input=load_sentences_w_stopwords,
    conduct_analysis=count_ands,
    produce_elements=and_plots,
)

if __name__ == "__main__":
    and_analysis.run()
