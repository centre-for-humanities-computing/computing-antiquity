#!/usr/bin/env python3
""""CLI for creating topic models and their visualizations."""

import argparse
import os
import subprocess
from functools import partial
from typing import List

import datapane as dp
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from tweetopic import DMM, TopicPipeline

from utils.files import load_dependencies
from utils.greek import clean
from utils.metadata import fetch_metadata

MODELS_PATH = "/work/data_wrangling/dat/greek/models"
RESULTS_PATH = "/work/data_wrangling/dat/greek/results/topic_models"
DATAPANE_NAME = "Topic Modellering"

# Mapping of names to topic models
TOPIC_MODELS = {
    "nmf": NMF,
    "lda": LatentDirichletAllocation,
    "lsa": TruncatedSVD,
    "lsi": TruncatedSVD,
    "dmm": DMM,
}

# Mapping of names to vectorizers
VECTORIZERS = {
    "tf-idf": TfidfVectorizer,
    "bow": CountVectorizer,
}


def create_parser() -> argparse.ArgumentParser:
    """ "Creates parser for the CLI"""
    parser = argparse.ArgumentParser(
        description="Train topic models over the corpus."
    )
    parser.add_argument(
        "topic_model",
        type=str,
        default="nmf",
        help="""The name of the topic model to train.
        Should be one of the following: {nmf, lda, lsa/lsi, dmm}
        "(optional, default='nmf')""",
    )
    parser.add_argument(
        "-k",
        "--n_components",
        dest="n_components",
        type=int,
        default=100,
        help="Number of topics that should be included in the model"
        "(optional, default=100)",
    )
    parser.add_argument(
        "-v",
        "--vectorizer",
        dest="vectorizer",
        type=str,
        default="bow",
        help="""Vectorizer to produce the matrix for training the topic model.
        Should be either 'bow' or 'tf-idf'.
        (optional, default='bow').
        """,
    )
    return parser


def prepare_corpus() -> pd.DataFrame:
    """Loads corpus, filters for nouns and tries to filter nonsense.

    Returns
    -------
    DataFrame
        Dataframe containing id numbers and textual content.
    """
    dependencies = load_dependencies()
    # Filtering based on metadata deduplication
    metadata = fetch_metadata()
    metadata = metadata[["id_nummer", "skal_fjernes"]]
    dependencies = dependencies.merge(metadata, on="id_nummer", how="inner")
    dependencies = dependencies[~dependencies.skal_fjernes].drop(
        columns="skal_fjernes"
    )
    # Filtering for nouns
    dependencies = dependencies[dependencies.upos == "NOUN"]
    # Joining individual texts together
    contents = dependencies.groupby("id_nummer").agg(text=("lemma", " ".join))
    # Cleaning texts
    contents = contents.assign(
        text=contents.text.map(partial(clean, remove_stops=True))
    )
    return contents


def genre_plot(topic: int, importance: pd.DataFrame) -> go.Pie:
    """Creates a piechart trace visualizing the relevance of
    different genres for a topic.

    Parameters
    ----------
    topic: int
        Index of the topic
    importance: DataFrame
        Table containing information about the
        importance of topics for each group.

    Returns
    -------
    Pie
        Trace of the piechart.
    """
    importance = importance[importance.topic == topic]
    return go.Pie(
        values=importance.importance,
        labels=importance.group,
        textinfo="label",
        domain=dict(x=[0, 0.5]),
        showlegend=False,
    )


def word_plot(topic: int, top_words: pd.DataFrame) -> go.Bar:
    """Shows top words for a topic on a horizontal bar plot.

    Parameters
    ----------
    topic: int
        Index of the topic
    top_words: DataFrame
        Table containing information about word importances
        for each topic.

    Returns
    -------
    Bar
        Bar chart visualizing the top words for a topic.
    """
    vis_df = top_words[top_words.topic == topic]
    return go.Bar(
        y=vis_df.word,
        x=vis_df.importance,
        orientation="h",
        base=dict(x=[0.5, 1]),
        showlegend=False,
    )


def join_plots(row: pd.Series) -> go.Figure:
    """Joins the plots together in one row of the data frame.

    Parameters
    ----------
    row: Series
        Series representing one row of a dataframe containing
        the index of the topic, a bar chart and a pie chart.

    Returns
    -------
    Figure
        Joint plot of the pie and bar charts with titles added.
    """
    fig = make_subplots(
        specs=[
            [{"type": "domain"}, {"type": "xy"}],
        ],
        rows=1,
        cols=2,
        subplot_titles=("Most relevant genres", "Most relevant words"),
    )
    fig.add_trace(row.genre_fig, row=1, col=1)
    fig.add_trace(row.word_fig, row=1, col=2)
    fig.update_layout(
        yaxis=dict(autorange="reversed"),
        height=900,
        title_text=f"Topic {row.topic}",
    )
    return fig


def topic_plots(
    corpus: pd.DataFrame, pipeline: TopicPipeline
) -> List[dp.Plot]:
    """Produces plots for each topic displayig the most relevant genres
    and most relevant words for each topic.

    Parameters
    ----------
    corpus: DataFrame
        Table containing text ids and textual content.
    pipeline: TopicPipeline
        Topic model pipeline.

    Returns
    -------
    list of Plot
        List of datapane plots.
    """
    # Selecting important works by joining with metadata
    # Only those get selected, which are assigned to a group
    metadata = fetch_metadata()
    metadata = metadata[["id_nummer", "group"]].dropna().set_index("id_nummer")
    # Removing unnecessary groups
    # NOTE: This will be removed in the future
    metadata = metadata[
        ~metadata.group.isin(
            [
                "Jewish Philosophy",
                "Jewish Pseudepigrapha",
                "NT Epistolography",
                "LXX Poetry/Wisdom",
            ]
        )
    ]
    important_works = corpus.join(metadata, how="inner")
    # Obtaining probabilities of each important working
    # belonging to a certain topic.
    probs = pipeline.transform(important_works.text)
    important_works = important_works.assign(
        # Assigning a topic embedding to each work
        topic=list(map(np.array, probs.tolist()))
    )
    # Computing aggregated topic importances for each group
    importance = important_works.groupby("group").agg(
        topic=("topic", lambda s: np.stack(s).sum(axis=0))
    )
    # Normalizing these quantities, so that group sizes do
    # not mess with the analysis
    importance = importance.assign(
        topic=importance.topic.map(lambda a: a / a.max())
    )
    # Adding topic labels to the embeddings by enumerating them
    # and then exploding them
    importance = importance.applymap(
        lambda importances: list(enumerate(importances))
    ).explode("topic")
    # Splitting the tuples created by the enumeration to two columns
    importance[["topic", "importance"]] = importance.topic.tolist()
    # Resetting index, cause remember, group was the index because
    # of the aggregation
    importance = importance.reset_index()
    # Creating a Dataframe, where each row represents a topic
    topics = pd.DataFrame({"topic": range(pipeline.topic_model.n_components)})
    # Assigning pie charts to each topic
    topics = topics.assign(
        genre_fig=topics.topic.map(partial(genre_plot, importance=importance))
    )
    # Obtaining top 30 words for each topic
    top = pipeline.top_words(top_n=30)
    # Wrangling the data into tuple records
    records = []
    for i_topic, topic in enumerate(top):
        for word, importance in topic.items():
            records.append((i_topic, word, importance))
    # Adding to a dataframe
    top_words = pd.DataFrame(records, columns=["topic", "word", "importance"])
    # Normalizing word importances
    top_words = top_words.assign(
        importance=top_words.importance / top_words.importance.max()
    )
    # Assigning word bar charts to each topic
    topics = topics.assign(
        word_fig=topics.topic.map(partial(word_plot, top_words=top_words))
    )
    # Joining all plots together for each topic
    plots = topics.apply(join_plots, axis=1)
    # Turning all plots into a list of datapane objects
    return list(map(dp.Plot, plots))


def main() -> None:
    """Main function for the CLI"""
    parser = create_parser()
    args = parser.parse_args()
    n_components = args.n_components
    topic_model_name = args.topic_model.lower()
    vectorizer_name = args.vectorizer.lower()
    topic_model = TOPIC_MODELS[topic_model_name](n_components=n_components)
    # TODO: Add max_df and min_df to CLI parser
    vectorizer = VECTORIZERS[vectorizer_name](max_df=0.1, min_df=10)
    print("Loading and preprocessing corpus")
    corpus = prepare_corpus()
    print("Fitting topic model")
    pipeline = TopicPipeline(vectorizer, topic_model).fit(corpus.text)
    save_path = os.path.join(
        MODELS_PATH,
        f"{vectorizer_name}_{topic_model_name}_{n_components}_pipeline.joblib",
    )
    print(f"Saving topic pipeline to: {save_path}")
    joblib.dump(pipeline, save_path)
    print("Creating plots")
    plots = topic_plots(corpus, pipeline)
    print("Saving them to Datapane")
    subprocess.call(["bash", "datapane_login.sh"])
    dp.Report(*plots).upload("Topic modeling", publicly_visible=True)


if __name__ == "__main__":
    main()
