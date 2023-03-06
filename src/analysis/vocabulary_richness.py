"""Module for conducting the vocabulary richness analysis."""
from collections import Counter
from typing import Any, List

import datapane as dp
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

from utils.analysis import Analysis, DatapaneElement
from utils.files import load_contents_w_stopwords
from utils.mark import n_chapter_words


def moving_ttr(doc: List[str], window_size: int = 50) -> List[float]:
    """Calculates moving type-token-ratios for each window in a text.

    Parameters
    ----------
    doc: list of str
        List of words in a given document.
    window_size: int, default 50
        Size of sliding windows.

    Returns
    -------
    list of float
        TTR for each window.
    """
    # Counts up the occurances of each word in the first window
    counter = Counter(doc[:window_size])
    # The number of unique items is determined by number of keys in the
    # hash map
    n_types = len(counter)
    # Collecting ttrs in a list
    ttrs = [n_types / window_size]
    for i in range(len(doc) - window_size):
        # Removing the word from the hashmap,
        # that has just gone out of the window
        # and adding the one that has just come in.
        old_word = doc[i]
        new_word = doc[i + window_size]
        counter[old_word] -= 1
        if not counter[old_word]:
            del counter[old_word]
        if new_word in counter:
            counter[new_word] += 1
        else:
            counter[new_word] = 1
        n_types = len(counter)
        ttrs.append(n_types / window_size)
    return ttrs


def unique_count(elements: List[Any]) -> int:
    """
    Counts how many unique elements there are in a list
    with fast pandas routines.

    Parameters
    ----------
    elements: list
        List to count the unique elements in.

    Returns
    -------
    int
        Number of unique elements.
    """
    return pd.Series(elements).unique().size


def calculate_vocab_richness(contents_df: pd.DataFrame) -> pd.DataFrame:
    """Calculates type-token-ratio for each work in the corpus."""
    # Splitting texts to words
    contents_df = contents_df.assign(
        tokens=contents_df.contents.str.split()
    ).dropna()
    # Counting up number of unique words and total number of words
    contents_df = contents_df.assign(
        n_unique_tokens=contents_df.tokens.map(unique_count),
        n_tokens=contents_df.tokens.map(len),
    )
    # Calculating vocabulary richness
    contents_df = contents_df.assign(
        ttr=contents_df.n_unique_tokens / contents_df.n_tokens
    )
    # Calculating moving ttrs with different window sizes
    contents_df = contents_df.assign(
        ttrs_500=contents_df.tokens.apply(moving_ttr, window_size=500),
        ttrs_10=contents_df.tokens.apply(moving_ttr, window_size=10),
        # ttrs_250=contents_df.tokens.apply(moving_ttr, window_size=250),
        # ttrs_300=contents_df.tokens.apply(moving_ttr, window_size=300),
    )
    # Calculating moving average ttr
    contents_df = contents_df.assign(
        mattr_500=contents_df.ttrs_500.map(np.mean),
        mattr_10=contents_df.ttrs_10.map(np.mean),
    )
    # Calculating standard deviation of moving ttr
    contents_df = contents_df.assign(
        sd_ttr_500=contents_df.ttrs_500.map(np.std)
    )
    # I'm dropping the two columns to free up memory, as they are quite huge
    contents_df = contents_df.drop(columns=["contents", "tokens"])
    vocab_richness = contents_df
    return vocab_richness


def make_ttr_timeline(ttrs: pd.Series, title: str) -> go.Figure:
    """Makes a timeline of type-token ratios with a rolling average.

    Parameters
    ----------
    ttrs: Series
        pandas series of TTRs for each window.
    title: str
        Title of the graph.
    """
    timeline = pd.DataFrame(
        {
            "ttrs": ttrs,
            "window": range(len(ttrs)),
            "moving_average": ttrs.rolling(500).mean(),
        }
    )
    timeline_plot = go.Figure()
    # Adding original data
    timeline_plot.add_scatter(
        x=timeline.window,
        y=timeline.ttrs,
        line=dict(color="blue", width=1),
        opacity=0.2,
        name="Original data",
    )
    # Adding trace for rolling average.
    timeline_plot.add_scatter(
        x=timeline.window,
        y=timeline.moving_average,
        line=dict(color="red", width=3),
        opacity=0.8,
        name="Rolling average (500)",
    )
    timeline_plot.update_layout(
        xaxis_title="Window",
        yaxis_title="Type-token ratio",
        title=title,
        height=900,
    )
    return timeline_plot


def generate_mark_timeline(
    vocab_richness: pd.DataFrame, window_size: int
) -> go.Figure:
    """Generates timeline with Mark's gospel with chapter boundaries added.

    Parameters
    ----------
    vocab_richness: DataFrame
        Dataframe containing the results of the analysis.
    window_size: int
        Indicates from which column to make the timeline.
    """
    mark_ttrs = vocab_richness.set_index("id_nummer")[
        f"ttrs_{window_size}"
    ].loc[783]
    mark_ttrs = pd.Series(mark_ttrs)
    plot = make_ttr_timeline(
        mark_ttrs,
        "Moving type-token-ratio for each window in Mark's gospel"
        f"(window_size={window_size})",
    )
    chapter_boundaries = np.cumsum(n_chapter_words) - window_size + 1
    chapter_boundaries = [0] + chapter_boundaries.tolist()
    for boundary in chapter_boundaries:
        plot.add_vline(
            x=boundary, line_width=2, line_color="black", opacity=0.8
        )
    return plot


def overlapping_histogram(results: pd.DataFrame) -> go.Figure:
    """Creates an overlapping histogram for comparison of vocabulary richness
    of Jewish and Greek works with percentiles for Mark's gospel."""
    # Selecting dat for Mark's gospel
    mark_mattr_500 = results.set_index("id_nummer").mattr_500.loc[783]
    # Selecting dat for greek works
    greek = results.mattr_500[results.etnicitet == "græsk"]
    # Selecting dat for jewish works
    jew = results.mattr_500[results.etnicitet == "jødisk"]
    # Calculating percentiles for both groups
    greek_percentile = stats.percentileofscore(greek, mark_mattr_500)
    jew_percentile = stats.percentileofscore(jew, mark_mattr_500)
    # Constructing plot
    fig = go.Figure()
    fig.add_histogram(x=greek, name="Greek texts")
    fig.add_histogram(x=jew, name="Jewish texts")
    # Overlay both histograms
    fig.update_layout(barmode="overlay")
    # Reduce opacity to see both histograms
    fig.update_traces(opacity=0.75)
    fig.add_vline(
        x=mark_mattr_500,
        annotation_text="<b>Mark's gospel</b><br>"
        f"<i>(greek percentile: {greek_percentile:.2f}%)<br>"
        f"(jewish percentile: {jew_percentile:.2f}%)</i>",
    )
    fig.update_layout(
        title="Distribution of MATTR scores for Jewish and Greek texts."
        "(window_size=500)",
        xaxis_title="MATTR",
        height=800,
    )
    return fig


def moving_trr_sd_histogram(results: pd.DataFrame) -> go.Figure:
    """Creates histogram over the standard deviations
    of TTR windows over the whole corpus, with a line marking
    Mark's gospel."""
    # Selecting dat for Mark's gospel
    mark_sd_ttr_500 = results.set_index("id_nummer").sd_ttr_500.loc[783]
    # Calculating Mark's percentile
    mark_percentile = stats.percentileofscore(
        results.sd_ttr_500, mark_sd_ttr_500
    )
    # Constructing plot
    fig = px.histogram(results, x="sd_ttr_500")
    fig.update_layout(
        title="Standard deviations of moving TTR in the corpus."
        "(window_size=500)",
        xaxis_title="SD of TTRs",
        yaxis_title="Density",
    )
    fig.add_vline(
        x=mark_sd_ttr_500,
        annotation_text="<b>Mark's gospel</b><br>"
        f"<i>(percentile: {mark_percentile:.2f}%)</i>",
    )
    return fig


def vocab_richness_plots(
    vocab_richness: pd.DataFrame,
) -> List[DatapaneElement]:
    """Produce plots for the vocab richness analysis"""
    ttr_plot = px.box(
        vocab_richness,
        x="group",
        y="ttr",
        color="group",
        points="all",
        hover_data=["værk", "forfatter"],
    ).update_layout(
        xaxis_title="",
        yaxis_title="Vocabulary richness",
        title="Type Token Ratios of different groups",
        showlegend=False,
        height=600,
    )
    mattr_500_plot = px.box(
        vocab_richness,
        x="group",
        y="mattr_500",
        color="group",
        points="all",
        hover_data=["værk", "forfatter"],
    ).update_layout(
        xaxis_title="",
        yaxis_title="MTTR",
        title="Moving type token ratio of different groups."
        "(window_size=500)",
        showlegend=False,
        height=600,
    )
    mattr_10_plot = px.box(
        vocab_richness,
        x="group",
        y="mattr_10",
        color="group",
        points="all",
        hover_data=["værk", "forfatter"],
    ).update_layout(
        xaxis_title="",
        yaxis_title="MTTR",
        title="Moving type token ratio of different groups."
        "(window_size=10)",
        showlegend=False,
        height=600,
    )
    hist_with_philosophy = overlapping_histogram(vocab_richness)
    greek = vocab_richness.mattr_500[
        vocab_richness.etnicitet == "græsk"
    ].astype(float)
    jew = vocab_richness.mattr_500[
        vocab_richness.etnicitet == "jødisk"
    ].astype(float)
    # Coducting statistical test for the report
    # TODO: Abstract this away to a function.
    n1, n2 = len(greek), len(jew)
    # print(greek, type(greek))
    # print(jew, type(jew))
    u_statistic, p_value = stats.mannwhitneyu(greek, jew)
    siginificance = (
        ">= 0.05"
        if p_value > 0.05
        else "< 0.05"
        if p_value > 0.01
        else "< 0.01"
        if p_value > 0.001
        else "< 0.001"
    )
    statistical_test_text = (
        "<b>Statistical test:</b> "
        f"Mann-Whitney _U_ = {u_statistic:.2f}, _n1_ = {n1}, "
        f"_n2_ = {n2}, _P_ {siginificance}"
    )
    hist_wo_philosophy = overlapping_histogram(
        vocab_richness[vocab_richness.group != "Jewish Philosophy"]
    ).update_layout(
        title="Distribution of MATTR scores for Jewish and Greek text."
        "(window_size=500)<br><i>without Jewish Philosophy</i>"
    )
    timelines = [
        generate_mark_timeline(vocab_richness, window_size)
        for window_size in (500,)
    ]
    sd_plot = moving_trr_sd_histogram(vocab_richness)
    return [
        dp.Plot(ttr_plot),
        dp.Plot(mattr_500_plot),
        dp.Plot(hist_with_philosophy),
        dp.Text(statistical_test_text),
        dp.Plot(hist_wo_philosophy),
        dp.Plot(mattr_10_plot),
        *map(dp.Plot, timelines),
        dp.Plot(sd_plot),
    ]


vocab_richness_analysis = Analysis(
    short_name="vocab_richness",
    datapane_name="Vocabulary Richness",
    load_input=load_contents_w_stopwords,
    conduct_analysis=calculate_vocab_richness,
    produce_elements=vocab_richness_plots,
)

if __name__ == "__main__":
    vocab_richness_analysis.run()
