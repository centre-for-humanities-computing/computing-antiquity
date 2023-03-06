"""
Useful functions for displaying purposes.

NOTE: Might merge either with learn.py, as most functions display classifier performance,
or add displaying function to this file from embeddings.py and clustering.py
"""
from typing import List

import pandas as pd
from plotly import express as px
from plotly import graph_objects as go
from plotly import subplots
from deprecated import deprecated


def stack_plots(figures: List[go.Figure], titles: List[str]) -> go.Figure:
    """
    Stacks the given list of figures on top of each other into one big ass figure.
    Titles have to be given separately.
    """
    stacked = subplots.make_subplots(
        rows=len(figures), cols=1, subplot_titles=titles
    )
    for i, fig in enumerate(figures):
        for trace in fig["data"]:
            stacked.append_trace(trace, row=i + 1, col=1)
    stacked.update_layout(height=len(figures) * 400, showlegend=False)
    return stacked


@deprecated(
    reason="Phased out in favor of plot_report and plot_vs_dummy, please use these instead."
)
def evaluation_plots(
    predictions: List[str], scores: List[float], report: pd.DataFrame
) -> go.Figure:
    """
    DEPRECATED: Please use plot_report() and plot_vs_dummy() instead.
    Plots results of classifier evaluation.
    """
    metrics = report.drop("label", axis=1).columns
    titles = []
    figures = []
    hist = px.histogram(x=predictions, color=predictions)
    hist.update_layout(showlegend=False)
    figures.append(hist)
    titles.append("Distribution of predicted labels for test_value:")
    figures.append(px.histogram(x=scores))
    titles.append(
        "Distribution of accuracy scores over a randomly generated test set:"
    )
    for metric in metrics:
        figures.append(
            px.box(report, x="label", y=metric, color="label", points="all")
        )
        titles.append(
            f"Distribution of {metric} over different labels in the test set:"
        )
    return stack_plots(figures, titles)


def plot_report(report: pd.DataFrame) -> go.Figure:
    """
    Plots all classification metrics from evaluation for all classes.
    """
    metrics = report.drop(columns=["label", "model", "support"]).columns
    titles = []
    figures = []
    for metric in metrics:
        figures.append(
            px.box(report, x="label", y=metric, color="label", points="all")
        )
        titles.append(
            f"Distribution of {metric} over different labels in the test set:"
        )
    return stack_plots(figures, titles)


def plot_vs_dummy(report: pd.DataFrame) -> go.Figure:
    """
    Plots weighted averages of all classification metrics
    from evaluation against chance level (dummy model).
    """
    metrics = report.drop(columns=["label", "model", "support"]).columns
    titles = []
    figures = []
    for metric in metrics:
        disp = report[report.label == "weighted avg"]
        figures.append(
            px.violin(disp, x="model", y=metric, points="all", color="model")
        )
        titles.append(
            f"Distribution of {metric} over different labels in the test set:"
        )
    return stack_plots(figures, titles)
