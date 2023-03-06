from typing import List

import datapane as dp
import pandas as pd
import plotly.express as px

from utils.analysis import Analysis, DatapaneElement
from utils.func import flatten
from theos_kyrios import load_dependencies


def calculate_upos_counts(input_data: pd.DataFrame) -> pd.DataFrame:
    """Calculates upos counts for each work in the corpus."""
    counts = (
        input_data.groupby(["id_nummer", "upos"])
        .lemma.count()
        .reset_index(name="lemma_count")
    )
    total_counts = (
        counts.groupby("id_nummer")
        .sum()
        .reset_index()
        .rename(columns={"lemma_count": "total_count"})
    )
    counts = counts.merge(total_counts, on="id_nummer", how="left")
    counts = counts.assign(freq=counts.lemma_count / counts.total_count)
    counts_filtered = counts[counts.upos.isin(("ADJ", "NOUN", "VERB"))]
    return counts_filtered


# TODO: Document this properly
def upos_plots(results: pd.DataFrame) -> List[DatapaneElement]:
    """Produce plots for the upos tag analysis."""
    upos_hist = px.histogram(
        results,
        x="freq",
        facet_col="upos",
        color="group",
        marginal="violin",
    )
    upos_hist.update_layout(
        height=1200,
        yaxis_title="Density",
        title="Distribution of relative frequencies across upos tags",
    )
    upos_hist.add_vline(
        col=1,
        x=0.054933,
        annotation_text="Mark's gospel",
        annotation_y=0.86,
        annotation_x=-0.078,
    )
    upos_hist.add_vline(
        col=2, x=0.148135, annotation_text="Mark's gospel", annotation_y=0.86
    )
    upos_hist.add_vline(
        col=3, x=0.218940, annotation_text="Mark's gospel", annotation_y=0.86
    )
    upos_hist.update_xaxes(title="")
    # TODO: Report this bug to Pylint
    upos_freqs = results.reset_index().pivot(
        index="id_nummer", columns="upos", values="freq"
    )
    upos_freqs = results[["id_nummer", "forfatter", "værk", "group"]].merge(
        upos_freqs.reset_index(), on="id_nummer"
    )
    desc_stats = (
        upos_freqs[["group", "ADJ", "VERB", "NOUN"]]
        .groupby("group")
        .describe()
    )
    desc_tables = [
        (
            dp.Text(f"## Descriptive stats for: {upos}"),
            dp.Table(desc_stats[upos]),
        )
        for upos in desc_stats.columns.levels[0]
    ]
    upos_box = px.box(
        results,
        x="group",
        y="freq",
        color="group",
        # color="group"
        facet_col="upos",
        hover_data=["forfatter", "værk"],
    )
    upos_box.update_layout(
        height=800,
        # showlegend=False,
        yaxis_title="Relative frequency",
        title="Relative frequency of upos tags in each group",
    )
    upos_box.update_xaxes(showticklabels=False)
    upos_scatter = px.scatter_3d(
        upos_freqs,
        x="NOUN",
        y="VERB",
        z="ADJ",
        hover_data=["forfatter", "værk"],
        color="group",
    ).update_layout(
        height=800,
        title="Relative frequencies of nouns, verbs,"
        "and adjectives, in each work",
    )
    return [
        dp.Plot(upos_box),
        dp.Plot(upos_scatter),
        dp.Plot(upos_hist),
        *flatten(desc_tables),
    ]


upos_analysis = Analysis(
    short_name="upos_tags",
    datapane_name="UPOS tag analysis",
    load_input=load_dependencies,
    conduct_analysis=calculate_upos_counts,
    produce_elements=upos_plots,
)

if __name__ == "__main__":
    upos_analysis.run()
