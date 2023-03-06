from typing import List

import datapane as dp
import pandas as pd
import plotly.express as px

from utils.analysis import Analysis, DatapaneElement
from utils.files import load_dependencies


def calculate_theos_kyrios(dep_df: pd.DataFrame) -> pd.DataFrame:
    """Calculates how many times theos and kyrios appear as objects vs subjects
    in the corpus.
    """
    summary = (
        # Selecting nouns and proper nouns
        dep_df[dep_df.upos.isin(("NOUN", "PROPN"))][
            # Selecting the columns we want to count values by
            ["id_nummer", "lemma", "deprel"]
        ]
        # Counting up each lemma in each work in each depedency relation
        .value_counts().reset_index(name="count")
    )
    # Summing all occurances of all lemmata in each work
    n_total_lemmata = summary.groupby("id_nummer")["count"].sum()
    n_total_lemmata = n_total_lemmata.reset_index(name="n_total")
    # Selecting the lemmata "θεός" and "κύριος"
    theos_kyrios = summary[summary.lemma.isin(("θεός", "κύριος"))]
    # Counting the number of total occurances of both lemmata in each work
    n_total_occurances = theos_kyrios.groupby(["id_nummer", "lemma"])[
        "count"
    ].sum()
    n_total_occurances = n_total_occurances.reset_index(name="n_occurances")
    # Counting the amount of times these lemmata occur as a nominal subject
    # in each work
    subj_counts = (
        theos_kyrios[theos_kyrios["deprel"] == "nsubj"]
        .groupby(["id_nummer", "lemma"])["count"]
        .sum()
    )
    subj_counts = subj_counts.reset_index(name="n_nsubj")
    # Merging the three together into a DataFrame
    theos_kyrios = subj_counts.merge(
        n_total_occurances, on=["id_nummer", "lemma"]
    ).merge(n_total_lemmata, on="id_nummer")
    # Calculating relative frequencies
    theos_kyrios = theos_kyrios.assign(
        lemma_freq=theos_kyrios.n_occurances / theos_kyrios.n_total,
        subj_freq=theos_kyrios.n_nsubj / theos_kyrios.n_occurances,
    )
    return theos_kyrios


def produce_plots(results: pd.DataFrame) -> List[DatapaneElement]:
    """Produces plots for the theos, kyrios analysis."""
    overall_freq_box = px.box(
        results,
        x="group",
        y="lemma_freq",
        color="group",
        points="all",
        hover_data=["værk", "forfatter"],
        facet_col="lemma",
    ).update_layout(
        xaxis_title="",
        yaxis_title="Frequency",
        title=(
            "Relative frequency of θεός and κύριος in different"
            "subgroups of the corpus."
        ),
        showlegend=False,
        height=600,
    )
    subj_freq_box = px.box(
        results,
        x="group",
        y="subj_freq",
        color="group",
        points="all",
        hover_data=["værk", "forfatter"],
        facet_col="lemma",
    ).update_layout(
        xaxis_title="",
        yaxis_title="Frequency",
        title=(
            "Relative frequency of θεός and κύριος being nominal subjects"
            "out of all of their occurances."
        ),
        showlegend=False,
        height=600,
    )
    return [dp.Plot(overall_freq_box), dp.Plot(subj_freq_box)]


theos_kyrios_analysis = Analysis(
    short_name="theos_kyrios",
    datapane_name="Theos, Kyrios",
    load_input=load_dependencies,
    conduct_analysis=calculate_theos_kyrios,
    produce_elements=produce_plots,
)

if __name__ == "__main__":
    theos_kyrios_analysis.run()
