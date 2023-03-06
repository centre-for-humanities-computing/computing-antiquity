"""Script that produces report of work lengths"""
from typing import List

import datapane as dp
import numpy as np
import pandas as pd
import plotly.express as px

from utils.analysis import Analysis, DatapaneElement
from utils.files import load_contents_w_stopwords


def calculate_lengths(corpus: pd.DataFrame) -> pd.DataFrame:
    """Calculates lengths of each text in the corpus"""
    corpus = corpus.dropna()
    lengths = corpus.assign(contents=corpus.contents.str.split()).contents.map(
        len
    )
    corpus = corpus.assign(length=lengths)
    return corpus[["id_nummer", "length"]]


def produce_report(results: pd.DataFrame) -> List[DatapaneElement]:
    """Produces report from work lengths"""
    results = results.assign(log10_length=np.log10(results["length"]))
    box = px.box(
        results,
        x="group",
        y="length",
        color="group",
        points="all",
        hover_data=["værk", "forfatter"],
    ).update_layout(
        xaxis_title="",
        yaxis_title="Work length",
        title="Work lengths in the different groups",
        showlegend=False,
        height=600,
    )
    log_box = px.box(
        results,
        x="group",
        y="log10_length",
        color="group",
        points="all",
        hover_data=["værk", "forfatter"],
    ).update_layout(
        xaxis_title="",
        yaxis_title="Log10 Work length",
        title="Log10 Work lengths in the different groups",
        showlegend=False,
        height=600,
    )
    return [dp.Plot(box), dp.Plot(log_box)]


length_analysis = Analysis(
    short_name="work_lengths",
    datapane_name="Værklængde",
    load_input=load_contents_w_stopwords,
    conduct_analysis=calculate_lengths,
    produce_elements=produce_report,
)

if __name__ == "__main__":
    length_analysis.run()
