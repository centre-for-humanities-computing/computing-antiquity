"""Module for loading data about the corpus"""

from os import path

import numpy as np
import pandas as pd
from scipy.stats import zscore

from utils.metadata import fetch_metadata

DATASET_PATH = "/work/data_wrangling/dat/greek/dataset/"

SENTENCES = "sentences.pkl"
SENTENCES_W_STOPWORDS = "sentences_w_stopwords.pkl"
CONTENTS = "index_to_content.csv"
CONTENT_W_STOPWORDS = "contents_w_stopwords.csv"
CONLL = "conll.feather"


def load_sentences() -> pd.DataFrame:
    """Loads the sentences dataframe for the corpus"""
    return pd.read_pickle(path.join(DATASET_PATH, SENTENCES))


def load_sentences_w_stopwords() -> pd.DataFrame:
    """Loads the sentences dataframe for the corpus"""
    return pd.read_pickle(path.join(DATASET_PATH, SENTENCES_W_STOPWORDS))


def load_contents() -> pd.DataFrame:
    """Loads contents of the corpus"""
    return pd.read_csv(path.join(DATASET_PATH, CONTENTS))


def load_contents_w_stopwords() -> pd.DataFrame:
    """Loads contents of the corpus with stopwords included."""
    return pd.read_csv(
        path.join(DATASET_PATH, CONTENT_W_STOPWORDS), index_col=0
    )


def load_dependencies() -> pd.DataFrame:
    """Loads CoNLL dependency table of the corpus"""
    return pd.read_feather(path.join(DATASET_PATH, CONLL))


def prepare_stylistic_features() -> pd.DataFrame:
    # Loading data
    RESULTS_DIR = "/work/data_wrangling/dat/greek/results/"
    and_analysis = pd.read_pickle(path.join(RESULTS_DIR, "and_analysis.pkl"))
    sentence_lengths = pd.read_pickle(
        path.join(RESULTS_DIR, "sentence_lengths.pkl")
    )
    upos_tags = pd.read_pickle(path.join(RESULTS_DIR, "upos_tags.pkl"))
    vocab_richness = pd.read_pickle(
        path.join(RESULTS_DIR, "vocab_richness.pkl")
    )
    work_lengths = pd.read_pickle(path.join(RESULTS_DIR, "work_lengths.pkl"))
    work_lengths = work_lengths.assign(
        log_work_length=np.log(work_lengths["length"])
    ).drop(columns="length")
    work_lengths["log_work_length"] = zscore(work_lengths["log_work_length"])
    and_analysis = and_analysis[["id_nummer", "first_freq", "and_freq"]]
    and_analysis["first_freq"] = zscore(and_analysis["first_freq"])
    and_analysis["and_freq"] = zscore(and_analysis["and_freq"])
    sentence_lengths = sentence_lengths.reset_index()[
        ["id_nummer", "mean_length"]
    ]
    sentence_lengths["mean_length"] = zscore(sentence_lengths["mean_length"])
    upos_tags["freq"] = zscore(upos_tags["freq"])
    upos_tags = upos_tags.pivot(
        index="id_nummer", columns="upos", values="freq"
    ).reset_index()[["id_nummer", "ADJ", "NOUN", "VERB"]]
    vocab_richness = vocab_richness[["id_nummer", "mattr_500"]]
    vocab_richness["mattr_500"] = zscore(vocab_richness["mattr_500"])
    vocab_richness
    tables = [
        and_analysis,
        sentence_lengths,
        upos_tags,
        vocab_richness,
        work_lengths,
    ]
    tables = [table.set_index("id_nummer") for table in tables]
    metadata = fetch_metadata()
    metadata = metadata[~metadata.skal_fjernes].set_index("id_nummer")
    metadata = metadata.dropna(subset="group")
    metadata = metadata[
        ~metadata.group.isin(["Jewish Philosophy", "Jewish Pseudepigrapha"])
    ]
    data = (
        metadata[["group"]]
        .join(tables)
        .dropna()
        .rename(
            columns={
                "mean_length": "mean_sentence_length",
                "ADJ": "freq_adjectives",
                "NOUN": "freq_nouns",
                "VERB": "freq_verbs",
                "group": "outcome",
                "first_freq": "sentence_starter_and_freq",
            }
        )
    )
    return data
