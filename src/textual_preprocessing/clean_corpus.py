"""Script responsible for cleaning the corpus."""
import subprocess
from typing import List
import os
import glob

import pandas as pd
import spacy
import wandb
from wandb.data_types import Plotly
import plotly.graph_objects as go

from utils.streams import stream_files

PARSED_INDEX_PATH = "/work/data_wrangling/dat/greek/parsed_data/index.csv"
OUT_PATH = "/work/data_wrangling/dat/greek/clean_data/"


def get_done_ids(path: str) -> List[str]:
    """Finds documents that have already been cleaned"""
    paths = glob.glob(os.path.join(path, "*"))
    ids = [path.split("/")[-1] for path in paths]
    return ids


def progress_piechart(n_processed: int, n_total: int) -> go.Figure:
    """Draws piechart of progress"""
    fig = go.Figure(
        data=[
            go.Pie(
                labels=["done", "left"],
                values=[n_processed, n_total - n_processed],
            )
        ]
    )
    return fig


MAX_LENGTH = 10**6


def process_document(
    text: str, nlp: spacy.Language
) -> spacy.tokens.Doc:
    """Turns text into a spaCy document.
    If the text is too long it is broken into pieces and processed that way.

    Parameters
    ----------
    text: str
        The raw text.
    nlp: spacy.Language
        SpaCy Language object for processing.

    Returns
    -------
    spacy.tokens.Doc
        SpaCy document object.
    """
    if len(text) > MAX_LENGTH:
        # If the text is too long, it's broken into its lines.
        texts = text.split("\n")
    else:
        texts = [text]
    docs = list(nlp.pipe(texts))
    return spacy.tokens.Doc.from_docs(docs)


def main():
    # Logging into wandb for logging
    subprocess.call(["python3", "-m", "wandb", "login"])
    wandb.init(project="greek-spacy-cleaning", entity="kardosdrur")

    # Loading model
    print("Loading NLP model")
    nlp = spacy.load("grc_ud_proiel_trf")
    # Resetting max length
    nlp.max_length = 10**8

    # Loading Index
    print("Loading index of parsed files")
    parsed_index = pd.read_csv(PARSED_INDEX_PATH, index_col=0)
    n_total = len(parsed_index.index)

    # Removing texts from the index that have already been cleaned
    done_ids = get_done_ids(OUT_PATH)
    done = parsed_index.document_id.isin(done_ids)
    n_done = done.sum()
    print(f"Ignoring previously completed documents (N={n_done})")
    parsed_index = parsed_index[~done]

    # Processing
    print("Processing texts")
    src_path = parsed_index.dest_path
    n_left = len(src_path)
    # Setting up file stream
    texts = stream_files(src_path)

    # Getting document ids from index
    doc_ids = parsed_index.document_id
    # Producing output file names
    doc_filenames = doc_ids.map(lambda doc_id: os.path.join(OUT_PATH, doc_id))

    # Saving SpaCy documents
    for doc_out_path, text, n_processed in zip(
        doc_filenames, texts, range(n_left)
    ):
        print(f" - Producing: {doc_out_path}")
        # Logging progress to Weights and Biases
        wandb.log(
            {
                "n_processed": n_processed,
                "progress": Plotly(
                    progress_piechart(n_processed + n_done, n_total)
                ),
            }
        )
        doc = process_document(text, nlp=nlp)
        doc.to_disk(doc_out_path)

    # Creating and saving index for cleaned documents
    index = pd.DataFrame(
        {
            "document_id": doc_ids,
            "dest_path": doc_filenames,
            "src_path": src_path,
        }
    )
    print("Saving index")
    index.to_csv(os.path.join(OUT_PATH, "index.csv"))


if __name__ == "__main__":
    main()
