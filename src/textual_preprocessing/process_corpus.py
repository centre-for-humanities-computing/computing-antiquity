"""Script responsible for cleaning the corpus."""
import glob
import os
import subprocess
from pathlib import Path
from typing import List

import pandas as pd
import plotly.graph_objects as go
import spacy
from spacy.tokens import Doc, DocBin
from tqdm import tqdm
from utils.streams import stream_files
from wandb.data_types import Plotly

import wandb

PARSED_INDEX_PATH = "dat/greek/parsed_data/index.csv"
OUT_PATH = "dat/greek/clean_data/"


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


def process_document(text: str, nlp: spacy.Language) -> Doc:
    """Turns text into a spaCy document.
    If the text is too long it is broken into lines and processed that way.
    """
    if len(text) > MAX_LENGTH:
        # If the text is too long, it's broken into its lines.
        texts = text.split("\n")
    else:
        texts = [text]
    docs = list(nlp.pipe(texts))
    return spacy.tokens.Doc.from_docs(docs)


TOKEN_ATTRS = [
    "IS_ALPHA",
    "IS_ASCII",
    "IS_DIGIT",
    "IS_LOWER",
    "IS_PUNCT",
    "IS_SPACE",
    "IS_TITLE",
    "IS_UPPER",
    "LIKE_URL",
    "LIKE_NUM",
    "LIKE_EMAIL",
    "IS_STOP",
    "IS_QUOTE",
    "IS_LEFT_PUNCT",
    "IS_RIGHT_PUNCT",
    "IS_CURRENCY",
    "ID",
    "ORTH",
    "LOWER",
    "NORM",
    "SHAPE",
    "PREFIX",
    "SUFFIX",
    "LENGTH",
    "LEMMA",
    "POS",
    "TAG",
    "DEP",
    "ENT_IOB",
    "ENT_TYPE",
    "ENT_ID",
    "ENT_KB_ID",
    "HEAD",
    "SENT_START",
    "SPACY",
    "LANG",
    "MORPH",
    "IDX",
]


def save_document(doc: Doc, dest: str) -> None:
    """Serializes and saves spaCy Document."""
    doc_bin = DocBin(attrs=TOKEN_ATTRS, docs=[doc])
    doc_bin.to_disk(dest)


MODEL_NAME = "grc_dep_treebanks_trf"
MODEL_CREATOR_NAME = "janko"


def main():
    print(
        "--------------------------\n"
        "------PROCESS CORPUS------\n"
        "--------------------------\n"
    )
    # Creating destination directory
    Path(OUT_PATH).mkdir(exist_ok=True, parents=True)

    # Logging into wandb for logging
    print("Logging into Wandb:")
    subprocess.call(["python3", "-m", "wandb", "login"])
    wandb.init(project="greek-spacy-cleaning", entity="kardosdrur")

    # Downloading spaCy model
    print(f"Downloading model {MODEL_CREATOR_NAME}\{MODEL_NAME}")
    model_source = (
        f"https://huggingface.co/{MODEL_CREATOR_NAME}/"
        f"{MODEL_NAME}/resolve/main/{MODEL_NAME}-any-py3-none-any.whl"
    )
    subprocess.call(["python3", "-m", "pip", "install", model_source])

    # Loading model
    print("Loading NLP model")
    nlp = spacy.load(MODEL_NAME)
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
        tqdm(doc_filenames), texts, range(n_left)
    ):
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
        save_document(doc, dest=doc_out_path)

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
    print("DONE")


if __name__ == "__main__":
    main()
