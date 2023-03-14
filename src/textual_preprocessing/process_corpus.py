"""Script responsible for cleaning the corpus."""
import argparse
import glob
import multiprocessing
import os
from pathlib import Path
from typing import List

import pandas as pd
import plotly.graph_objects as go
import spacy
import torch
import wandb
from spacy.language import Language
from spacy.tokens import Doc, DocBin
from tqdm import tqdm
from utils.streams import stream_files
from wandb.data_types import Plotly

MAX_LENGTH = 10**4

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


def get_done_ids(path: str) -> List[str]:
    """Finds documents that have already been cleaned"""
    paths = glob.glob(os.path.join(path, "*"))
    filenames = [path.split("/")[-1] for path in paths]
    ids = [
        filename.split(".")[0]
        for filename in filenames
        if filename.endswith(".spacy")
    ]
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


def save_document(doc: Doc, dest: str) -> None:
    """Serializes and saves spaCy Document."""
    doc_bin = DocBin(attrs=TOKEN_ATTRS, docs=[doc])
    doc_bin.to_disk(dest)


def process_document(text: str, nlp: Language, dest: str) -> None:
    """Turns text into a spaCy document.
    If the text is too long it is broken into lines and processed that way.
    """
    torch.set_num_threads(1)
    if len(text) > MAX_LENGTH:
        # If the text is too long, it's broken into its lines.
        texts = text.split("\n")
    else:
        texts = [text]
    docs = list(nlp.pipe(texts))
    doc = Doc.from_docs(docs)
    save_document(doc, dest=dest)


def process_doc_in_subprocess(text: str, nlp: Language, dest: str) -> None:
    """Runs processing in subprocess and then
    deletes that subprocess to free up all memory.
    This is needed because spaCy slowly fills up
    CUDA's memory for some reason.
    Blocks execution so that no other subprocesses are started
    (Because CUDA would act very strange.)
    """
    with multiprocessing.Manager():
        process = multiprocessing.Process(
            target=process_document,
            kwargs={"text": text, "nlp": nlp, "dest": dest},
        )
        process.start()
        process.join()


def create_parser() -> argparse.ArgumentParser:
    """Creates parser for the CLI."""
    parser = argparse.ArgumentParser(
        prog="Corpus processor",
        description="Processes all documents in a corpus on GPU",
    )
    parser.add_argument("--model", type=str, default="grc_dep_treebanks_trf")
    parser.add_argument("--dest", type=str, default="dat/greek/clean_data/")
    parser.add_argument(
        "--src_index", type=str, default="dat/greek/parsed_data/index.csv"
    )
    parser.add_argument("--wandb_user", type=str, default="kardosdrur")
    parser.add_argument(
        "--wandb_project", type=str, default="greek-spacy-cleaning"
    )
    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()
    print(
        "--------------------------\n"
        "------PROCESS CORPUS------\n"
        "--------------------------\n"
    )

    # Creating destination directory
    print(f"Creating destination directory ({args.dest})")
    Path(args.dest).mkdir(exist_ok=True, parents=True)

    # Logging into wandb for logging
    print("Initialising wandb")
    wandb.init(project=args.wandb_project, entity=args.wandb_user)

    # Requiring GPU with spaCy
    spacy.require_gpu()

    # Loading model
    print("Loading NLP model")
    nlp = spacy.load(args.model)
    # Resetting max length
    nlp.max_length = MAX_LENGTH

    # Loading Index
    print("Loading index of parsed files")
    parsed_index = pd.read_csv(args.src_index, index_col=0)
    n_total = len(parsed_index.index)

    # Removing texts from the index that have already been cleaned
    done_ids = get_done_ids(args.dest)
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
    doc_filenames = doc_ids.map(
        lambda doc_id: os.path.join(args.dest, f"{doc_id}.spacy")
    )

    # Setting multiprocessing to 'spawn' because of
    # a bug in torch https://github.com/pytorch/pytorch/issues/17199
    multiprocessing.set_start_method("spawn")
    torch.set_num_threads(1)
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
        process_doc_in_subprocess(text, nlp=nlp, dest=doc_out_path)

    # Creating and saving index for cleaned documents
    index = pd.DataFrame(
        {
            "document_id": doc_ids,
            "dest_path": doc_filenames,
            "src_path": src_path,
        }
    )
    print("Saving index")
    index.to_csv(os.path.join(args.dest, "index.csv"))
    print("DONE")


if __name__ == "__main__":
    main()
