"""Script for producing a jsonl file for pretraining SpaCy models."""
from typing import Iterable, Dict
import os
import json

import pandas as pd
from tqdm import tqdm

PARSED_PATH = "/work/data_wrangling/dat/greek/parsed_data/"


def stream_docs(index_df: pd.DataFrame) -> Iterable[Dict[str, str]]:
    """Streams doc ids and textual content for the whole corpus."""
    for index, row in index_df.iterrows():
        with open(row.dest_path) as in_file:
            text = in_file.read()
        document = {"document_id": row.document_id, "text": text}
        yield document


def main() -> None:
    print("Loading index")
    index_path = os.path.join(PARSED_PATH, "index.csv")
    index_df = pd.read_csv(index_path)
    n_docs = len(index_df.index)
    out_path = os.path.join(PARSED_PATH, "corpus.jsonl")
    print("Processing documents:")
    # Using progress bar
    with tqdm(total=n_docs) as progress_bar:
        with open(out_path, "w") as out_file:
            # Writing all json documents to the output file into lines
            for document in stream_docs(index_df):
                out_file.write(json.dumps(document) + "\n")
                # Updating progress bar on each document
                progress_bar.update()
    print("DONE")


if __name__ == "__main__":
    main()
