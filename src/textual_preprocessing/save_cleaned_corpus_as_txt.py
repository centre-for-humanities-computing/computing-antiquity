import argparse
import os
import re
import unicodedata
from pathlib import Path

import pandas as pd
from tqdm import tqdm

SRC_DIR = "dat/greek/clean_data/"
DEST_DIR = "dat/greek/exported_data/"
CORPORA = [
    "with_stopwords",
    "without_stopwords",
    "nouns_lemmatized_without_stopwords",
    "verbs_lemmatized_without_stopwords",
    "lemmatized_with_stopwords",
    "lemmatized_without_stopwords",
]


def create_parser() -> argparse.ArgumentParser:
    """Creates parser for the CLI."""
    parser = argparse.ArgumentParser(
        prog="Corpus Exporter",
        description="Exports cleaned corpus into .txt files.",
    )
    parser.add_argument("--dest_dir", type=str, default=DEST_DIR)
    parser.add_argument("--src_dir", type=str, default=SRC_DIR)
    parser.add_argument(
        "--metadata_sheet_url",
        type=str,
        default=os.environ.get("METADATA_SHEET_URL"),
    )
    return parser


def fetch_metadata(sheet_url):
    sheet_url = sheet_url.replace("/edit#gid=", "/export?format=csv&gid=")
    metadata = pd.read_csv(sheet_url)
    return metadata


def to_legal_filename(filename: str) -> str:
    filename = re.sub(r"[^\w_. -]", "_", filename)
    if len(filename) > 40:
        filename = filename[:40]
    return filename


def remove_diacritics(s: str) -> str:
    return unicodedata.normalize("NFKD", s).translate(
        {ord(c): None for c in "̓̔́̀͂̈ͅ"}
    )


def main():
    parser = create_parser()
    args = parser.parse_args()
    if args.metadata_sheet_url is None:
        raise ValueError(
            "METADATA_SHEET_URL environment variable is not specified,"
            "metadata cannot be downloaded."
        )

    print("Creating output directory")
    Path(args.dest_dir).mkdir(parents=True, exist_ok=True)

    print("Fetching metadata")
    md = fetch_metadata(args.metadata_sheet_url)

    for corpus in CORPORA:
        print(f"\nExporting {corpus}")
        in_path = Path(args.src_dir).joinpath(f"{corpus}.csv")
        out_path = Path(args.dest_dir).joinpath(corpus)
        out_path.mkdir(exist_ok=True)
        data = pd.read_csv(in_path)
        data = data.merge(md, on="document_id").dropna(subset="text")
        data["author"] = data.author.fillna("Anonymous")
        for _, row in tqdm(data.iterrows(), total=len(data)):
            out_file_name = (
                to_legal_filename(f"{row.author} - {row.work}") + ".txt"
            )
            out_file_path = out_path.joinpath(out_file_name)
            content = remove_diacritics(row.text)
            with open(out_file_path, "w") as out_file:
                out_file.write(content)
    print("DONE")


if __name__ == "__main__":
    main()
