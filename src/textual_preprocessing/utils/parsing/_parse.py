"""Module for defining parsers for the project."""
import os
import pathlib
from typing import Protocol, TypedDict, Iterable

from IPython.display import clear_output
import pandas as pd
from lxml import etree


class Document(TypedDict):
    """Interface for documents."""

    id: str
    title: str
    author: str
    text: str


class Parser(Protocol):
    """Interface for parsers."""

    def parse_file(self, file: str) -> Iterable[Document]:
        """Turns the given file into an iterable of documents."""
        pass


def out_filename(doc: Document) -> str:
    """Creates filename for the output file."""
    # This is there to respect the maximal filename length
    # title = doc["title"][:60] + "..."
    # return f"{doc['id']}_{title}_{doc['author']}.txt"
    return f"{doc['id']}.txt"


DEST_DIR = "/work/data_wrangling/dat/greek/parsed_data/"


def process_files(paths: Iterable[str], parser: Parser, source_name: str):
    """Parses the given file and puts it in the output folder.
    The output folder will also have an index.csv file with ids
    and information about source and destination files.

    Parameters
    ----------
    paths: iterable of str
        Paths to the files that have to be processed.
    parser: Parser
        Parser object to parse the files into documents.
    source_name: str
        Name of the source project. (e.g. perseus)

    Note
    ----
    This function doesn't stop if it encounters corrupted files,
    but lists them in a corrput_files.log file in the output directory.
    """
    out_dir = os.path.join(DEST_DIR, source_name)
    # Creating directory if it doesn't exist
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    corrupt_path = os.path.join(out_dir, "corrupt_files.log")
    index_records = []
    with open(corrupt_path, "w") as f:
        f.write("")
    for path in paths:
        clear_output(wait=True)
        print(f"processing: {path}")
        try:
            docs = list(parser.parse_file(path))
            # NOTE: This error is implementation detail and should be hidden
            # by the Parser interface
        except etree.XMLSyntaxError:
            with open(corrupt_path, "a") as f:
                f.write(path + "\n")
            continue
        for doc in docs:
            out_path = os.path.join(out_dir, out_filename(doc))
            with open(out_path, "w") as f:
                f.write(doc["text"])
            doc_index = dict(
                src_path=path,
                dest_path=out_path,
                source_name=source_name,
                source_id=doc["id"],
                document_id=source_name + "_" + doc["id"],
                title=doc["title"],
                author=doc["author"],
            )
            index_records.append(doc_index)
    index_df = pd.DataFrame.from_records(index_records)
    index_path = os.path.join(out_dir, "index.csv")
    index_df.to_csv(index_path)
