"""Trains GloVe embeddings using glovpy on Plutarch and Philon."""
from pathlib import Path

import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.utils import deaccent

DATA_PATH = "dat/greek/cleaned_data/lemmatized_without_stopwords.csv"
OUT_PATH = "dat/greek/models/doc2vec/"

SHEET_URL = "https://docs.google.com/spreadsheets/d/15WIzk2aV3vCQLnDihdnNCLxMbDmJZiZKmuiM_xRKbwk/edit#gid=282554525"


def fetch_metadata(url: str) -> pd.DataFrame:
    """Loads metadata from Google Sheets url."""
    url = url.replace("/edit#gid=", "/export?format=csv&gid=")
    metadata = pd.read_csv(url)
    metadata.skal_fjernes = metadata.skal_fjernes == "True"
    return metadata


def train_on_data(data: pd.DataFrame) -> Doc2Vec:
    print("Producing tagged corpus.")
    corpus = [
        TaggedDocument(text, [doc_id])
        for text, doc_id in zip(data.text, data.document_id)
    ]
    out_dir = Path(OUT_PATH)
    out_dir.mkdir(exist_ok=True, parents=True)
    print("Training model.")
    model = Doc2Vec(corpus)
    return model


def main():
    out_dir = Path(OUT_PATH)
    out_dir.mkdir(exist_ok=True, parents=True)
    print("Loading data.")
    data = pd.read_csv(DATA_PATH)
    print("Loading metadata.")
    metadata = fetch_metadata(SHEET_URL)
    data = data.merge(metadata, on="document_id")
    print("Preprocessing corpus.")
    data = data.assign(
        text=data.text.fillna("")
        .str.lower()
        .map(deaccent)
        .map(lambda s: s.split())
    )
    print("Training on the whole corpus.")
    model = train_on_data(data)
    print("Saving keyed vectors.")
    model.dv.save(str(out_dir.joinpath("lemmatized_corpus.vectors.gensim")))
    print("Training on important subset.")
    important_data = data.dropna(subset=["group"])
    important_data = important_data[
        ~important_data["group"].isin(
            ["Jewish Pseudepigrapha", "Jewish Philosophy"]
        )
    ]
    model = train_on_data(important_data)
    print("Saving keyed vectors.")
    model.dv.save(str(out_dir.joinpath("important_works.vectors.gensim")))
    print("DONE")


if __name__ == "__main__":
    main()
