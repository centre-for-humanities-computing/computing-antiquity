"""Trains GloVe embeddings using glovpy on Plutarch and Philon."""
from pathlib import Path

import pandas as pd
from gensim.utils import deaccent
from glovpy import GloVe

AUTHORS = ["Plutarch", "Philo Judaeus"]
DATA_PATH = "dat/greek/cleaned_data/lemmatized_without_stopwords.csv"
OUT_PATH = "dat/greek/models/glove/"
SHEET_URL = "https://docs.google.com/spreadsheets/d/15WIzk2aV3vCQLnDihdnNCLxMbDmJZiZKmuiM_xRKbwk/edit#gid=282554525"


def fetch_metadata(url: str) -> pd.DataFrame:
    """Loads metadata from Google Sheets url."""
    url = url.replace("/edit#gid=", "/export?format=csv&gid=")
    metadata = pd.read_csv(url)
    metadata.skal_fjernes = metadata.skal_fjernes == "True"
    return metadata


def main():
    print("Loading data.")
    data = pd.read_csv(DATA_PATH)
    print("Loading metadata.")
    metadata = fetch_metadata(SHEET_URL)
    data = data.merge(metadata, on="document_id")
    data = data[data.author.isin(AUTHORS)]
    print("Preprocessing corpus.")
    data.text = data.text.str.lower()
    data.text = data.text.map(deaccent)
    data.text = data.text.map(lambda s: s.split())
    out_dir = Path(OUT_PATH)
    out_dir.mkdir(exist_ok=True, parents=True)
    for author in AUTHORS:
        print(f"Training model for {author}...")
        corpus = list(data.text[data.author == author])
        model = GloVe(vector_size=50, window_size=15, iter=25)
        model.train(corpus)
        print("Saving.")
        out_filename = f"Author - {author} - Glove.gensim"
        model.wv.save(str(out_dir.joinpath(out_filename)))
    print("DONE")


if __name__ == "__main__":
    main()
