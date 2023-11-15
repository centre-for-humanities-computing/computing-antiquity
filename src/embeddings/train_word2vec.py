import os
import unicodedata
from pathlib import Path
from typing import Iterable

import pandas as pd
from gensim.models import Word2Vec
from tqdm import tqdm

DATA_PATH = "dat/greek/cleaned_data/lemmatized_without_stopwords.csv"
OUT_PATH = "dat/greek/models/word2vec"

AUTHORS = [
    "Aristotle",
    "Aelian",
    "Appian",
    "Aristides, Aelius",
    "Arrian",
    "Aristophanes",
    "Cassius Dio",
    "Demosthenes",
    "Diodorus Siculus",
    "Dionysius of Halicarnassus",
    "Flavius Josephus",
    "Galen",
    "Herodotus",
    "Hippocrates",
    "Homer",
    "Isocrates",
    "Libanius",
    "Liber Enoch",
    "Lucian",
    "Lysias",
    "Musonius Rufus",
    "Philo Judaeus",
    "Plato",
    "Plutarch",
    "Testamentum Abrahae",
    "Thucydides",
    "Xenophon of Athens",
    ("Aeschylus", "Sophocles", "Euripides"),
]

GROUPS = [
    "Greek Novels",
    "LXX Historiographies",
    "LXX Propheteia",
    "LXX Poetry/Wisdom",
    "Jewish Novels",
    "NT Narratives",
]


def deaccent(text):
    """Remove letter accents from the given string."""
    if not isinstance(text, str):
        # assume utf8 for byte strings, use default (strict) error handling
        text = text.decode("utf8")
    norm = unicodedata.normalize("NFD", text)
    result = "".join(ch for ch in norm if unicodedata.category(ch) != "Mn")
    return unicodedata.normalize("NFC", result)


def filter_works(
    data: pd.DataFrame, value: tuple[str], column: str
) -> pd.DataFrame:
    """Filters data according to requirements."""
    mask = data[column].isin(value)
    return data[mask]


def fetch_metadata(url: str) -> pd.DataFrame:
    """Loads metadata from Google Sheets url."""
    url = url.replace("/edit#gid=", "/export?format=csv&gid=")
    metadata = pd.read_csv(url)
    metadata.skal_fjernes = metadata.skal_fjernes == "True"
    return metadata


def get_sentences(texts: Iterable[str]) -> list[str]:
    sentences = []
    for text in texts:
        # Deaccenting text
        text = deaccent(text)
        for line in text.split("\n"):
            sentences.append(line.split())
    return sentences


def main() -> None:
    # Creating output directory
    Path(OUT_PATH).mkdir(parents=True, exist_ok=True)
    print("Fetching metadata.")
    try:
        sheet_url = os.environ["GREEK_METADATA_URL"]
    except KeyError:
        print(
            "Please set the GREEK_METADATA_URL environment variable"
            ", otherwise metadata cannot be fetched"
        )
        exit()
    metadata = fetch_metadata(sheet_url)

    print("Loading data.")
    data = pd.read_csv(DATA_PATH)
    # Joining with metadata
    data = data.merge(metadata, how="inner", on="document_id")

    # Author models
    print("Fitting models for author groups.")
    for author in tqdm(AUTHORS):
        # If not a tuple turning it into one
        if not isinstance(author, tuple):
            author = (author,)
        group_data = filter_works(data, value=author, column="author")
        # If no entries are found we skip
        if not len(group_data.index):
            print(f"No data fround for {author}, skipping.")
            continue
        sentences = get_sentences(group_data.text)
        model = Word2Vec(sentences)
        model_name = "Author - {}".format(",".join(author)).replace("/", "-")
        model_dir = Path(OUT_PATH).joinpath(model_name)
        model_dir.mkdir(exist_ok=True)
        model_path = model_dir.joinpath("model.gensim")
        # We turn this into a String because that's what gensim expects
        model.wv.save(str(model_path))

    # Genre models
    print("Fitting models for genre groups.")
    for genre in tqdm(GROUPS):
        # If not a tuple turning it into one
        if not isinstance(genre, tuple):
            genre = (genre,)
        group_data = filter_works(data, value=genre, column="group")
        # If no entries are found we skip
        if not len(group_data.index):
            print(f"No data fround for {genre}, skipping.")
            continue
        sentences = get_sentences(group_data.text)
        model = Word2Vec(sentences)
        model_name = "Genre - {}".format(",".join(genre)).replace("/", "-")
        model_dir = Path(OUT_PATH).joinpath(model_name)
        model_dir.mkdir(exist_ok=True)
        model_path = model_dir.joinpath("model.gensim")
        # We turn this into a String because that's what gensim expects
        model.wv.save(str(model_path))
    print("DONE")


if __name__ == "__main__":
    main()
