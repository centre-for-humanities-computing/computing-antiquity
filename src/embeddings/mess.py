import pandas as pd
from gensim.utils import deaccent

DATA_PATH = "dat/greek/cleaned_data/lemmatized_without_stopwords.csv"
SHEET_URL = "https://docs.google.com/spreadsheets/d/15WIzk2aV3vCQLnDihdnNCLxMbDmJZiZKmuiM_xRKbwk/edit#gid=282554525"


def fetch_metadata(url: str) -> pd.DataFrame:
    """Loads metadata from Google Sheets url."""
    url = url.replace("/edit#gid=", "/export?format=csv&gid=")
    metadata = pd.read_csv(url)
    metadata.skal_fjernes = metadata.skal_fjernes == "True"
    return metadata


data = pd.read_csv(DATA_PATH)
metadata = fetch_metadata(SHEET_URL)
data = data.merge(metadata, on="document_id")
data = data[data.author == "Plutarch"]
data.text = data.text.str.lower()
data.text = data.text.map(deaccent)
data.text = data.text.map(lambda s: " ".join(s.split()))

corpus = "\n".join(data.text)
with open("dat/plutarch_corpus.txt", "w") as out_file:
    out_file.write(corpus)
