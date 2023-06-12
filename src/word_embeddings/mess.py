from collections import Counter

import pandas as pd
from thefuzz import process

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

whole_fucking_plutarch = " ".join(data[data.author == "Plutarch"].text)

whole_fucking_plutarch.find("παθη")

plutarch_tokens = whole_fucking_plutarch.split()
token_count = Counter(plutarch_tokens)

token_count["παθος"]

process.extract(query="παθος", choices=token_count.keys(), limit=30)
