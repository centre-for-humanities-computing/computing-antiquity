import os
import pandas as pd
import numpy as np
from IPython.display import clear_output

# this is probably the most inelegant way to do this (khm os.walk for example)
# But since I did this for the greek files, I don't wanna mess with it
def all_files(path):
    return [
        os.path.join(path, file)
        for file in os.listdir(path)
        if os.path.isfile(os.path.join(path, file))
    ]


def index_to_file_df(path):
    paths = all_files(path)
    return pd.DataFrame({"path": paths})


def file_metadata(file_path):
    file_name = ".".join(os.path.basename(file_path).split(".")[:-1])
    parts = file_name.split("_")
    res = [np.nan] * 3
    for i, part in enumerate(parts):
        res[i] = part
    if len(parts) < 2:
        print(parts)
    [ID, title, author] = res
    return ID, title, author


def create_metadata_df(index_to_file):
    ID, title, author = zip(
        *[file_metadata(path) for path in index_to_file["path"]]
    )
    # Replaces empty strings with nan values
    author = [a if a else np.nan for a in author]
    df = pd.DataFrame(
        {
            "perseus_id": ID,
            "forfatter": author,
            "værk": title,
            "Genre": np.nan,
            "Komplet/Fragment": np.nan,
            "Årstal": np.nan,
            "Geografi": np.nan,
            "skal_fjernes": False,
            "Gender": np.nan,
        }
    )
    return df.sort_values(by="forfatter")
