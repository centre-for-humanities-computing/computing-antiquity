"""Script responsible for parsing the corpus,
producing log and metadata files"""
# %% Importing packages
import os
import glob

import pandas as pd

from utils.parsing import (
    process_files,
    PerseusParser,
    SEPAParser,
    PseudepigraphaParser,
)

# %% Producing paths for raw files
RAW_PATH = "/work/data_wrangling/dat/greek/raw_data"
perseus_path = os.path.join(RAW_PATH, "canonical-greekLit/data")
first1k_path = os.path.join(RAW_PATH, "First1KGreek/data")
sepa_path = os.path.join(RAW_PATH, "SEPA")
pseudepigrapha_path = os.path.join(
    RAW_PATH, "Online-Critical-Pseudepigrapha/static/docs"
)
(perseus_path, first1k_path, sepa_path, pseudepigrapha_path)

# %% Getting all exact file paths
perseus_files = glob.glob(
    os.path.join(perseus_path, "**/*grc*.xml"), recursive=True
)
first1k_files = glob.glob(
    os.path.join(first1k_path, "**/*grc*.xml"), recursive=True
)
sepa_files = glob.glob(os.path.join(sepa_path, "**/*.usx"), recursive=True)
pseudepigrapha_files = glob.glob(os.path.join(pseudepigrapha_path, "*.xml"))

# %% Processing perseus files
process_files(
    paths=perseus_files, parser=PerseusParser(), source_name="perseus"
)
# %% Processing first1k files
process_files(
    paths=first1k_files, parser=PerseusParser(), source_name="first1k"
)
# %% Processing sepa files
process_files(paths=sepa_files, parser=SEPAParser(), source_name="SEPA")
# %% Processing pseudepigrapha files
process_files(
    paths=pseudepigrapha_files,
    parser=PseudepigraphaParser(),
    source_name="pseudepigrapha",
)

# %% Producing common index file
parsed_path = "/work/data_wrangling/dat/greek/parsed_data/"
index_files = glob.glob(os.path.join(parsed_path, "**/index.csv"))
index_dfs = [pd.read_csv(file, index_col=0) for file in index_files]
index_dat = pd.concat(index_dfs, ignore_index=True)
index_dat

# %% Saving index file
index_dat.to_csv(os.path.join(parsed_path, "index.csv"))
