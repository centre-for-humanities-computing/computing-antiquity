"""Just messing around with regexes and TLG ID-s"""
# %%
import re
import pandas as pd

# %%
MD_PATH = (
    "/work/data_wrangling/dat/greek/"
    "dataset/Greek Masterchart - new_metadata.csv"
)
md = pd.read_csv(MD_PATH)
md

# %%
tlg_regex = r"(tlg[0-9]+)"
re.findall(tlg_regex, "first1k_tlg0005.tlg003.1st1K-grc1")

# %%
tlg_regex = r"(?P<tlg_author_id>tlg[0-9]+)\.(?P<tlg_work_id>tlg[0-9]+)"
md = md.join(md.document_id.str.extract(tlg_regex))
# %%
md
# %%
md.to_csv(MD_PATH)

# %%
