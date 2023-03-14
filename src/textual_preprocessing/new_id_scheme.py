"""LEGACY FILE, IT IS NO LONGER ACTIVELY USED AND SHOULD BE IGNORED"""
# %%
import os
import re
import shutil

import numpy as np
import pandas as pd
from utils.metadata import fetch_metadata

# %%
mapping = pd.DataFrame(columns=["document_id", "id_nummer"])

# %%
index_to_file = pd.read_csv(
    "/work/data_wrangling/dat/greek/dataset/index_to_file.csv", index_col=0
)
index_to_file[["folder", "file"]] = index_to_file.path.str.split("/").tolist()
index_to_file = index_to_file.drop(columns=["path", "new_id"])
index_to_file = index_to_file.assign(
    file=index_to_file.file.str.replace(".txt", "", regex=False)
)
index_to_file[["legacy_id", "title", "author"]] = index_to_file.file.str.split(
    "_"
).tolist()
index_to_file = index_to_file.assign(
    legacy_id=index_to_file.legacy_id.str.replace(".xm", "", regex=False)
)
index_to_file = index_to_file.drop(columns="file")
index_to_file = index_to_file.rename(columns={"folder": "source_name"})
missing_id_nums = index_to_file.id_nummer.unique()
index_to_file

# %%
index_dat = pd.read_csv(
    "/work/data_wrangling/dat/greek/parsed_data/index.csv", index_col=0
)
perseus_legacy_ids = index_dat.source_id[
    index_dat.source_name == "perseus"
].map(lambda s: ".".join(s.split(".")[:2]))
sepa_legacy_ids = index_dat.source_id[index_dat.source_name == "SEPA"].map(
    lambda s: s.split("-")[1]
)
index_dat = index_dat.assign(
    legacy_id=index_dat.source_id.mask(
        index_dat.source_name == "perseus", perseus_legacy_ids
    ).mask(index_dat.source_name == "SEPA", sepa_legacy_ids)
)
index_dat = index_dat.drop(columns=["src_path", "dest_path"])
missing_doc_ids = index_dat.document_id.unique()

# %%
print(
    f"""
    Missing id_nummer: {missing_id_nums.size}
    Missing document_id: {missing_doc_ids.size}
"""
)

# %%
joint = index_dat.merge(
    index_to_file,
    on=["source_name", "legacy_id", "title", "author"],
    how="outer",
)
joint = joint[
    joint.id_nummer.isin(missing_id_nums)
    & joint.document_id.isin(missing_doc_ids)
]
mapping = pd.concat((mapping, joint[["id_nummer", "document_id"]])).dropna()
missing_id_nums = np.setdiff1d(missing_id_nums, mapping.id_nummer.unique())
missing_doc_ids = np.setdiff1d(missing_doc_ids, mapping.document_id.unique())

print(
    f"""
    Missing id_nummer: {missing_id_nums.size}
    Missing document_id: {missing_doc_ids.size}
"""
)

# %%
joint = index_dat.merge(
    index_to_file.drop(columns=["title", "author"]),
    on=["legacy_id", "source_name"],
    how="outer",
)
joint = joint[
    joint.id_nummer.isin(missing_id_nums)
    & joint.document_id.isin(missing_doc_ids)
]
mapping = pd.concat((mapping, joint[["id_nummer", "document_id"]])).dropna()
missing_id_nums = np.setdiff1d(missing_id_nums, mapping.id_nummer.unique())
missing_doc_ids = np.setdiff1d(missing_doc_ids, mapping.document_id.unique())

print(
    f"""
    Missing id_nummer: {missing_id_nums.size}
    Missing document_id: {missing_doc_ids.size}
"""
)

# %%
joint = index_dat.merge(
    index_to_file.drop(columns=["legacy_id", "source_name"]),
    on=["title", "author"],
    how="outer",
)
joint = joint[
    joint.id_nummer.isin(missing_id_nums)
    & joint.document_id.isin(missing_doc_ids)
]
mapping = pd.concat((mapping, joint[["id_nummer", "document_id"]])).dropna()
missing_id_nums = np.setdiff1d(missing_id_nums, mapping.id_nummer.unique())
missing_doc_ids = np.setdiff1d(missing_doc_ids, mapping.document_id.unique())

print(
    f"""
    Missing id_nummer: {missing_id_nums.size}
    Missing document_id: {missing_doc_ids.size}
"""
)

# %%
index_to_file.set_index("id_nummer").loc[missing_id_nums]
index_dat.set_index("document_id").loc[
    missing_doc_ids
].source_name.value_counts()

# %%
missing_new = pd.DataFrame(
    {
        "document_id": missing_doc_ids,
        "id_nummer": None,
    }
)
missing_old = pd.DataFrame(
    {
        "document_id": None,
        "id_nummer": missing_id_nums,
    }
)
mapping = pd.concat((mapping, missing_old, missing_new))
mapping

# %%
joint = mapping.merge(
    index_dat[["document_id", "author", "title"]], on="document_id", how="left"
)
joint

# %%
md = fetch_metadata()
md = md.merge(joint, how="right", on="id_nummer")
md["værk"] = md.værk.fillna(md.title)
md["forfatter"] = md.forfatter.fillna(md.author)
md = md[
    [
        "document_id",
        "forfatter",
        "værk",
        "group",
        "etnicitet",
        "tlg_genre",
        "skal_fjernes",
        "årstal",
        "geografi",
        "gender",
        "hovedperson",
        "komplet_fragment",
        "genre_first",
        "genre_second",
        "genre_second_notes",
        "id_nummer",
    ]
]
md = md.assign(duplicate_doc_id=md.document_id.duplicated(keep=False))
md = md.sort_values("document_id")
md

# %%
md[md.id_nummer.isna()]

# %%
md.to_csv("new_metadata.csv")
# %%
duplicate_id_nummer = md.document_id[
    md.id_nummer.duplicated(keep=False) & md.id_nummer.notna()
]
duplicate_doc_id = md[
    md.document_id.duplicated(keep=False) & md.document_id.notna()
].sort_values("document_id")

# %%
joint[joint.id_nummer == 640]

# %%
