# %%
from utils.files import load_contents
from utils.metadata import fetch_metadata

# %%
corpus = load_contents()
corpus

# %%
md = fetch_metadata()
md = md[~md.skal_fjernes]
corpus = corpus[corpus.id_nummer.isin(md.id_nummer)]
corpus = corpus.dropna()
corpus = corpus.rename(columns={"contents": "text"})
corpus
# %%
corpus.to_csv("/work/data_wrangling/dat/greek/dataset/cleaned_cltk.csv")
