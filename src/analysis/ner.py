# %%
from cltk.ner.ner import tag_ner

from utils.files import load_contents

# %%
def ner_greek(tokens):
    return tag_ner(iso_code="grc", input_tokens=tokens)

# %%
corpus = load_contents().rename(columns={"contents": "text"})
corpus = corpus.dropna()
corpus = corpus.assign(tokens=corpus.text.str.split())
corpus = corpus.assign(ner_tag=corpus.tokens.map(ner_greek))
corpus

# %%