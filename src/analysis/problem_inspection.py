# %%
import re
from string import ascii_letters, digits

import cltk
import pandas as pd

from utils.files import load_contents
from utils.greek import normalize, lemmatize

# %%
id_to_path = pd.read_csv("/work/data_wrangling/dat/greek/dataset/index_to_file.csv", index_col=0)
id_to_path["path"] = "/work/data_wrangling/dat/greek/parsed_data/" + id_to_path["path"]
id_to_path = id_to_path.set_index("id_nummer")
id_to_path

# %%
contents = load_contents()
contents = contents.dropna()

# %%
tokens = contents.contents.str.split()
contents = contents.assign(tokens=tokens)
contents = contents.drop(columns="contents").explode("tokens")

# %%
contents

# %%
latin_lowercase = "[a-z]"
latin_uppercase = "[A-Z]"
latin_letters = "|".join((latin_lowercase, latin_uppercase))
digits = "[0-9]"
all_problems = "|".join((latin_letters, digits))

# %%
problems = contents[contents.tokens.str.contains(latin_letters)]
# %%
problems
# %%
text_path = id_to_path.loc[105].path
with open(text_path) as f:
    text = f.read()
text[:30]
# %%
norm_text = normalize(text, keep_sentences=False)
lemmatised_text = " ".join(lemmatize(norm_text.split()))
# %%
m = re.search(re.compile(latin_letters), norm_text)
print(m)

# %%
latin_letters
# %%
backoff = cltk.lemmatize.grc.GreekBackoffLemmatizer()

# %%
old_lemmata = pd.DataFrame({
    "token": backoff.GREEK_OLD_MODEL.keys(),
    "lemma": backoff.GREEK_OLD_MODEL.values()
})
old_lemmata[old_lemmata.lemma.str.contains(all_problems)].to_excel("backoff2_problems.xlsx")

# %%
train_lemmata = []
train_tokens = []
for sentence in backoff.train_sents:
    for token, lemma in sentence:
        train_lemmata.append(lemma)
        train_tokens.append(token)
train_lemmata = pd.DataFrame({
    "token": train_tokens,
    "lemma": train_lemmata
})
train_lemmata[train_lemmata.lemma.str.contains(all_problems)].to_excel("backoff4_problems.xlsx")
# %%
# %%
new_lemmata = pd.DataFrame({
    "token": backoff.GREEK_MODEL.keys(),
    "lemma": backoff.GREEK_MODEL.values()
})
new_lemmata = new_lemmata[new_lemmata.lemma != "punc"]
new_lemmata[new_lemmata.lemma.str.contains(all_problems)].to_excel("backoff5_problems.xlsx")
