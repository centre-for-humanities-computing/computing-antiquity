# Computing Antiquity

This repository contains scripts for fetching and preprocessing data as well as all analysis scripts in the 
"Computing Antiquity" project.

> Please note that everything is designed for usage in Debian based system with a venv-compatible
> Python distribution and bash shell.

## Usage

### Data fetching

To fetch the corpus run:

```bash
bash src/scripts/get_corpus.py
```

This command will put all texts from the corpus in raw XML format in `dat/greek/raw_data/`.

> Note that some Septuigant texts do not get fetched by the scripts
> as they are closed source and cannot be disclosed to outside parties.
> You have to manually insert `SEPA.zip` to `dat/greek/` before running the script.

### Parsing

To Parse the XML files into raw text:

```bash
bash src/scripts/parse_corpus.py
```

This command will put all texts from the corpus in raw txt format in `dat/greek/parsed_data/`
All files will follow this naming convention: `<source_corpus>/<corpus_specific_id>.txt`
Index of destinations, document ids and source files will be found in `dat/greek/parsed_data/index.csv`

### Processing

This step processes all documents with odyCy and saves them as DocBins.
The code requires an environment where an Nvidia GPU can be used.
It is specifically designed for usage in the Ubuntu(CUDA+Jupyter) Virtual Machine on AAU in Ucloud.

For installing spaCy GPU dependencies run:

```bash
bash src/scripts/init_gpu_server.sh
sudo reboot

# After reboot
bash src/install_processing_env.sh
```

For preprocessing texts and saving them as spaCy DocBins run:

```bash
bash src/scripts/spacy_process_corpus.py
```

This will save everything in `dat/greek/processed_data/`. All files will have their document id as name and .spacy extension.
Index of files can be accessed under `dat/greek/processed_data/index.csv`

### Cleaning

This step cleans texts by normalizing them and removing non-greek tokens.
It produces the following corpora (under `dat/greek/cleaned_data`):
  - Normalized with stopwords (`with_stopwords.csv`)
  - Normalized without stopwords (`without_stopwords.csv`)
  - Lemmatized with stopwords (`lemmatized_with_stopwords.csv`)
  - Lemmatized without stopwords (`lemmatized_without_stopwords.csv`)

These files include a `document_id` and a `text` column.
Texts are represented as follows:
  - Sentences are separated by newlines
  - Tokens in sentences are separated by spaces

```bash
bash src/scripts/clean_corpus.py
```

 > It is recommended that you do these steps consecutively on the same server, as they use the same environment.

