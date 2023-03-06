# Computing Antiquity

This repository contains scripts for fetching and preprocessing data as well as all analysis scripts in the 
"Computing Antiquity" project.

## Usage

For fetching and parsing the corpus run:

```bash
bash src/scripts/get_corpus.py
```

> Note that some Septuigant texts do not get fetched by the scripts
> as they are closed source and cannot be disclosed to outside parties.
> You have to manually insert `SEPA.zip` to `dat/greek/` before running the script.

For installing spaCy GPU dependencies run:

```bash
bash src/scripts/init_gpu_server.sh
bash src/install_processing_env.sh
```

For preprocessing texts and saving them as spaCy DocBins run:

```bash
bash src/scripts/spacy_process_corpus.py
```

