mkdir -p environments
python3 -m venv environments/processing

source environments/processing/bin/activate
pip install "pandas>=1.5.0,<1.6.0"
pip install "tqdm>=4.65.0,<4.66.0"
pip install "spacy[cuda-autodetect]>=3.5.0,<3.6.0"
pip install "wandb>=0.13.0,<0.14.0"
pip install "plotly>=5.13.0,<5.14.0"

python3 src/textual_preprocessing/process_corpus.py

deactivate
