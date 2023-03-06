mkdir -p environments
python3 -m venv environments/parsing

source environments/parsing/bin/activate
pip install "lxml>=4.9.0,<5.0.0"
pip install "pandas>=1.5.0,<1.6.0"
pip install "tqdm>=4.65.0,<4.66.0"

python3 src/textual_preprocessing/parse_corpus.py

deactivate
