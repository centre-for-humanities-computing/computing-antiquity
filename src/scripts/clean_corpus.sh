source environments/processing/bin/activate

MODEL_CREATOR_NAME="janko"
MODEL_NAME="grc_dep_treebanks_trf"
DEST_PATH="dat/greek/clean_data"
PROCESSED_INDEX="dat/greek/processed_data/index.csv"

MODEL_SOURCE="https://huggingface.co/${MODEL_CREATOR_NAME}/${MODEL_NAME}/resolve/main/${MODEL_NAME}-any-py3-none-any.whl"

echo "Downloading model ${MODEL_CREATOR_NAME}/${MODEL_NAME}"
python3 -m pip install $MODEL_SOURCE

python3 src/textual_preprocessing/clean_corpus.py --model $MODEL_NAME --dest $DEST_PATH --src_index $PROCESSED_INDEX

deactivate
