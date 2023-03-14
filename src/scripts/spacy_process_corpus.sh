source environments/processing/bin/activate

MODEL_CREATOR_NAME="janko"
MODEL_NAME="grc_dep_treebanks_trf"
DEST_PATH="dat/greek/processed_data"
SRC_INDEX="dat/greek/parsed_data/index.csv"

MODEL_SOURCE="https://huggingface.co/${MODEL_CREATOR_NAME}/${MODEL_NAME}/resolve/main/${MODEL_NAME}-any-py3-none-any.whl"


echo "Logging into Wandb:"
python3 -m wandb login

echo "Downloading model ${MODEL_CREATOR_NAME}/${MODEL_NAME}"
python3 -m pip install $MODEL_SOURCE

echo "Starting processing in the background."
nohup python3 -u src/textual_preprocessing/process_corpus.py --model $MODEL_NAME --dest $DEST_PATH --src_index $SRC_INDEX &

deactivate
