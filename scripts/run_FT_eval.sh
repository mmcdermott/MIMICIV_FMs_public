cd /path/to/repository/MIMICIV_FMs_public

export $(cat .env | xargs)

# Activate Env

PYTHONPATH="$EVENT_STREAM_PATH:$PYTHONPATH" \
  python $EVENT_STREAM_PATH/scripts/finetune.py \
  load_from_model_dir="$1" \
  task_df_name=$2 \
  ++data_config_overrides.train_subset_size=$3 \
  ++data_config_overrides.train_subset_seed=1
