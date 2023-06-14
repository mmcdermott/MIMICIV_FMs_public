#!/usr/bin/env bash

cd /path/to/repository/MIMICIV_FMs_public

export $(cat .env | xargs)

# Activate conda environment

PYTHONPATH="$EVENT_STREAM_PATH:$PYTHONPATH" \
  python $EVENT_STREAM_PATH/scripts/zeroshot.py \
  load_from_model_dir="$1" \
  task_df_name=$2 \
  task_specific_params.num_samples=3 ++data_config_overrides.do_include_start_time_min=True \
  ++data_config_overrides.seq_padding_side=left ++data_config_overrides.max_seq_len=128
