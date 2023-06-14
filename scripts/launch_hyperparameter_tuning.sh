#!/usr/bin/env bash
# shellcheck disable=SC2002,SC2086,SC2046
export $(cat .env | xargs)

PYTHONPATH="$EVENT_STREAM_PATH:$PYTHONPATH" python \
  $EVENT_STREAM_PATH/scripts/launch_wandb_hp_sweep.py \
  --config-path=$(pwd)/configs \
  --config-name=hyperparameter_sweep \
  "hydra.searchpath=[$EVENT_STREAM_PATH/configs]" "$@"
