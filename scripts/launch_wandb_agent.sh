#!/usr/bin/env bash
# shellcheck disable=SC2002,SC2086,SC2046
export $(cat .env | xargs)

PYTHONPATH="$EVENT_STREAM_PATH:$PYTHONPATH" wandb agent "$@"
