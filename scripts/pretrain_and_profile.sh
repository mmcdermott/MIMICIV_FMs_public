#!/usr/bin/env bash
# shellcheck disable=SC2002,SC2086,SC2046
export $(cat .env | xargs)

N_RUNS=$1
ESGPT_COMMIT=$3
NRT_COMMIT=$4
BASE_EXP_DIR="$2/${ESGPT_COMMIT:0:8}_${NRT_COMMIT:0:8}"
DATA_DIR=$5
shift
shift
shift
shift
shift

echo "Running $N_RUNS runs of pretrain.sh with ESGPT commit $ESGPT_COMMIT and NRT commit $NRT_COMMIT"
echo "Making experiment directory $BASE_EXP_DIR"
mkdir -p $BASE_EXP_DIR
echo "Setting permissions on $BASE_EXP_DIR"
chmod go-rw $BASE_EXP_DIR

echo "Copying ESGPT code to $BASE_EXP_DIR/ESGPT_code"
cp -r $EVENT_STREAM_PATH $BASE_EXP_DIR/ESGPT_code

current_dir=$(pwd)

echo "ESGPT_COMMIT=$ESGPT_COMMIT\nNRT_COMMIT=$NRT_COMMIT" > $BASE_EXP_DIR/commits.txt

cd $BASE_EXP_DIR/ESGPT_code
echo "Checking out ESGPT commit $ESGPT_COMMIT in $(pwd)"
git checkout $ESGPT_COMMIT

cd $current_dir
echo "Back in $(pwd)"

export EVENT_STREAM_PATH=$BASE_EXP_DIR/ESGPT_code
echo "Set EVENT_STREAM_PATH to ${EVENT_STREAM_PATH}"

export PYTHONPATH="$EVENT_STREAM_PATH:$PYTHONPATH"
echo "Set PYTHONPATH to ${PYTHONPATH}"

# Check if NRT_COMMIT is not None before copying NRT code
if [ "$NRT_COMMIT" != "None" ]; then
  echo "Copying NRT code to $BASE_EXP_DIR/NRT_code"
  cp -r /n/data1/hms/dbmi/zaklab/mmd/nested_ragged_tensors $BASE_EXP_DIR/NRT_code

  cd $BASE_EXP_DIR/NRT_code
  echo "Checking out NRT commit $NRT_COMMIT in $(pwd)"
  git checkout $NRT_COMMIT
  echo "Installing using pip at $(which pip)"
  pip install -e .

  cd $current_dir
  echo "Back in $(pwd)"

  export NRT_PATH=$BASE_EXP_DIR/NRT_code
  echo "Set NRT_PATH to ${NRT_PATH}"

  export PYTHONPATH="$NRT_PATH:$PYTHONPATH"
  echo "Set PYTHONPATH to ${PYTHONPATH}"
fi


echo "Copying data from $DATA_DIR to $BASE_EXP_DIR/data"
cp -r $DATA_DIR $BASE_EXP_DIR/data


for RUN in $(seq 1 $N_RUNS)
do
  # Set CACHE_STATUS based on the run number
  if [ "$RUN" -eq 1 ]; then
      CACHE_STATUS="CACHING"
  else
      CACHE_STATUS="PRE_CACHED"
  fi

  # Set the run name
  RUN_NAME="pretrain_compute_exp_${ESGPT_COMMIT:0:8}_${NRT_COMMIT:0:8}_run_${RUN}"

  EXP_DIR="$BASE_EXP_DIR/$RUN"
  LOG_DIR="$EXP_DIR/.logs"
  echo "Running command:\n
      mprof run --include-children --exit-code --output $LOG_DIR/mprofile.dat \
          ./scripts/pretrain.sh experiment_dir=$EXP_DIR save_dir=$EXP_DIR \
          data_config.save_dir=$BASE_EXP_DIR/data \
          ++wandb_logger_kwargs.name=$RUN_NAME \
          ++wandb_experiment_config_kwargs.cache_status=$CACHE_STATUS \
          ++wandb_experiment_config_kwargs.esgpt_commit=${ESGPT_COMMIT:0:8} \
          ++wandb_experiment_config_kwargs.nested_ragged_tensor_commit=${NRT_COMMIT:0:8} \
          $@"
  mkdir -p $LOG_DIR
  { time \
      mprof run --include-children --exit-code --output "$LOG_DIR/mprofile.dat" \
          ./scripts/pretrain.sh experiment_dir="$EXP_DIR" save_dir="$EXP_DIR" \
          data_config.save_dir="$BASE_EXP_DIR/data" \
          "++wandb_logger_kwargs.name=$RUN_NAME" \
          "++wandb_experiment_config_kwargs.cache_status=$CACHE_STATUS" \
          "++wandb_experiment_config_kwargs.esgpt_commit=${ESGPT_COMMIT:0:8}" \
          "++wandb_experiment_config_kwargs.nested_ragged_tensor_commit=${NRT_COMMIT:0:8}" \
          "$@" \
      2> $LOG_DIR/cmd.stderr
  } 2> $LOG_DIR/timings.txt

  cmd_exit_status=${PIPESTATUS[0]}
  # Check the exit status of the second command in the pipeline (mprof run ...)
  if [ -n "$cmd_exit_status" ] && [ "$cmd_exit_status" -ne 0 ]; then
    echo "pretrain.sh failed with status $cmd_exit_status."
    echo "Stderr from pretrain.sh (see $LOG_DIR/cmd.stderr):"
    tail $LOG_DIR/cmd.stderr
    exit $cmd_exit_status
  fi

  # Get the size of the cached files
  du -sh $BASE_EXP_DIR/data/DL_reps > $LOG_DIR/final_data_size.txt

  # Memory profiling
  mprof peak $LOG_DIR/mprofile.dat > $LOG_DIR/peak_memory_usage.txt
  mprof plot -o $LOG_DIR/mprofile.png $LOG_DIR/mprofile.dat
done
