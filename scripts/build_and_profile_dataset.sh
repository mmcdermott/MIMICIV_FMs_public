#!/usr/bin/env bash
export $(cat .env | xargs)

COHORT_NAME_BASE=$1 
N_RUNS=$2
for RUN in $(seq 1 $N_RUNS)
do
  COHORT_NAME="$COHORT_NAME_BASE-$RUN"
  LOG_DIR="$PROJECT_DATA_DIR/$COHORT_NAME/.logs"
  mkdir -p $LOG_DIR
  { time \
      mprof run --include-children --exit-code --output "$LOG_DIR/mprofile.dat" \
          ./scripts/build_dataset.sh cohort_name="$COHORT_NAME" do_overwrite=True \
      2> $LOG_DIR/cmd.stderr 
  } 2> $LOG_DIR/timings.txt

  cmd_exit_status=${PIPESTATUS[0]}
  # Check the exit status of the second command in the pipeline (mprof run ...)
  if [ -n "$cmd_exit_status" ] && [ "$cmd_exit_status" -ne 0 ]; then
    echo "build_dataset.sh failed with status $cmd_exit_status."
    echo "Stderr from build_dataset.sh (see $LOG_DIR/cmd.stderr):"
    tail $LOG_DIR/cmd.stderr
    exit $cmd_exit_status
  fi
  mprof plot -o $LOG_DIR/mprofile.png $LOG_DIR/mprofile.dat
  mprof peak $LOG_DIR/mprofile.dat > $LOG_DIR/peak_memory_usage.txt
done
