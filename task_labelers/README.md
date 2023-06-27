# Task Labelers for MIMIC-IV Foundation Models

## Task Labelers in General

Task labelers in the [Event Stream GPT](https://github.com/mmcdermott/EventStreamGPT) codebase are used to
enable zero-shot evaluation of a generative foundation model. At a high level, the workflow works like this:

1. Given an input and a (true) label for a downstream task, the zero-shot evaluator first generates $N$
   forecasted samples from that input forward in time using the foundation model.
2. Next, the zero-shot evaluator calls the `__call__` method on the `TaskLabeler` for that specific task on
   those generated samples. On each sample in the batch, the `TaskLabeler` returns both an empirical predicted
   label for that generated sample as well as a boolean indicating whether or not it was able to form any
   valid prediction on that sample. These empirical predictions and boolean indicators are then aggregated
   across all $N$ copies of the original input that were used in step 1 to form an empirical prediction for
   this input.
3. The zero-shot evaluator iterates over the entire dataset, collecting these empirical predictions (and
   tracking for what fraction of cases are no results able to be formed), collating performance metrics by
   comparing the empirical predictions from the `TaskLabeler` against the true labels in the input data.

Task labelers must be written by users directly, and then copied into the task directories as self-sufficient
python files (meaning python files that only import environment-level dependencies, not any other local files)
to be used in the system. They must be further named `${task_df_name}_labeler.py`. E.g.,

```bash
cp task_labelers/readmission.py $DATA_DIR/task_dfs/readmission_30d_all_labeler.py
```
