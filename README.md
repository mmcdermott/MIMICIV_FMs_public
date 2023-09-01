# MIMICIV Foundation Models

A proof of concept repository demonstrating foundation models on MIMIC-IV Built using the
[Event Stream GPT](https://github.com/mmcdermott/EventStreamGPT) library. See [this
tutorial](https://eventstreamml.readthedocs.io/en/dev/MIMIC_IV_tutorial/index.html) in that repository for a
walk through of the components of this repository!

# Overview

This repository demonstrates how to use the [Event Stream GPT](https://github.com/mmcdermott/EventStreamGPT)
library for foundation modeling applications. It walks through the process in an end-to-end manner over
MIMIC-IV, including the following steps:

1. Extracting and pre-processing a dataset from a local postgresql MIMIC-IV database. This entails:
    - Running postegresql queries on MIMIC-IV
    - Extracting and processing diagnoses, laboratory test results, infusions, procedures, medications, and more
      from the results of those queries by converting entries to categorical variables, detecting and removing
      outliers, normalizing data, etc.
    - Producing a deep-learning centered data view that can be leveraged by Event Stream GPT built in PyTorch
      dataset classes.
    - Building task-specific datasets for in-hospital mortality and 30-day readmission risk prediction.
2. Pre-training generative, point-process transformers over this modality via a negative log likelihood loss.
   The final result of this step is a generative model spanning all modalities extracted in the data,
   continuous, categorical, and temporal alike. In addition to static training of models, this repository also
   illustrates how to perform hyperparameter tuning on those models across a variety of parameters via Weights
   and Biases.
3. Evaluating those pre-trained models via
    - Pure generative performance.
    - Fine-tuning performance across in-hospital mortality and readmission risk prediction.
    - Zero-shot performance across in-hospital mortality and readmission risk prediction.

# Set-up

## Software

1. Download [Event Stream GPT](https://github.com/mmcdermott/EventStreamGPT) into a directory on your machine.
2. Install a conda environment according to ESGPT's instructions.
3. Clone this repository onto your machine.
4. Copy [`.env.example`](.env.example) to a file named `.env` on your machine. Populate the environment
   variables therein for your specific system, as defined below:
5. `PROJECT_NAME`: This is a simple string name for this project. It will be used in logging, weights and
   biases, etc.
6. `PROJECT_DIR`: This is a directory to where you want output artifacts for this project to be stored
   (e.g., datasets, model runs, etc.). It is used as a base path in various scripts.
7. `EVENT_STREAM_PATH`: This is the path at which you cloned the Event Stream GPT repository. It is used to
   ensure that library can be used by python files in this repository.
8. Install [rootutils](https://github.com/ashleve/rootutils) (`pip install rootutils`)

## Data

Download the MIMIC-IV dataset. To do so, follow the instructions on PhysioNet. Note that we use [MIMIC-IV
version 2.0](https://physionet.org/content/mimiciv/2.0/) in this work, though it may still work for later
versions too. For our code, you need to set up the postgres version of the database, and this postgres
database needs to be accessible in the script you run to build the dataset (detailed below).

# To build the dataset

## Building the raw dataset

```bash
./scripts/build_dataset.sh cohort_name=???
```

You can visualize the dataset with the [`notebooks/Visualize ESD.ipynb`](notebooks/Visualize%20ESD.ipynb)
notebook.

## Building task dataframes

To build task dataframes, use the
[`notebooks/Build Task DataFrames.ipynb`](notebooks/Build%20Task%20DataFrames.ipynb) notebook. Stay tuned for
future developments here where tasks can also be specified via configuration files, rather than in raw code.

To see the zero-shot task labelers for these tasks, see the [`task_labelers`](task_labelers) folder.

# To hyperparameter tune pre-training model parameters

## Initiate the `wandb` Sweep:

Run the [`./scripts/launch_hyperparameter_tuning.sh`](scripts/launch_hyperparameter_tuning.sh) script from the
root directory of this repository with the appropriate conda env activated.

To adjust any parameter choices, modify the
[`hyperparameter_sweep` config file](configs/hyperparameter_sweep.yaml).

This script will initiate a Weights and Biases Sweep over this parameter space.

## Run Sweep Agents:

To run agents to execute the models in the sweep, run
[`./scripts/launch_wandb_agent.sh $WANDB_USERNAME $SWEEP_ID`](scripts/launch_wandb_agent.sh), again from the
root directory of this repository with the appropriate conda env. Note that you will need to populate
`$WANDB_USERNAME` and `$SWEEP_ID` yourself, the latter of which is printed to the command line when you run
the sweep initiation script.

Launching a wandb agent kicks off a process which will repeatedly fetch model parameters from the sweep server
and run them locally, in the environment where the agent is started. So, if you want to control this
environment (e.g., have it run in a slurm job, have it only have access to a select GPU, etc.), make sure that
you begin the job in the appropriate manner to do so.

## Analyzing Hyperparameter Tuning Results

You can use this
[template wandb report](https://wandb.ai/mmd/MIMIC_FMs_Public/reports/Hyperparameter-Tuning-Sweep--Vmlldzo0NjM3MDg1?accessToken=c5g4i8ba2solm7k92j0id9ihm3w9or0uuh50wshhuop42bcioksm0f40teeqd8yu)
to analyze your hyperparameter tuning sweep. Simply clone the report to your project, and adjust the filters
in the various tabs of the run selector in the bottom to point to your sweep ID.

## Fine-tuning Results
You can also run sklearn baseline, from scratch supervised neural network training, or fine-tuning
hyperparameter searches using the respective scripts and configs in the ESGPT library. See [this
tutorial](https://eventstreamml.readthedocs.io/en/dev/MIMIC_IV_tutorial/index.html) for more details!

## Evaluating On Zero-shot Performance

To evaluate the pre-trained models on zero-shot, forecasting-based performance, you need to take several
steps:

1. Produce an appropriate python labeling class, which can take as input a batch of data and a configuration
   file and predict your task's label.
2. You need to copy this labeling class definition file (all necessary functions and such used by the class
   must live in that single file) into the data directories task dataframes subfolder with the name
   `${task_df_name}_labeler.py`:
   `cp task_labelers/readmission.py $DATA_DIR/task_dfs/readmission_30d_all_labeler.py`
3. You can then produce and run the zero-shot evaluation commands over your pre-trained model and tasks! To do
   so, simply use the `scripts/run_zero_shot_eval.sh` script.
