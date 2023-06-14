# MIMICIV Foundation Models

A proof of concept repository demonstrating foundation models on MIMIC-IV Built using the
[Event Stream ML](https://github.com/mmcdermott/EventStreamML) library.

# Overview

This repository demonstrates how to use the [Event Stream ML](https://github.com/mmcdermott/EventStreamML)
library for foundation modeling applications. It walks through the process in an end-to-end manner over
MIMIC-IV, including the following steps:

1. Extracting and pre-processing a dataset from a local postgresql MIMIC-IV database. This entails:
    - Running postegresql queries on MIMIC-IV
    - Extracting and processing diagnoses, laboratory test results, infusions, procedures, medications, and more
      from the results of those queries by converting entries to categorical variables, detecting and removing
      outliers, normalizing data, etc.
    - Producing a deep-learning centered data view that can be leveraged by Event Stream ML built in PyTorch
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

1. Download [Event Stream ML](https://github.com/mmcdermott/EventStreamML) into a directory on your machine.
2. Install a conda environment according to ESML's instructions.
3. Clone this repository onto your machine.
4. Copy [`.env.example`](.env.example) to a file named `.env` on your machine. Populate the environment
   variables therein for your specific system, as defined below:
5. `PROJECT_NAME`: This is a simple string name for this project. It will be used in logging, weights and
   biases, etc.
6. `PROJECT_DIR`: This is a directory to where you want output artifacts for this project to be stored
   (e.g., datasets, model runs, etc.). It is used as a base path in various scripts.
7. `EVENT_STREAM_PATH`: This is the path at which you cloned the Event Stream ML repository. It is used to
   ensure that library can be used by python files in this repository.
8. Install [rootutils](https://github.com/ashleve/rootutils) (`pip install rootutils`)

## Data

Download the MIMIC-IV dataset. To do so, follow the instructions on PhysioNet. Note that we use [MIMIC-IV
version 2.0](https://physionet.org/content/mimiciv/2.0/) in this work, though it may still work for later
versions too. For our code, you need to set up the postgres version of the database, and this postgres
database needs to be accessible in the script you run to build the dataset (detailed below).

# To build the dataset

## Building the raw dataset

Using SLURM:

```bash
./scripts/build_dataset.sh \
  hydra/launcher=submitit_slurm \
  hydra.launcher.partition=short \
  hydra.launcher.cpus_per_task=15 \
  hydra.launcher.mem_gb=100 \
  cohort_name=??? \
  --multirun
```

Without SLURM:

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
[template wandb report](https://wandb.ai/mmd/MIMIC_FMs/reports/Hyperparameter-Tuning-Sweep--Vmlldzo0MzUxMzc0?accessToken=gn88k9cvl4s66dsw2bupszawjgqosak5k0uqhzeb05ri3s1k43tqq85le9hluabr)
to analyze your hyperparameter tuning sweep. Simply clone the report to your project, and adjust the filters
in the various tabs of the run selector in the bottom to point to your sweep ID.

# To perform the end-to-end foundation modeling suite

## 1. Pre-train models on various subsets of the data

### 1a. Prepare your subset directories.

To do this, you can use the [`prepare_pretrain_subsets`](scripts/prepare_pretrain_subsets.sh) script. It calls
the built in ESML script `prepare_pretrain_subsets.py`, which takes as arguments the configuration variables
specified in the ESML hydra config `pretrain_subsets_base.yaml`, listed below:

```yaml
initial_model_path: ???
subset_sizes: ???
experiment_dir: null
experiment_name: "subset_experiments/${now:%Y-%m-%d_%H-%M-%S}"
seeds: 5
```

This script will copy the configuration file from the initial model directory you specify and create a set of
nested subdirectories for each of `seeds` random seeds across `subset_sizes` PT dataset subset sizes and
produce a file `${experiment_dir}/${experiment_name}/commands.txt` such that each line of that file will
contain a hydra commands that, when run, will pre-train one of the resulting subset models from scratch under
the given config. Note that the `experiment_name` parameter will be passed as an additional key to `wandb` for
tracking of these subset runs specifically.

### 1b. Launch the Pre-trained Models

Next, you merely have to launch all of the commands in the `PT_commands.txt` file produced in step 2a. This can
be a chore, but if you have access to various scheduler libraries, it can be made more simple. For example, in
this repository, we use a [sbatch SLURM script](scripts/sbatch_run_PT_subset_experiments.sh) to launch them all in parallel, using the below commands:

```bash
export EXPERIMENT_DIR=/path/to/experiment
sbatch --array=1-$(wc -l "$EXPERIMENT_DIR/PT_commands.txt" | awk '{ print $1 }') scripts/sbatch_run_PT_subset_experiments.sh $EXPERIMENT_DIR
```

## 2. Evaluate the models:

### 2a. On Generative Performance

To evaluate the pre-trained models on generative performance, use the following
[template wandb report](https://api.wandb.ai/links/mmd/m5unsyll)

### 2b. On Zero-shot Performance

To evaluate the pre-trained models on zero-shot, forecasting-based performance, you need to take several
steps:

1. Produce an appropriate python labeling class, which can take as input a batch of data and a configuration
   file and predict your task's label.
2. You need to copy this labeling class definition file (all necessary functions and such used by the class
   must live in that single file) into the data directories task dataframes subfolder with the name
   `${task_df_name}_labeler.py`:
   `cp task_labelers/readmission.py $DATA_DIR/task_dfs/readmission_30d_all_labeler.py`
3. You can then produce and run the zero-shot evaluation commands over your pre-trained models and tasks!
   Constructing and running the commands uses much the same commands as does the pre-training and fine-tuning
   task setups. To prepare the zero-shot directories and run configs, you can run
   [`./scripts/prepare_zero_shot_subsets.sh`](scripts/prepare_zero_shot_subsets.sh) Running them can be done
   with the below command as well.

```bash
export EXPERIMENT_DIR=/path/to/experiment
sbatch --array=1-$(wc -l "$EXPERIMENT_DIR/zero_shot_commands.txt" | awk '{ print $1 }') scripts/sbatch_run_zero_shot_subset_experiments.sh $EXPERIMENT_DIR
```

Few and zero-shot performance can be further evaluated via this [template wandb report](https://api.wandb.ai/links/mmd/jirgxdir)

### 2c. On Few-shot Transfer Learning Performance

To do this, you need to run the [`prepare_finetune_subsets`](scripts/prepare_finetune_subsets.sh) script, much
like for 2a, then you merely have to launch all of the commands in the `FT_commands.txt` file.
We use a [sbatch SLURM script](scripts/sbatch_run_PT_subset_experiments.sh) to launch them all in parallel,
using the below commands:

```bash
export EXPERIMENT_DIR=/path/to/experiment
sbatch --array=1-$(wc -l "$EXPERIMENT_DIR/FT_commands.txt" | awk '{ print $1 }') scripts/sbatch_run_FT_subset_experiments.sh $EXPERIMENT_DIR
```

Few and zero-shot performance can be further evaluated via this [template wandb report](https://api.wandb.ai/links/mmd/jirgxdir)

### 2d. At Unsupervised Data Understanding

This evaluation layer consists of several steps: getting embeddings, producing clusters, and producing the
aligned dendogram.

#### 2d.i Generated Embeddings

Much as before, use the [`prepare_get_embeddings_subsets`](scripts/prepare_get_embeddings_subsets.sh) script,
then launch all the commands, again for example via an
[sbatch SLURM script](scripts/sbatch_run_get_embeddings_subset_expreiments.sh):

```bash
export EXPERIMENT_DIR=/path/to/experiment
sbatch --array=1-$(wc -l "$EXPERIMENT_DIR/get_embeddings_commands.txt" | awk '{ print $1 }') scripts/sbatch_run_get_embeddings_subset_experiments.sh $EXPERIMENT_DIR
```
