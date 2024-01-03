#!/usr/bin/env bash
#SBATCH -c 8                               # Request one core
#SBATCH -t 0-12:00                         # Runtime in D-HH:MM format
#SBATCH -p short                          # Partition to run in
#SBATCH --mem=250GB                         # Memory total in MiB (for all cores)
#SBATCH -o run_PT_%j_sbatch.out            # File to which STDOUT will be written, including job ID (%j)
#SBATCH -e run_PT_%j_sbatch.err            # File to which STDERR will be written, including job ID (%j)

cd /n/data1/hms/dbmi/zaklab/mmd/MIMICIV_FMs_public
module load miniconda3/4.10.3

ESGPT_COMMIT=$3
NRT_COMMIT=$4

env="/n/data1/hms/dbmi/zaklab/mmd/.conda_envs/NRT_${NRT_COMMIT:0:8}"

# Check if the environment directory exists
if [ ! -d "$env" ]; then
    # If it doesn't exist, create the new conda environment
    conda create --prefix "$env" --clone ESGPT
fi

# Activate the environment
source activate "$env"

echo "Which python: $(which python)"
echo "Which pip: $(which pip)"
echo "Conda environment: $CONDA_PREFIX"

# Update PATH and run the script
PATH="$CONDA_PREFIX/bin:$PATH" scripts/pretrain_and_profile.sh "$@"
