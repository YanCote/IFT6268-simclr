#!/bin/bash
#SBATCH --time=00:15:00
#SBATCH --output=%N-%j.out
#SBATCH --gres=gpu:1
#SBATCH --mem=8000
#SBATCH --account=def-bengioy

ENVDIR=$SLURM_TMPDIR/env

module load python/3.6
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install torch --no-index

python pytorch-test.py

