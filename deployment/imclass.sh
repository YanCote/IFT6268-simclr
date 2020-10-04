#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:0              # Number of GPU(s) per node
#SBATCH --mem=500                 # increase as needed
#SBATCH --account=def-bengioy

echo 'Starting task !'
module load python/3.6

ENVDIR=~/ENV/imclass
ENVDIR=$SLURM_TMPDIR/env/imgclass

echo 'Creating VENV'
virtualenv --no-download $ENVDIR
source $ENVDIR/bin/activate
pip install torch --no-index
pip install torchvision --no-index
pip install matplotlib --no-index

echo 'Calling python script'
python pt_imgclass.py

deactivate
