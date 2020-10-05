#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:0
#SBATCH --account=def-bengioy
#SBATCH --cpus-per-task=1
#SBATCH --mem=32000M

echo 'Starting task !'
module load python/3.6

ENVDIR=~/ENV/imclass
ENVDIR=$SLURM_TMPDIR/env/imgclass

echo 'Creating VENV'
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

pip install torch --no-index
pip install torchvision --no-index
pip install matplotlib --no-index

echo 'Calling python script'
python pt_imgclass.py

deactivate
