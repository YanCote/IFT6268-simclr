#!/bin/bash
#SBATCH --time=00:45:00
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=1
#SBATCH --account=def-bengioy
#SBATCH --mem=32000M
#SBATCH --output=%x-%j.out

echo 'Starting task !'
module load python/3.7
module load scipy-stack
#module load cuda cudnn

ENVDIR=~/ENV/imclass
ENVDIR=$SLURM_TMPDIR/env/imgclass

echo 'Creating VENV'
virtualenv --no-download ~/ENV

echo 'Source VENV'
source ~/ENV/bin/activate

echo 'Installing package'
pip install --no-index --upgrade pip
#pip install torch --no-index
#pip install torchvision --no-index
#pip install matplotlib --no-index
#pip install --no-index -r cc_requirements.txt

echo 'Calling python script'
python finetuning.py

deactivate
