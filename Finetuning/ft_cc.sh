#!/bin/bash
#SBATCH --time=00:45:00
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=1
#SBATCH --account=def-bengioy
#SBATCH --mem=8G
#SBATCH --output=out_%j.out

echo 'Starting task !'
module load python/3.7
echo 'Load Python !'
module load scipy-stack
echo 'Load scipy'
#module load cuda cudnn

ENVDIR=~/ENV/imclass
ENVDIR=$SLURM_TMPDIR/env
#ENVDIR=~/env/

echo 'Creating VENV'
virtualenv --no-download $ENVDIR

echo 'Source VENV'
source $ENVDIR/bin/activate

echo 'Installing package'
pip install --no-index --upgrade pip
#pip install torch --no-index
#pip install torchvision --no-index
#pip install matplotlib --no-index
pip install --no-index -r cc_requirements.txt

echo 'Calling python script'
python finetuning.py

deactivate
