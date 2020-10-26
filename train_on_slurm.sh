#!/bin/bash
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:k80:0
#SBATCH --cpus-per-task=1
#SBATCH --account=def-bengioy
#SBATCH --mem=8000M

module load python/3.7
virtualenv --no-download $SLURM_TMPDIR/env
echo $SLURM_TMPDIR
source $SLURM_TMPDIR/env/bin/activate


pip install --no-index tensorflow_gpu==2
pip install --no-index pandas
pip install --no-index opencv-python
pip install --no-index matplotlib
pip install --no-index tqdm
pip install --no-index pytz

# cp -ru /project/cq-training-1/project1/teams/team07/.keras ~/

# python --version
which python
which pip
pip freeze

echo ""
echo "Calling python train script."

stdbuf -oL python -u test.py
echo "testing done!!!!!!"

mkdir ~/IFT6268-simclr/slurm_out
rsync slurm-%a.out ~/IFT6268-simclr/slurm_out/slurm-%a.out

