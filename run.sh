#!/bin/bash
#SBATCH --time=0-10:00
#SBATCH --gres=gpu:p100:2
#SBATCH --cpus-per-task=8
#SBATCH --account=def-bengioy
#SBATCH --mem=20G

echo 'Copying and unpacking dataset on local compute node...'
tar -xf ~/scratch/data/images-224.tar -C $SLURM_TMPDIR
cp ~/scratch/data/Data_Entry_2017.csv $SLURM_TMPDIR

echo ''
echo 'Starting task !'
echo 'Load Modules Python !'
# module load arch/avx512 StdEnv/2018.3
# nvidia-smi
module load python/3.7
module load scipy-stack
#module load cuda cudnn

echo 'Creating VENV'
virtualenv --no-download $SLURM_TMPDIR/env

echo 'Source VENV'
source $SLURM_TMPDIR/env/bin/activate
echo 'Installing package'
# pip3 install --no-index --upgrade pip
pip3 install --no-index pyasn1
echo -e 'Installing tensorflow_gpu-hub ******************************\n'
pip3 install --no-index tensorflow_gpu
echo -e 'Installing TensorFlow-hub ******************************\n'
pip3 install --no-index ~/python_packages/tensorflow_hub-0.9.0-py2.py3-none-any.whl
pip3 install --no-index tensorboard
pip3 install --no-index termcolor
pip3 install --no-index pandas
pip3 install --no-index protobuf
pip3 install --no-index termcolor
pip3 install --no-index Markdown
pip3 install --no-index h5py
pip3 install --no-index pyYAML

echo 'Calling python script'
dt=$(date '+%d-%m-%Y-%H-%M-%S');
stdbuf -oL python -u simclr-master/run.py --local_tmp_folder $SLURM_TMPDIR --train_batch_size 32 --eval_batch_size 32 --use_multi_gpus --optimizer adam --model_dir /scratch/maruel/runs/pretrain-simclr/$dt
# deactivate

