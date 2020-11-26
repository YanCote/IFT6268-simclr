#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=1-10:00
#SBATCH --gres=gpu:v100:8
#SBATCH --cpus-per-task=28
#SBATCH --account=def-bengioy
#SBATCH --output=pre_%j.out
#SBATCH --mem=178G

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
pip3 install --no-index ~/python_packages/tensorflow-hub/tensorflow_hub-0.9.0-py2.py3-none-any.whl
pip3 install --no-index tensorboard
pip3 install --no-index termcolor
pip3 install --no-index pandas
pip3 install --no-index protobuf
pip3 install --no-index termcolor
pip3 install --no-index Markdown
pip3 install --no-index h5py
pip3 install --no-index pyYAML
pip3 install --no-index scikit-learn

echo 'Calling python script'
dt=$(date '+%d-%m-%Y-%H-%M-%S');
echo dt
stdbuf -oL python -u ./simclr_master/run.py --data_dir $SLURM_TMPDIR --train_batch_size 64 \
--eval_batch_size 64 --use_multi_gpus --optimizer adam --model_dir /home/yancote1/pretraining/$dt \
--temperature 0.4 --train_epochs 300 --checkpoint_epochs 50 --weight_decay=0.0 --warmup_epochs=0 \
--color_jitter_strength 0.5  >> /home/yancote1/pretraining/$dt/run_$dt.out
mv out_%j.out /home/yancote1/pretraining/$dt
