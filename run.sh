#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=4-10:00
#SBATCH --cpus-per-task=24
#SBATCH --account=def-bengioy
#SBATCH --output=pre_%j.out
#SBATCH --mem=178G
#SBATCH --gres=gpu:v100l:4

# Compute Canada Configuration
#CEDAR= --gres=gpul:v100:4 -> mem=178G or 250G || gres=gpul:p100l:4 ->  mem=120G
#GRAHAM --gres=gpu:v100:8 -> mem=178G or 377G  || gres=gpul:p100:2 ->  mem=120G
#BELUGA --gres=gpu:v100:4 -> mem=186G
format_time() {
  ((h=${1}/3600))
  ((m=(${1}%3600)/60))
  ((s=${1}%60))
  printf "%02d:%02d:%02d\n" $h $m $s
 }

echo 'Copying and unpacking dataset on local compute node...'
tar -xf ~/scratch/data/images-224.tar -C $SLURM_TMPDIR
cp ~/scratch/data/Data_Entry_2017.csv $SLURM_TMPDIR

echo ''
echo 'Starting task !'
#dt=$(date '+%d-%m-%Y-%H-%M-%S');
dt=$(date '+%d-%m-%Y-%H-%M-%S-%3N');
echo 'Time Signature: $dt'
echo $dt
pretrain_dir="/home/${1:-yancote1}/models/pretrain/"
mkdir -p $pretrain_dir
out_dir=$pretrain_dir$dt

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
if
stdbuf -oL nohup python -u ./simclr_master/run.py --data_dir $SLURM_TMPDIR \
--train_batch_size 80 \
--optimizer adam \
--model_dir $out_dir \
--use_multi_gpus \
--checkpoint_path $out_dir \
--use_blur \
--learning_rate 0.1 \
--temperature 0.5 \
--train_epochs 10 \
--proj_out_dim 256 \
--checkpoint_epochs 500 \
--train_summary_steps 50 \
--color_jitter_strength 0.5 > run_${dt}.txt 2>&1;
then
echo "Time Signature:"$dt
echo "Saving Monolytic File Archive in : ${out_dir}/run_${dt}.txt"
cp run_${dt}.txt "${out_dir}/run_${dt}.txt"

cd $pretrain_dir
echo "PWD"
echo $PWD
tar -cvf $dt.tar.gz $dt

fi
echo $dt
echo "Script completed in $(format_time $SECONDS)"
echo 'PreTraining Completed !!! '