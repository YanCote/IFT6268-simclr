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
dt=$(date '+%d-%m-%Y-%H-%M-%S');
echo 'Time Signature: $dt'
pretrain_dir=/home/yancote1/models/pretrain/
mkdir -p $pretrain_dir
out_dir=$pretrain_dir/$dt
exec > $out_dir/run1_$dt.txt

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


echo $out_dir
# --use_multi_gpus
stdbuf -oL python ./simclr_master/run.py --data_dir $SLURM_TMPDIR \
--train_batch_size 2 \
--optimizer adam \
--model_dir $out_dir \
--checkpoint_path $out_dir \
--temperature 0.4 --train_epochs 1 --checkpoint_epochs 50 --weight_decay=0.0 --warmup_epochs=0 \
--color_jitter_strength 0.5 > out.txt 2>&1
#$out_dir/run2_$dt.txt
cd $pretrain_dir
tar -zcvf $dt.tar.gz $out_dir
#--remove-files
echo 'PreTraining Completed !!! '