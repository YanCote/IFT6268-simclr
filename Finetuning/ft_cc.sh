#!/bin/bash
#SBATCH --time=00:40:00
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=1
#SBATCH --account=def-bengioy
#SBATCH --mem=8G
#SBATCH --output=out_%j.out



echo 'List1'
ls -l -d ~/scratch/*/
echo 'List2'
ls -l -d ~/*/
echo 'List3'
ls -l -d ~/$SCRATCH/tf_flowers/*/


echo 'Starting task !'
echo 'Load Modules Python !'
module load python/3.7
module load scipy-stack
#module load cuda cudnn

~/python_
echo 'Creating VENV'

echo 'Creating VENV'
virtualenv --no-download $SLURM_TMPDIR/env

echo 'Source VENV'
source $SLURM_TMPDIR/env/bin/activate
echo 'Installing package'
# pip3 install --no-index --upgrade pip
pip3 install --no-index pyasn1
echo -e 'Installing tensorflow_gpu-hub ******************************\n'
pip3 install --no-index tensorflow_cpu
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

echo -e 'Installing TensorFlow-Datasets ******************************\n'
pip3 install --no-index ~/python_packages/tensorflow-datasets/googleapis_common_protos-1.52.0-py2.py3-none-any.whl
pip3 install --no-index ~/python_packages/tensorflow-metadata/absl_py-0.10.0-py3-none-any.whl
pip3 install --no-index ~/python_packages/tensorflow-datasets/promise-2.3
echo -e 'Installing tensorflow_metadata ******************************\n'
pip3 install --no-index ~/python_packages/tensorflow-metadata/tensorflow_metadata-0.24.0-py3-none-any.whl
echo -e 'Installing tensorflow_datasets ******************************\n'
pip3 install --no-index ~/python_packages/tensorflow-datasets/zipp-3.4.0-py3-none-any.whl
pip3 install --no-index ~/python_packages/tensorflow-datasets/tensorflow_datasets-4.0.1-py3-none-any.whl

echo 'Calling python script'
stdbuf -oL python -u finetuning.py
# deactivate