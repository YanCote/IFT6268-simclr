#!/bin/bash
#SBATCH --partition=unkillable                # Ask for unkillable job
#SBATCH --cpus-per-task=2                     # Ask for 1 CPUs
#SBATCH --gres=gpu:1                          # Ask for 1 GPU
#SBATCH --mem=8G                             # Ask for 12 GB of RAM
#SBATCH --time=03:40:00                       # The job will run for 3 hours
#SBATCH -o /miniscratch/gauthies/slurm-%j.out  # Write the log on tmp1


echo '1. Load the required module'
#module --quiet load anaconda/3
module load python/3.7
#module load scipy-stack



echo 'Creating VENV'
virtualenv --no-download $SLURM_TMPDIR/env

echo 'Source VENV'
source $SLURM_TMPDIR/env/bin/activate
echo 'Installing package'
# pip3 install --no-index --upgrade pip
#pip3 install --no-index pyasn1
echo -e 'Installing tensorflow_gpu-hub ******************************\n'
pip3 install tensorflow_gpu==2.2.1
#pip3 install --no-index tensorflow_gpu
#pip install -r 'host_requirement.txt'

pip3 install --no-index termcolor
pip3 install --no-index protobuf
pip3 install --no-index termcolor
pip3 install --no-index Markdown
pip3 install --no-index h5py
pip3 install pyYAML==5.1.1
pip3 install cop
pip3 install tensorflow-datasets
pip3 install tensorflow-hub
pip3 install numpy
pip3 install matplotlib
pip3 install --no-index tensorboard
pip3 install pandas==1.1.0

#echo -e 'Installing TensorFlow-hub ******************************\n'
#pip3 install --no-index ~/python_packages/tensorflow-hub/tensorflow_hub-0.9.0-py2.py3-none-any.whl
#pip3 install --no-index tensorboard
#pip3 install --no-index termcolor
#pip3 install --no-index pandas

#echo '2. Load your environment'
#conda activate test_simclr
#echo -e 'Installing tensorflow_gpu-hub ******************************\n'
#pip3 install --no-index tensorflow_gpu

echo '3. Copy your dataset on the compute node'
cp /network/projects/g/gauthies/chest_xray/01_raw/images-224 $SLURM_TMPDIR
cp /network/projects/g/gauthies/chest_xray/01_raw/Data_Entry_2017.csv $SLURM_TMPDIR

echo '4. Launch your job, tell it to save the model in $SLURM_TMPDIR'
echo 'and look for the dataset into $SLURM_TMPDIR'

python finetuning.py -p $SLURM_TMPDIR #--data_path $SLURM_TMPDIR

echo '5. Copy whatever you want to save on $SCRATCH'
#cp $SLURM_TMPDIR/<to_save> /network/tmp1/<user>/
# deactivate