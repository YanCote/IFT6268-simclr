#!/bin/bash

echo "Copying files to scratch folder..."
rsync -av --exclude=".*" . ~/scratch/IFT6268-simclr

echo "Copying Data to scratch folder..."
mkdir ~/scratch/data
#rsync -av --exclude=".*" ~/data/Data_Entry_2017.csv ~/scratch/data/Data_Entry_2017.csv
#rsync -av --exclude=".*" ~/data/images-224 ~/scratch/data/images-224

cd ~/scratch/IFT6268-simclr

echo "Lunching sbatch job..."
sbatch train_on_slurm.sh

echo ""
echo "For more details on the job use: sq"
echo "To see progress live use: tail -f ~/scratch/IFT6268-simclr/slurm-.out"

