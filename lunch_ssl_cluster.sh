#!/bin/bash

echo "Copying files to scratch folder..."
rsync -av --exclude=".*" . ~/scratch/IFT6268-simclr-gpus

cd ~/scratch/IFT6268-simclr-gpus

echo "Lunching sbatch job..."
sbatch run.sh

echo ""
echo "For more details on the job use: sq"
echo "To see progress live use: tail -f ~/scratch/IFT6268-simclr-gpus/slurm-.out"

