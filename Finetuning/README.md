# Project IFT6268 - Exploration Path on SimCLRv2 
## Exploration Path on SimCLRv2 - Big Self-Supervised Models are Strong Semi-Supervised Learners 


Original Git Repo: https://github.com/google-research/simclr
Project Repo: https://github.com/YanCote/IFT6268-simclr
Pytorch lightning: https://pytorch-lightning.readthedocs.io/en/latest/
papers 

## Project description

Finetuning on SimCLRv2

## Environment setup

1. conda create --name simclr
2. pip install -r requirement.txt
3. install google sdk https://cloud.google.com/sdk/docs/downloads-versioned-archives
4. gsutil cp -r 'gs://simclr-checkpoints/simclrv2/finetuned_100pct/r50_1x_sk0/hub/'. Make sure checkpoint are named correctly if there are import errors 
    - renamed saved_model.pb tfhub_model.pb and  for variables
6. Install cuda 10.0, cudnn 7.6 ( cudnn-10.0-windows10-x64-v7.6.0.64)
