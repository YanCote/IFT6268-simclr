# Project IFT6268 - Exploration Path on SimCLRv2 
## Exploration Path on SimCLRv2 - Big Self-Supervised Models are Strong Semi-Supervised Learners 


<div align="center">
  <img width="50%" alt="SimCLR Illustration" src="https://1.bp.blogspot.com/--vH4PKpE9Yo/Xo4a2BYervI/AAAAAAAAFpM/vaFDwPXOyAokAC8Xh852DzOgEs22NhbXwCLcBGAsYHQ/s1600/image4.gif">
</div>
<div align="center">
  An illustration of SimCLR (from <a href="https://ai.googleblog.com/2020/04/advancing-self-supervised-and-semi.html">our blog here</a>).
</div> <br>

[Original Google SimCLR Git Repo](https://github.com/google-research/simclr) <br>


## Project description

Project looks at low data and compute regime as well as how it generalize well on other dataset.
### Methodology
....

## Environment setup

Conda ENV
- conda create --name simclr python=3.7
- pip install -r requirement.txt

Download XRAY(2.5GB): <https://academictorrents.com/details/e615d3aebce373f1dc8bd9d11064da55bdadede0>
Download Google Models:
- install google sdk https://cloud.google.com/sdk/docs/downloads-versioned-archives
- gsutil cp -r 'gs://simclr-checkpoints/simclrv2/finetuned_100pct/r50_1x_sk0/hub/'. Make sure checkpoint are named correctly if there are import errors 
    - renamed saved_model.pb tfhub_model.pb and  for variables

Both graham and cedar are used in the project <https://docs.computecanada.ca/wiki/Compute_Canada_Documentation/fr>
## Pre-Training on XRAY

Pre-Training is achieved using run.py. Locally, There's a template for parameter un launch_template.json which could be use with VSCode.
ft_cc.sh is used to launch the script on compute node.

Every Run generate a Monolithic output such as archived and named using datetime:
- One or several Checkpoint
- A final HUB file
- FLAGs(active arguments) in a text and pickle file
- TensorBoard Files.
- Run log in a human readable format *.txt

>In:  XRAY Dataset <br>
>Out: XRAY PreTrain Monolithic output

*scripts/down_pretrain_models.sh username*: download pretrained models locally
*scripts/sync_scratch.sh*: Use on Compute Canada to sync home with Scratch
*scripts/initial_down_whl.sh*: script to download the whl file for packages not available on CC

## FineTuning and validation

Must run Finetuning/finetuning.py and use config.yml as a template.
>In:  XRAY PreTrain Monolithic output or Google Pretrained
>Out: Monolithic output

Every Run generate a Monolithic output such as archived and named using datetime:
- One or several Checkpoint
- A final HUB
- MLFLow information merge in Project Finetuning/mlruns
- TensorBoard Files.
- 

Author: Shannel Gauthier, Marc-Andre Ruel, Yan Cote

