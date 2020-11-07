#!/bin/bash
echo 'Downloading propoer python package to run Finetuning on CC !'
mkdir -p ./remote_log/cedar
mkdir -p ./remote_log/graham
scp -r -p yancote1@cedar.calculcanada.ca:/home/yancote1/tb/ ./remote_log/cedar
scp -r -p yancote1@graham.calculcanada.ca:/home/yancote1/tb/ ./remote_log/graham
tensorboard --logdir='remote_log'
