#!/bin/bash
mkdir -p ./remote_log/cedar
mkdir -p ./remote_log/graham
echo 'Downloading Tensor Board and ML Flow from Cedar !'
scp -r -p yancote1@cedar.calculcanada.ca:/home/yancote1/tb/ ./remote_log/cedar
scp -r -p yancote1@cedar.calculcanada.ca:/home/yancote1/mlruns/ ./remote_log/cedar

echo 'Downloading Tensor Board and ML Flow from Graham !'
scp -r -p yancote1@cedar.calculcanada.ca:/home/yancote1/tb/ ./remote_log/cedar
scp -r -p yancote1@cedar.calculcanada.ca:/home/yancote1/mlruns/ ./remote_log/cedar

echo 'Running Tensorboard CTRL+C to Stop !'
tensorboard --logdir='remote_log'

echo 'Running ML Flow CTRL+C to Stop !'
cd remote_log
mlflow ui
cd ../
