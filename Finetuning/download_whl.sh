#!/bin/bash

echo 'Downloading propoer python package to run Finetuning on CC !'

INT_DIR='~/scratch/TMP'
[ ! -d "$INT_DIR" ] && mkdir $INT_DIR

mkdir $INT_DIR/tensorflow-datasets
cd $INT_DIR/tensorflow-datasets
pip3 download tensorflow-datasets

mkdir $INT_DIR/tensorflow-hub
cd $INT_DIR/tensorflow-hub
pip3 download tensorflow-hubs

mkdir $INT_DIR/tensorflow-metadata
cd $INT_DIR/tensorflow-metadata
pip3 download tensorflow-metadata
