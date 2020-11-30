#!/bin/bash

echo 'Downloading proper python package to run Finetuning on CC !'

#INT_DIR='~/scratch/TMP'
#[ ! -d "$INT_DIR" ] && mkdir $INT_DIR
#
#mkdir $INT_DIR/tensorflow-datasets
#cd $INT_DIR/tensorflow-datasets
#pip3 download tensorflow-datasets
#
#mkdir $INT_DIR/tensorflow-hub
#cd $INT_DIR/tensorflow-hub
#pip3 download tensorflow-hubs
#
#mkdir $INT_DIR/tensorflow-metadata
#cd $INT_DIR/tensorflow-metadata
#pip3 download tensorflow-metadata

mkdir ~/python_packages
cd  ~/python_packages

mkdir tensorflow-datasets
cd  tensorflow-datasets
pip3 download tensorflow-datasets
cd ..

mkdir tensorflow-hub
cd  tensorflow-hub
pip3 download tensorflow-hub
cd ..

mkdir tensorflow-metadata
cd  tensorflow-metadata
pip3 download tensorflow-metadata
cd ..
