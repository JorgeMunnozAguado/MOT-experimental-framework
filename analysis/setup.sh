#!/bin/sh

# Create the environment
conda create -y -n analysis python=3.5

source activate analysis

# Install the needed packages
conda install -y pandas
conda install -y -c conda-forge notebook

pip install matplotlib
pip install scipy
