#!/bin/sh

# Create and activate env
conda create -y --name sort python=3.6
source activate sort

# Install requirements
pip install numpy
pip install -r requirements.txt
