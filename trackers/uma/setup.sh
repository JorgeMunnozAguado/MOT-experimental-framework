#!/bin/sh


# Dowload models
mkdir models
mkdir models/npair0.1-id0.1-se_block2
cd models/npair0.1-id0.1-se_block2

PAHT_DOWNLOAD=https://rachaelhome.synology.me/UMA
PAHT_DOWNLOAD=https://raw.githubusercontent.com/yinjunbo/UMA-MOT/master/UMA-TEST/models/npair0.1-id0.1-se_block2

wget ${PAHT_DOWNLOAD}/checkpoint
wget ${PAHT_DOWNLOAD}/model.ckpt-332499.data-00000-of-00001
wget ${PAHT_DOWNLOAD}/model.ckpt-332499.index
wget ${PAHT_DOWNLOAD}/model.ckpt-332499.meta
wget ${PAHT_DOWNLOAD}/model_config.json
wget ${PAHT_DOWNLOAD}/track_config.json
wget ${PAHT_DOWNLOAD}/train_config.json

cd ../..


# Create and activate env
conda create -y --name uma python=3.6
source activate uma

# Install requirements
pip install numpy
pip install -r requirements.txt