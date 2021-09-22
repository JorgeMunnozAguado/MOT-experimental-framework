#!/bin/sh

# Download models
#  https://drive.google.com/drive/folders/18fKzfqnqhqW3s9zwsCbnVJ5XF2JFeqMp
mkdir models
cd models

PAHT_DOWNLOAD=https://rachaelhome.synology.me/deepSort

wget ${PAHT_DOWNLOAD}/mars-small128.pb
wget ${PAHT_DOWNLOAD}/mars-small128.ckpt-68577.meta
wget ${PAHT_DOWNLOAD}/mars-small128.ckpt-68577

cd ..


# Create env
conda create -y --name deep_sort python=3.6
source activate deep_sort

pip install numpy opencv-python tensorflow-gpu Keras scipy
