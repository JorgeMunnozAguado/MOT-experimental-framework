#!/bin/sh

mkdir detections images predictions output

cd images
wget https://motchallenge.net/data/MOT20.zip
unzip MOT20.zip
rm MOT20.zip
cd MOT20
mv train/* .
rm -r test
rmdir train
cd ../..

mkdir detections/default
mkdir detections/default/MOT20

python MOT_structure.py
