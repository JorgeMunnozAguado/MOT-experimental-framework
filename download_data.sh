#!/bin/sh

mkdir outputs

cd dataset
wget https://motchallenge.net/data/MOT20.zip
unzip MOT20.zip
rm MOT20.zip
cd MOT20
mv train/* .
rm -r test
rmdir train
cd ../..

mkdir outputs/public
mkdir outputs/public/MOT20

python dataset/MOT_structure.py
