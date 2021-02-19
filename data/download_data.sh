#!/bin/sh

mkdir detections images predictions

cd images
wget https://motchallenge.net/data/MOT20.zip
unzip MOT20.zip
rm MOT20.zip
cd MOT20
mv train/* .
rm -r test
rmdir train

rm -r MOT20/*/det
