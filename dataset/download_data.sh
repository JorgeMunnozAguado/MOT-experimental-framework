#!/bin/sh

# Download datasets
wget https://motchallenge.net/data/MOT17.zip
wget https://motchallenge.net/data/MOT20.zip

# Unzip
unzip -qq MOT17.zip
unzip -qq MOT20.zip

# Remove Zip files
rm MOT17.zip
rm MOT20.zip

# Apply to MOT20
cd MOT20
mv train/* .
rm -r test
rmdir train
cd ..

# Apply to MOT17
cd MOT17
mv train/* .
rm -r test
rmdir train
cd ..


cd ..

# Create new folders
mkdir outputs
mkdir outputs/detections
mkdir outputs/detections/public
mkdir outputs/detections/public/MOT20
mkdir outputs/detections/public/MOT17

# Save public detections
python dataset/MOT_structure.py
