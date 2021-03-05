#!/bin/sh

mkdir outputs

cd dataset
sh download_data.sh
cd ..

cd detectors
sh setup-env.sh
cd ..