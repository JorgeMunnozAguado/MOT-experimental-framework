#!/bin/sh

# Download datasets.
cd dataset
sh download_data.sh
cd ..

# Setup detectors enviroment. (and download models)
cd detectors
sh setup-env.sh
cd ..










