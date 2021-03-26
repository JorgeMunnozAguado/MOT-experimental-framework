#!/bin/sh

# Download datasets.
cd dataset
bash download_data.sh
cd ..


# Setup detectors enviroment. (and download models)
cd detectors
bash setup.sh
cd ..


# Download mor datasets.
cd dataset
bash download_visDrone.sh
cd ..


# Setup trackers enviroment. (and download models)
cd trackers
bash setup.sh
cd ..










