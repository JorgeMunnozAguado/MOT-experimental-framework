#!/bin/sh


# Setup detectors environment. (and download models)
cd detectors
bash setup.sh
cd ..


# Download datasets.
cd dataset
bash setup.sh
cd ..


# Setup trackers environment. (and download models)
cd trackers
bash setup.sh
cd ..


# Setup evaluation environment.
cd evaluation
bash setup.sh
cd ..


# Setup analysis environment.
cd analysis
bash setup.sh
cd ..






