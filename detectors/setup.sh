#!/bin/sh

# Create env
conda env create -f environment.yml


# Run all setup files of detectors
for d in */ ; do
	cd $d
    bash "setup.sh"
    cd ..
done







# conda install -c conda-forge opencv