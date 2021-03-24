#!/bin/sh

# Run all setup files of detectors
for d in */ ; do
	cd $d
    bash "setup.sh"
    cd ..
done