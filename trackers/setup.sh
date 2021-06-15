#!/bin/sh

# Run all setup files of trackers
for d in */ ; do
	cd $d
    bash "setup.sh"
    cd ..
done