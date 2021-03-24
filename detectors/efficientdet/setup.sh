#!/bin/sh

# Available models
# ['efficientdet-d0', 'efficientdet-d1', 'efficientdet-d2', 'efficientdet-d3', 'efficientdet-d4', 'efficientdet-d5', 'efficientdet-d6', 'efficientdet-d7', 'efficientdet-d7x']

# Change also in efficientdet.py
MODEL=efficientdet-d2

# Batch need to be the same as in Inference
BATCH=5

source activate detec

python setup.py --model $MODEL --batch $BATCH
