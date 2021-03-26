#!/bin/sh

# Detectors
####################################


# Yolo3
#python detectors/main.py --detector yolo3 --set_data MOT20 --batch 25 --clean_log True -dvc cuda

# Yolo4
#python detectors/main.py --detector yolo4 --set_data MOT20 --batch 25 -dvc cuda

# faster_rcnn
#python detectors/main.py --detector faster_rcnn --set_data MOT20 --batch 5 -dvc cuda

# keypoint_rcnn
python detectors/main.py --detector keypoint_rcnn --set_data MOT20 --batch 3 -dvc cuda

# retinanet
python detectors/main.py --detector retinanet --set_data MOT20 --batch 3 -dvc cuda

# mask_rcnn
python detectors/main.py --detector mask_rcnn --set_data MOT20 --batch 3 -dvc cuda

# efficientdet
# python detectors/main.py --detector efficientdet --set_data MOT20 --batch 5 -dvc cuda




# Trackers
####################################
