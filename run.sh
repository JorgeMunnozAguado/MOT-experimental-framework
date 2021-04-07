#!/bin/sh

# Detectors
####################################
# (public), yolo3, yolo4, faster_rcnn, keypoint_rcnn, retinanet, mask_rcnn, efficientdet
source activate detec

# Yolo3
python detectors/main.py --detector yolo3 --set_data MOT20 --batch 25 --clean_log True -dvc cuda
python detectors/main.py --detector yolo3 --set_data MOT17 --batch 25 -dvc cuda
python detectors/main.py --detector yolo3 --set_data VisDrone2019-MOT-val --batch 25 -dvc cuda

# Yolo4
python detectors/main.py --detector yolo4 --set_data MOT20 --batch 25 -dvc cuda
python detectors/main.py --detector yolo4 --set_data MOT17 --batch 25 -dvc cuda
python detectors/main.py --detector yolo4 --set_data VisDrone2019-MOT-val --batch 25 -dvc cuda

# faster_rcnn
python detectors/main.py --detector faster_rcnn --set_data MOT20 --batch 5 -dvc cuda
python detectors/main.py --detector faster_rcnn --set_data MOT17 --batch 5 -dvc cuda
python detectors/main.py --detector faster_rcnn --set_data VisDrone2019-MOT-val --batch 5 -dvc cuda

# keypoint_rcnn
python detectors/main.py --detector keypoint_rcnn --set_data MOT20 --batch 3 -dvc cuda
python detectors/main.py --detector keypoint_rcnn --set_data MOT17 --batch 3 -dvc cuda
python detectors/main.py --detector keypoint_rcnn --set_data VisDrone2019-MOT-val --batch 3 -dvc cuda

# retinanet
python detectors/main.py --detector retinanet --set_data MOT20 --batch 3 -dvc cuda
python detectors/main.py --detector retinanet --set_data MOT17 --batch 3 -dvc cuda
python detectors/main.py --detector retinanet --set_data VisDrone2019-MOT-val --batch 3 -dvc cuda

# mask_rcnn
python detectors/main.py --detector mask_rcnn --set_data MOT20 --batch 1 -dvc cuda
python detectors/main.py --detector mask_rcnn --set_data MOT17 --batch 1 -dvc cuda
python detectors/main.py --detector mask_rcnn --set_data VisDrone2019-MOT-val --batch 1 -dvc cuda

# efficientdet
source activate efficientdet
python detectors/main.py --detector efficientdet --name d7x --set_data MOT20 --batch 15 -dvc cuda
python detectors/main.py --detector efficientdet --name d7x --set_data MOT17 --batch 15 -dvc cuda
python detectors/main.py --detector efficientdet --name d7x --set_data VisDrone2019-MOT-val --batch 15 -dvc cuda




# Trackers
####################################
# sort, deep_sort, uma

# sort
source activate sort

python trackers/main.py --tracker sort --detector public --set_data MOT20 --clean_log True
python trackers/main.py --tracker sort --detector public --set_data MOT17
python trackers/main.py --tracker sort --detector public --set_data VisDrone2019-MOT-val

python trackers/main.py --tracker sort --detector yolo3 --set_data MOT20
python trackers/main.py --tracker sort --detector yolo3 --set_data MOT17
python trackers/main.py --tracker sort --detector yolo3 --set_data VisDrone2019-MOT-val

python trackers/main.py --tracker sort --detector yolo4 --set_data MOT20
python trackers/main.py --tracker sort --detector yolo4 --set_data MOT17
python trackers/main.py --tracker sort --detector yolo4 --set_data VisDrone2019-MOT-val

python trackers/main.py --tracker sort --detector faster_rcnn --set_data MOT20
python trackers/main.py --tracker sort --detector faster_rcnn --set_data MOT17
python trackers/main.py --tracker sort --detector faster_rcnn --set_data VisDrone2019-MOT-val

python trackers/main.py --tracker sort --detector keypoint_rcnn --set_data MOT20
python trackers/main.py --tracker sort --detector keypoint_rcnn --set_data MOT17
python trackers/main.py --tracker sort --detector keypoint_rcnn --set_data VisDrone2019-MOT-val

python trackers/main.py --tracker sort --detector retinanet --set_data MOT20
python trackers/main.py --tracker sort --detector retinanet --set_data MOT17
python trackers/main.py --tracker sort --detector retinanet --set_data VisDrone2019-MOT-val

python trackers/main.py --tracker sort --detector mask_rcnn --set_data MOT20
python trackers/main.py --tracker sort --detector mask_rcnn --set_data MOT17
python trackers/main.py --tracker sort --detector mask_rcnn --set_data VisDrone2019-MOT-val

python trackers/main.py --tracker sort --detector efficientdet --set_data MOT20
python trackers/main.py --tracker sort --detector efficientdet --set_data MOT17
python trackers/main.py --tracker sort --detector efficientdet --set_data VisDrone2019-MOT-val



# deep_sort
source activate deep_sort

python trackers/main.py --tracker deep_sort --detector public --set_data MOT20
python trackers/main.py --tracker deep_sort --detector public --set_data MOT17
python trackers/main.py --tracker deep_sort --detector public --set_data VisDrone2019-MOT-val

python trackers/main.py --tracker deep_sort --detector yolo3 --set_data MOT20
python trackers/main.py --tracker deep_sort --detector yolo3 --set_data MOT17
python trackers/main.py --tracker deep_sort --detector yolo3 --set_data VisDrone2019-MOT-val

python trackers/main.py --tracker deep_sort --detector yolo4 --set_data MOT20
python trackers/main.py --tracker deep_sort --detector yolo4 --set_data MOT17
python trackers/main.py --tracker deep_sort --detector yolo4 --set_data VisDrone2019-MOT-val

python trackers/main.py --tracker deep_sort --detector faster_rcnn --set_data MOT20
python trackers/main.py --tracker deep_sort --detector faster_rcnn --set_data MOT17
python trackers/main.py --tracker deep_sort --detector faster_rcnn --set_data VisDrone2019-MOT-val

python trackers/main.py --tracker deep_sort --detector keypoint_rcnn --set_data MOT20
python trackers/main.py --tracker deep_sort --detector keypoint_rcnn --set_data MOT17
python trackers/main.py --tracker deep_sort --detector keypoint_rcnn --set_data VisDrone2019-MOT-val

python trackers/main.py --tracker deep_sort --detector retinanet --set_data MOT20
python trackers/main.py --tracker deep_sort --detector retinanet --set_data MOT17
python trackers/main.py --tracker deep_sort --detector retinanet --set_data VisDrone2019-MOT-val

python trackers/main.py --tracker deep_sort --detector mask_rcnn --set_data MOT20
python trackers/main.py --tracker deep_sort --detector mask_rcnn --set_data MOT17
python trackers/main.py --tracker deep_sort --detector mask_rcnn --set_data VisDrone2019-MOT-val

python trackers/main.py --tracker deep_sort --detector efficientdet --set_data MOT20
python trackers/main.py --tracker deep_sort --detector efficientdet --set_data MOT17
python trackers/main.py --tracker deep_sort --detector efficientdet --set_data VisDrone2019-MOT-val




# uma
source activate uma

python trackers/main.py --tracker uma --detector public --set_data MOT20
python trackers/main.py --tracker uma --detector public --set_data MOT17
python trackers/main.py --tracker uma --detector public --set_data VisDrone2019-MOT-val

python trackers/main.py --tracker uma --detector yolo3 --set_data MOT20
python trackers/main.py --tracker uma --detector yolo3 --set_data MOT17
python trackers/main.py --tracker uma --detector yolo3 --set_data VisDrone2019-MOT-val

python trackers/main.py --tracker uma --detector yolo4 --set_data MOT20
python trackers/main.py --tracker uma --detector yolo4 --set_data MOT17
python trackers/main.py --tracker uma --detector yolo4 --set_data VisDrone2019-MOT-val

python trackers/main.py --tracker uma --detector faster_rcnn --set_data MOT20
python trackers/main.py --tracker uma --detector faster_rcnn --set_data MOT17
python trackers/main.py --tracker uma --detector faster_rcnn --set_data VisDrone2019-MOT-val

python trackers/main.py --tracker uma --detector keypoint_rcnn --set_data MOT20
python trackers/main.py --tracker uma --detector keypoint_rcnn --set_data MOT17
python trackers/main.py --tracker uma --detector keypoint_rcnn --set_data VisDrone2019-MOT-val

python trackers/main.py --tracker uma --detector retinanet --set_data MOT20
python trackers/main.py --tracker uma --detector retinanet --set_data MOT17
python trackers/main.py --tracker uma --detector retinanet --set_data VisDrone2019-MOT-val

python trackers/main.py --tracker uma --detector mask_rcnn --set_data MOT20
python trackers/main.py --tracker uma --detector mask_rcnn --set_data MOT17
python trackers/main.py --tracker uma --detector mask_rcnn --set_data VisDrone2019-MOT-val

python trackers/main.py --tracker uma --detector efficientdet --set_data MOT20
python trackers/main.py --tracker uma --detector efficientdet --set_data MOT17
python trackers/main.py --tracker uma --detector efficientdet --set_data VisDrone2019-MOT-val
