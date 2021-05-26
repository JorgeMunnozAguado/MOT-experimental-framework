
# Experimental Framework for MOT

This repository is an experimental framework for Multi Object Tracking analysis. Here you can find multiple detectors, trackers and evaluation metrics. Also with the following documentation you might implement your own model in the framework.

The repository is divided in folders:

- **tools** : Useful tools for visualize the gt or outputs.
- **dataset** : Datasets used for Multi Object Tracking is store in this folder.
- _outputs_ : Outputs of detectors and trackers will be save in this folder.
- **detectors** : Folder with detection models.
- **evaluation** : Contains an evaluation script with multiple metrics.
- **trackers** : Tracking algorithms are stored here.

In the following sections we will explore how are organized this folders and how to run experiments.

To setup the environment run `bash setup.sh`. It is recomended to configure the flags in order to get the best performance running the experiments. This script will take 1 hour or more to run.

## Installation & Execution

Download datasets and create enviroments:
```
bash setup.sh
```

Run detection step:
```
python detectors/main.py --detector <detector_name>
```

Run tracking step:
```
python trackers/main.py --detector <tracker_name>
```

Evaluating results:
```
python evaluation/Evaluate.py
```

## Detection models

You can find detection models in the folder `detectors`. Each model has its own folder. If you want some tips of how implement a new detector read the `README.md` file in that folder.

The implemented detectors are listed below:

| Detector Name | Code                                | Publication Year | Publication                      |
| ------------- |:-----------------------------------:|:----------------:|:--------------------------------:|
| Faster-RCNN   | [https://pytorch.org/](https://pytorch.org/vision/0.8/models.html#faster-r-cnn) | 2016             | https://arxiv.org/abs/1506.01497 |
| Yolo V4       | https://github.com/AlexeyAB/darknet | 2020             | https://arxiv.org/abs/2004.10934 |
| Yolo V3       | https://github.com/AlexeyAB/darknet | 2018             | https://arxiv.org/abs/1804.02767 |
| RetinaNet     | [https://pytorch.org/](https://pytorch.org/vision/0.8/models.html#retinanet) | 2018             | https://arxiv.org/abs/1708.02002 |
| Mask R-CNN    | [https://pytorch.org/](https://pytorch.org/vision/0.8/models.html#mask-r-cnn) | 2018             | https://arxiv.org/abs/1703.06870 |
| Keypoint R-CNN | [https://pytorch.org/](https://pytorch.org/vision/0.8/models.html#keypoint-r-cnn) |              |  |
| EfficientDet  | https://github.com/google/automl    | 2020             | https://arxiv.org/abs/1911.09070 |


Example of running detection step.
```
python detectors/main.py --detector <detector_name>
```
If needed help, run the script with `-h` flag.



## Tracking models

You can find detection models in the folder `trackers`. Each model has its own folder. If you want some tips of how implement a new detector read the `README.md` file in that folder.

The implemented trackers are listed below:

| Tracker Name  | Code                                | Publication Year | Publication                      | Journal         | Type   |
| ------------- |:-----------------------------------:|:----------------:|:--------------------------------:|:---------------:| ------ |
| Sort          | https://github.com/abewley/sort     | 2017             | https://arxiv.org/abs/1602.00763 | IEEE Conference | Online |
| Deep Sort     | https://github.com/nwojke/deep_sort | 2017             | https://arxiv.org/abs/1703.07402 | IEEE Conference | Online |
| SST           | https://github.com/shijieS/SST      | 2018             | https://arxiv.org/abs/1703.07402 | Arxiv           | Online |
| UMA           | https://github.com/yinjunbo/UMA-MOT | 2020             | https://arxiv.org/abs/2003.11291 | CVPR            | Online |


Example of running tracking step.
```
python trackers/main.py --detector <tracker_name>
```
If needed help, run the script with `-h` flag.


## Dataset

The base dataset is compose of the following datasets. This sets will be automaticly downloaded running the `setup.sh` script.

| Dataset Name | URL | Download |
---------------|-----|----------|
| MOT20 | [MOT Challenge](https://motchallenge.net/data/MOT20/) | https://motchallenge.net/data/MOT20.zip |
| MOT17 | [MOT Challenge](https://motchallenge.net/data/MOT17/) | https://motchallenge.net/data/MOT17.zip |
| VisDrone | [VisDrone](http://aiskyeye.com/) | (Create an account) |


The folder structure used to store the data is based on MOT Challenge structure. In this repository are two folders where to store data: `dataset` where is loaded the raw data and `outputs` where are stored the outputs of the detector and tracking algorithms. The following sub-sections explore how the data is strutured in those folders.


### Dataset distribution

In the `dataset` folder you can find the GT and the images of each set of data.

The directory contains folders for each set of data. Inside them the are folders for each sequence of video. For each sequence there is a `gt/gt.txt` file and a folder `img1` with a list images associated to each frame. Below you can find an example of the tree structure.

```
dataset/
├── MOT20
│   ├── MOT20-01
│   │   ├── gt
│   │   │   └── gt.txt
│   │   ├── img1
│   │   │   └── <frame_id>.jpg
│
└── MOT17
    .
    .
```

### Outputs distribution

In the `output` directory you can find the results of the detectors and tracker models.

```
outputs/
├── detections
│   ├── public
│   │   └── MOT20
│   │       └── <sequence name>
│   │           └── det
│   │               └── det.txt
│   └── <detector>
│       └── MOT20
│           └── ...
│
└── tracks
    └── <tracker>
        ├── public
        │   └── MOT20
        │       └── <sequence name>.txt
        └── <detector>
            └── MOT20
                └── ...
```


## Evaluation

Evaluation module was based on [mot-metrics](https://github.com/cheind/py-motmetrics) repository. We have perform some changes in order to perform evaluation over the predictions perform with all the possible combinatios of detectors and trackers.

We also add HOTA metric to prove performance.


To perform the evaluation run:
```
python3 evaluation/Evaluate.py
```

