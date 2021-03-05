
# Experimental Framework for MOT

This repository is an experimental framework for Multi Object Tracking analysis. Here you can find multiple detectors, trackers and evaluation metrics. Also with the following documentation you might implement your own algorithm in the framework.

The repository is divided in folders:

- **dataset** : Datasets used for Multi Object Tracking are saved in this folder.
- _outputs_ : Outputs of detectors and trackers will be save in this folder.
- **detectors** : Folder were there are the detection models.
- **trackers** : Tracking algorithms are stored here.
- **evaluation** : Contains an evaluation script with multiple metrics.

In the following sections we will explore how are organized this folders and how to run experiments.

To setup the environment run `setup.sh`. It is recomended to configure the flags in order to get the best performance.

## Detection models

You can find detection models in the folder `detectors`. Each model has its own folder. If you want some tips of how implement a new detector read the `README.md` file in that folder.

| Detector Name | Code                                | Publication Year | Publication                      |
| ------------- |:-----------------------------------:|:----------------:|:--------------------------------:|
| Faster-RCNN   | https://pytorch.org/                | 2016             | https://arxiv.org/abs/1506.01497 |
| Yolo V4       | https://github.com/AlexeyAB/darknet | 2020             | https://arxiv.org/abs/2004.10934 |
| Yolo V3       | https://github.com/AlexeyAB/darknet | 2018             | https://arxiv.org/abs/1804.02767 |

These detectors are encapsulated in one script file call `detectors/main.py`. To run the experiments it is necessary to select the detector you want to test. The script also have some parameters to explore. See more running `python detectors/main.py -h`.

Example of running Faster-RCNN.
```
python detectors/main.py --detector faster_rcnn
```


## Tracking models

| Tracker Name  | Code                                | Publication Year | Publication                      |
| ------------- |:-----------------------------------:|:----------------:|:--------------------------------:|
| Sort          | https://github.com/abewley/sort     | 2017             | https://arxiv.org/abs/1602.00763 |


## Dataset

The base dataset is compose of [Mot Challenge 2020 dataset](https://motchallenge.net/data/MOT20/). To download it just run the following command:

```
./data/download_data.sh
```

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
    └── sort
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








# Packages (dev)

conda install -y -c pytorch pytorch torchvision
