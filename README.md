
# Experimental Framework for MOT

This repository is an experimental framework for Multi Object Tracking analysis. Here you can find multiple detectors, trackers and evaluation metrics. Also with the following documentation you might implement your own algorithm in the framework.

The repository is divided in folders:

- **dataset** : Datasets used for Multi Object Tracking are saved in this folder.
- _outputs_ : Outputs of detectors and trackers will be save in this folder.
- **detectors** : There are multiple detector algorithms.
- **trackers** : Tracking algorithms are stored here.
- **evaluation** : Contains an evaluation script with multiple metrics.

In the following sections we will explore how are organized this folders. And how to run experiments.


## Detection models

You can find detection models in the folder `detectors`. Each model has its own folder. If you want some tips of how implement a new detector read the `README.md` file in that folder.

| Detector Name | Code | Publication Year | Publication |
| ------------- |:----:|:----------------:|:-----------:|
| Faster-RCNN   |      |                  |             |


## Tracking models

| Tracker Name  | Code | Publication Year | Publication |
| ------------- |:----:|:----------------:|:-----------:|
| Sort          |      |                  |             |


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

Folder must contain the

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

To install the enviroment is needed to run the following command:
```
```

To perform the evaluation run:
```
python3 evaluation/Evaluate.py
```








# Packages (dev)

conda install -y -c pytorch pytorch torchvision