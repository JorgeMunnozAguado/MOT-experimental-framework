
# Experimental Framework for MOT

The repository is divided in folders:

- data : Folder where you can find the dataset.
- detection : Object detection algorithms will be saved in this folder.
- tracking : Tracking algorithms will be save and store in this folder.
- evaluation : The folder contains an evaluation program with different metrics.



## Detection models


## Tracking models


## Data

The dataset is compose of [Mot Challenge 2020 dataset](https://motchallenge.net/data/MOT20/) and .........

To dowload the dataset execute next command:

```
./data/download_data.sh
```

### New data

If want to add new data it must follow the distibution specify in the previous section.



## Evaluation

Evaluation module was based on [mot-metrics]https://github.com/cheind/py-motmetrics) repository. We have perform some changes in order to perform evaluation over the predictions perform with all the possible combinatios of detectors and trackers.

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