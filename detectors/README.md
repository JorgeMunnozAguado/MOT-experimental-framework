
# Detectors

As we previously commented, detectors were implemented in the experimental framework using an encapsulated class. This method helps implement multiple detectors with various arquitectures. In this section you will be able to find information about the architecture used and some tips to include your own detector in the framework.


## Add new detector

Steps to add a new model to the framework.


1. Create a folder with the name of the detector: `<detector_name>`
1. Update `detectors.conf` adding the `<detector_name>` name in a new row.
1. Inside the created folder, add a file called `<detector_name>.py`. In this file create a new class named `<detector_name>`, with `Detector` as parent class, as shown in the example:
```python
from Detector import Detector

class faster_rcnn(Detector):

    def __init__(self, batch_size, trained=False):

    	super().__init__('faster_rcnn', batch_size)
    	pass

    def eval_set(self, dataset, loader, device='cuda', verbose=0):
    	pass
```
1. Code the required methods.
	1. `__init__`: initialize all required variables and hyperparameters.
	1. `eval_set`: perform inference over a sequene stored in `dataset`. Save the results in `loader`, for each frame a list of `boxes`, `scores` and `labels`.
1. In the created folder, create a `__init__.py` file similar to the next example.
```python
from .<detector_name> import <detector_name>
```

> If you need a new conda environment add a `setup.sh` file.
> Inside the created folder you can create all the subfolders and python files you want to.
> We recommend to see a working model before include a new one. Check for example `faster_rcnn`.

## Finetune or Train models
To train a model with this you might need to modify some methods.

At least you need to code the next method in `<detector_name>/<detector_name>.py` file.
```python
def train_model(self, dataloader, epochs, device='cuda'):
	pass
```

Other functions that you can find coded in `Detector.py` are designed to work with *Pytorch*. If you want to include a new environment override the functions in the `<detector_name>/<detector_name>.py` file.
```python
def train_epoch(self, train_loader, criterion, optimizer, device):
	pass

def train(self, train_loader, valid_loader, criterion, optimizer, epochs, first_epoch=0, validation=True, device='cuda'):
	pass

def save_checkpoint(optimizer, model, epoch, filename):
	pass

def load_checkpoint(optimizer, model, filename):
	pass
```


## Some tips

Some tips for adding your own detector to the framework:

1. Use the encapsulated class, will help you to execute the detector.
1. Include a flag to add a postfix to the name of the detector. This will be useful to test the same detector with multiple configurations.
1. Create a `setup.sh` file inside detectors folder. In that script create the enviroment and download the models.
1. Create a configuration file to setup some basic configurations of the detector.



## Scores

MOT17 scores:

| Detector | Subset name | mAP |
|----------|-------------|-----|
| gt | AVERAGE | 100.0 | 
| faster_rcnn | AVERAGE | 60.6 | 
| mask_rcnn | AVERAGE | 60.6 | 
| retinanet | AVERAGE | 59.2 | 
| keypoint_rcnn | AVERAGE | 58.0 | 
| yolo4 | AVERAGE | 53.2 | 
| efficientdet-d7x | AVERAGE | 47.7 | 
| yolo3 | AVERAGE | 46.4 | 
