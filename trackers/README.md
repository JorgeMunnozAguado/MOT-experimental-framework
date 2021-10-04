
# Trackers

As we previously commented, trackers were implemented in the experimental framework using an encapsulated class. This method helps implement multiple trackers with various arquitectures. In this section you will be able to find information about the architecture used and some tips to include your own tracker in the framework.


## Add new tracker

Steps to add a new model to the framework.


1. Create a folder with the name of the tracker: `<tracker_name>`
1. Update `trackers.conf` adding the `<tracker_name>` name in a new row.
1. Inside the created folder, add a file called `<tracker_name>.py`. In this file create a new class named `<tracker_name>`, with `Tracker_abs` as parent class, as shown in the example:
```python
from Tracker import Tracker_abs

class sort(Tracker_abs):

    def __init__(self, batch_size):

        super().__init__('sort', batch_size)
    	pass

    def load_data(self, img_path, detections_path, aux_path):
    	pass

    def calculate_tracks(self, img_path, aux_path, track_path, verbose=0):
    	pass
```
1. Code the required methods.
	1. `__init__`: initialize all required variables and hyperparameters.
	1. `load_data`: used if the tracker need to perform a preprocesing step of the frames stored in `img_path`, with the results from detection step `detections_path`. `aux_path` is where to store intermediate data in the next step.
	1. `calculate_tracks`: perform inference over the frames stored in `img_path`, with the info from detection step, and store the results in `track_path`.
1. In the created folder, create a `__init__.py` file similar to the next example.
```python
from .<tracker_name> import <tracker_name>
```

> If you need a new conda environment add a `setup.sh` file.
> Inside the created folder you can create all the subfolders and python files you want to.
> We recommend to see a working model before include a new one. Check for example `sort`.

## Finetune or Train models
In tracking step we do not include a training step.


## Some tips

Some tips for adding your own tracker to the framework:

1. Use the encapsulated class, will help you to execute the tracker.
1. Include a flag to add a postfix to the name of the tracker. This will be useful to test the same tracker with multiple configurations.
1. Create a `setup.sh` file inside trackers folder. In that script create the enviroment and download the models.
1. Create a configuration file to setup some basic configurations of the tracker.
