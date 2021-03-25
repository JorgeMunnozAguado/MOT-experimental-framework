
# Detectors

As we previously commented, detectors were implemented in the experimental framework using an encapsulated class. This method helps implement multiple detectors with various arquitectures. In this section you will be able to find information about the architecture used and some tips to include your own detector in the framework.


## Architecture



## Some tips

Some tips for adding your own detector to the framework:

1. Use the encapsulated class, will help you to execute the detector.
1. Include a flag to add a postfix to the name of the detector. This will be useful to test the same detector with multiple configurations.
1. Create a `setup.sh` file inside detectors folder. In that script create the enviroment and download the models.
1. Create a configuration file to setup some basic configurations of the detector.
