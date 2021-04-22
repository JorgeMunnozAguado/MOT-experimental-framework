
# Detectors

As we previously commented, detectors were implemented in the experimental framework using an encapsulated class. This method helps implement multiple detectors with various arquitectures. In this section you will be able to find information about the architecture used and some tips to include your own detector in the framework.


## Architecture



## Some tips

Some tips for adding your own detector to the framework:

1. Use the encapsulated class, will help you to execute the detector.
1. Include a flag to add a postfix to the name of the detector. This will be useful to test the same detector with multiple configurations.
1. Create a `setup.sh` file inside detectors folder. In that script create the enviroment and download the models.
1. Create a configuration file to setup some basic configurations of the detector.



## Scores

MOT17 scores:

| Detector | Subset name | mAP |
--------------------------------
| gt | AVERAGE | 100.0 | 
| faster_rcnn | AVERAGE | 60.60716427905071 | 
| mask_rcnn | AVERAGE | 60.62512214547893 | 
| retinanet | AVERAGE | 59.2690341231101 | 
| keypoint_rcnn | AVERAGE | 58.05662412102981 | 
| yolo4 | AVERAGE | 53.2806016853395 | 
| efficientdet-d7x | AVERAGE | 47.70367103320473 | 
| yolo3 | AVERAGE | 46.44393456607872 | 