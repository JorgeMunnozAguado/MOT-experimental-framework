
import os
import numpy as np
from torchvision import datasets


IMAGES = 'img1'

class DatasetLoader:

    def __init__(self, detectorName, path='data/images', set_data='MOT20', savePath='outputs/detections'):
        '''Create a 
        '''

        self.path = os.path.join(path, set_data)
        self.setName = set_data

        self.savePath = savePath
        self.detectorName = detectorName

        self.results = []


        # Create path where to save detections
        if not os.path.exists(savePath):
            os.makedirs(savePath)

        if not os.path.exists(os.path.join(savePath, detectorName)):
            os.makedirs(os.path.join(savePath, detectorName))

        if not os.path.exists(os.path.join(savePath, detectorName, set_data)):
            os.makedirs(os.path.join(savePath, detectorName, set_data))



    def listData(self):

        list_d = os.listdir(self.path)
        list_d.sort()

        return list_d



    def loadData(self, setName, transform=None):

        path = os.path.join(self.path, setName)

        self.sets = datasets.ImageFolder(path, transform=transform)

        return self.sets


    def update(self, frame, boxes, scores, labels, label_permited=[1], preprocess=True):

        # Process bounding boxes
        if preprocess:
            boxes = DatasetLoader.processBoxes(boxes)

        # Save permited boxes in variable.
        for b, s, l in zip(boxes, scores, labels):

            if not l in label_permited: continue

            # Set format and save in variable
            # detection = [frame] + [-1] + b + [s.item()] + [-1, -1, -1]
            detection = [frame] + list(b) + [s]
            self.results.append(detection)





    def save(self, setName):

        file = os.path.join(self.savePath, self.detectorName, self.setName, setName)

        if not os.path.exists(file):
            os.makedirs(file)

        file = os.path.join(file, 'det')

        if not os.path.exists(file):
            os.makedirs(file)

        file = os.path.join(file, 'det.txt')

        np.savetxt(file, self.results, delimiter=',', fmt=['%d,-1', '%.0f', '%.0f', '%.0f', '%.0f', '%.4f,-1,-1,-1'])

        self.results = []



    @staticmethod
    def processBoxes(boxes):

        # x1, y1, x2, y2 --> x1, y1, width, height
        boxes = np.stack((boxes[:, 0],
                          boxes[:, 1],
                          boxes[:, 2] - boxes[:, 0],
                          boxes[:, 3] - boxes[:, 1]),
                          axis=1)

        return boxes
