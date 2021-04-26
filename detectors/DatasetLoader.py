
import os
import numpy as np

from torchvision import datasets
from torchvision import transforms


IMAGES = 'img1'
TRANSLATE_str_int = {'person':1, 'bicycle':4, 'car':3, 'motorbike':5, 'bus':3, 'truck':3}
TRANSLATE_int = {1:1, 2:4, 3:3, 4:5, 6:3, 7:3, 8:3}

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



    def loadData(self, setName, resize=400, transform=None):


        # self.resize = None

        # if transform is None:

        #     transform = transforms.Compose([
        #         transforms.Resize(resize),
        #         transforms.ToTensor(),
        #     ])

        #     self.resize = resize


        path = os.path.join(self.path, setName)

        self.sets = datasets.ImageFolder(path, transform=transform)

        return self.sets


    def framesData(self, setName):

        path = os.path.join(self.path, setName, 'img1')

        return len( os.listdir(path) )




    def update(self, frame, boxes, scores, labels, label_permited=[1], preprocess=True):

        # Process bounding boxes
        if preprocess:
            boxes = DatasetLoader.processBoxes(boxes)

        # Save permited boxes in variable.
        for b, s, l in zip(boxes, scores, labels):

            # Check if label is permited
            if (label_permited is not None) and (not l in label_permited): continue

            # Translate label
            if type(l) == str:      l = TRANSLATE_str_int[l]
            elif type(l) == int:    l = TRANSLATE_int[l]

            if (b[2] == 0) or (b[3] == 0): print('continue'); continue

            # Set format and save in variable
            # detection = [frame] + [-1] + b + [s.item()] + [-1, -1, -1]
            detection = [frame] + list(b) + [s] + [l]
            self.results.append(detection)



    def detectionsData(self):

        return len( self.results )



    def save(self, setName):

        file = os.path.join(self.savePath, self.detectorName, self.setName, setName)

        if not os.path.exists(file):
            os.makedirs(file)

        file = os.path.join(file, 'det')

        if not os.path.exists(file):
            os.makedirs(file)

        file = os.path.join(file, 'det.txt')

        np.savetxt(file, self.results, delimiter=',', fmt=['%d,-1', '%.0f', '%.0f', '%.0f', '%.0f', '%.4f', '%d,-1,-1'])

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
