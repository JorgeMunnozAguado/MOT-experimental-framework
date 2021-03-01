
import os
import numpy as np
from torchvision import datasets


IMAGES = 'img1'
TEST_NAME  = 'test'
TRAIN_NAME = 'train'


class DatasetLoader:

    def __init__(self, path='data/images', mot='MOT20', savePath='data/detections', detectorName='faster-rcnn'):

        self.path = os.path.join(path, mot)
        self.setName = mot

        self.savePath = savePath
        self.detectorName = detectorName

        if not os.path.exists(savePath):
            os.makedirs(savePath)

        if not os.path.exists(os.path.join(savePath, detectorName)):
            os.makedirs(os.path.join(savePath, detectorName))

        if not os.path.exists(os.path.join(savePath, detectorName, mot)):
            os.makedirs(os.path.join(savePath, detectorName, mot))



    def listData(self):

        return os.listdir(self.path)



    def loadData(self, setName, transform=None):

        path = os.path.join(self.path, setName)

        self.sets = datasets.ImageFolder(path, transform=transform)

        return self.sets



    def saveResult(self, setName, frame, boxes, scores, labels, label_permited=[1]):

        file = os.path.join(self.savePath, self.detectorName, self.setName, setName)

        if not os.path.exists(file):
            os.makedirs(file)

        file = os.path.join(file, 'det')

        if not os.path.exists(file):
            os.makedirs(file)

        file = os.path.join(file, 'det.txt')

        
        boxes = DatasetLoader.processBoxes(boxes)

        with open(file, 'a') as fp:

            for b, s, l in zip(boxes, scores, labels):

                if not l in label_permited: continue

                # print(b.tolist())
                b = [str(int(a.item())) for a in b]
                # print(','.join(b.tolist()))

                fp.write(str(frame) + ',-1' + ',' + ','.join(b) + ',' + str(s.item()) + ',-1' + ',-1' + ',-1\n')



    @staticmethod
    def processBoxes(boxes):

        boxes = boxes.detach().numpy()
        # x1, y1, x2, y2 --> x1, y1, width, height
        boxes = np.stack((boxes[:, 0],
                          boxes[:, 1],
                          boxes[:, 2] - boxes[:, 0],
                          boxes[:, 3] - boxes[:, 1]),
                          axis=1)

        return boxes
