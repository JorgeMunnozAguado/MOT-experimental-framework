import argparse
import os
import glob
import random
import darknet
import time
import cv2
import numpy as np
import darknet

from Detector import Detector


import os

PATH = 'detectors/yolo4/'


class yolo4(Detector):

    def __init__(self):

        super().__init__('yolo4')

        CONFIG_FILE = PATH + 'cfg/yolov4.cfg'
        DATA_FILE   = PATH + 'cfg/coco.data'
        WEIGHTS     = PATH + 'yolov4.weights'

        # Load model (pretrained)
        network, class_names, _ = darknet.load_network(
            CONFIG_FILE,
            DATA_FILE,
            WEIGHTS#,
            #batch_size=args.batch_size
        )

        self.model = network
        self.class_names = class_names



    def eval_set(self, dataloader, loader, batch_size, device, verbose=0):
        '''Run evaluation over loaded data.
        '''

        for i_batch, (images, y) in enumerate(dataloader):

            print(images.shape)

        #     images = images.to(device)
        #     output = self.model(images)

        #     if verbose: print('Frame:', i_batch * batch_size)

        #     for i, frame in enumerate(output):

        #         i_frame = (i_batch * batch_size) + i + 1

        #         loader.update(i_frame, frame['boxes'], frame['scores'], frame['labels'])







def image_detection(image_path, network, class_names, thresh):
    # Darknet doesn't accept numpy images.
    # Create one with image we reuse for each detect



    # width = darknet.network_width(network)
    # height = darknet.network_height(network)
    # darknet_image = darknet.make_image(width, height, 3)

    # image = cv2.imread(image_path)
    # image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image_resized = cv2.resize(image_rgb, (width, height),
    #                            interpolation=cv2.INTER_LINEAR)
    # darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())


    detections = darknet.detect_image(network, class_names, darknet_image, thresh=thresh)
    return detections






# detections = image_detection(image_name, network, class_names, args.thresh)
# print(detections)
