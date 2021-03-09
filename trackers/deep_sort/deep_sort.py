
import os
import glob
import time

from Tracker import Tracker

from deep_sort.tools.generate_detections import *
from deep_sort.deep_sort_app import *


class deep_sort(Tracker):

    def __init__(self, batch_size):

        super().__init__('deep_sort', batch_size)

        self.model = 'trackers/deep_sort/models/mars-small128.pb'



    def load_data(self, img_path, detections_path, aux_path):

        encoder = create_box_encoder(self.model, batch_size=self.batch_size)
        generate_detections(encoder, img_path, aux_path, detections_path)


    def calculate_tracks(self, aux_path, track_path, verbose=0):
        
        run(args.sequence_dir, args.detection_file, args.output_file,
        args.min_confidence, args.nms_max_overlap, args.min_detection_height,
        args.max_cosine_distance, args.nn_budget, args.display)



        