
import os
import argparse
import importlib
from Tracker import Tracker


def parseInput(list_detectors):
    '''Parse input of the script.
    '''

    parser = argparse.ArgumentParser(description='Trackers demo')

    # Select an able detector from list.
    parser.add_argument("--tracker", help="Name of the tracker to use.", choices=list_detectors, required=True)

    # Important arguments, but optional.
    parser.add_argument("--name", help="Add a postfix to tracker name.")
    parser.add_argument("--detector", help="Detector used to generate detections.", default='public')
    parser.add_argument("--batch", help="Size of the batch.", default='5', type=int)

    # Other optional arguments
    parser.add_argument("--path", help="Path were to find the data.", default='dataset')
    parser.add_argument("--set_data", help="Name of set of data.", default='MOT20')
    parser.add_argument("--detc_path", help="Path were to find detections.", default='outputs/detections')
    parser.add_argument("--track_path", help="Name of folder used to store the tracks.", default='outputs/tracks')
    parser.add_argument("--aux_path", help="Path were to store auxiliar information of detectors.", default='trackers/auxiliar')
    parser.add_argument("-dvc", "--device", help="Device where is suposed to execute.", default='Auto')
    parser.add_argument("-v", "--verbose", help="Print debug information.", default=1, type=int)

    return parser.parse_args()



def readConfigFile():
    '''Read from the configuration file the available detectors.
    '''

    with open('trackers/trackers.conf', 'r') as fp:

        list_detectors = fp.read().split('\n')


    return list_detectors



if __name__ == '__main__':


    # Read configuration.
    list_trackers= readConfigFile()
    args = parseInput(list_trackers)


    # Import the selected detector.
    module = importlib.import_module(args.tracker)
    class_ = getattr(module, args.tracker)
    tracker = class_(args.batch)                              # Create tracker object


    # Setup paths.
    img_path =        os.path.join(args.path, args.set_data)
    detections_path = os.path.join(args.detc_path, args.detector, args.set_data)
    track_path =      os.path.join(args.track_path, tracker.tracker_name(), args.detector, args.set_data)

    Tracker.create_path(track_path)
    Tracker.create_path(args.aux_path)


    # Load data
    tracker.load_data(img_path, detections_path, args.aux_path)


    # Run Evaluation
    tracker.calculate_tracks(args.aux_path, track_path)


