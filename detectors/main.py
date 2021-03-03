
import torch
import argparse
import importlib

from torchvision import transforms
from torch.utils.data import DataLoader

from DatasetLoader import DatasetLoader



def parseInput(list_detectors):
    '''Parse input of the script.
    '''

    parser = argparse.ArgumentParser(description='Faster-RCNN demo')

    # Select an able detector from list.
    parser.add_argument("--detector", help="Name of the detector to use.", choices=list_detectors, required=True)

    # Important arguments, but optional.
    parser.add_argument("--batch", help="Size of the batch.", default='5', type=int)
    parser.add_argument("--name", help="Add a postfix to 'faster-rcnn' detector name.")

    # Other optional arguments
    parser.add_argument("--path", help="Path were to find the data.", default='dataset')
    parser.add_argument("--set_data", help="Name of set of data.", default='MOT20')
    parser.add_argument("--detc_path", help="Name of folder used to store detections.", default='outputs/detections')
    parser.add_argument("-dvc", "--device", help="Device where is suposed to execute.", default='Auto')
    parser.add_argument("-v", "--verbose", help="Print debug information.", default=1, type=int)

    return parser.parse_args()



def readConfigFile():
    '''Read from the configuration file the available detectors.
    '''

    with open('detectors/detectors.conf', 'r') as fp:

        list_detectors = fp.read().split('\n')


    return list_detectors



def eval_sets(detector, loader, batch, device):
    '''Evaluate each set of data.
    '''

    t = transforms.Compose([transforms.ToTensor()])


    for setName in loader.listData():

        if args.verbose: print('Processing set', setName, '...')

        dataset = loader.loadData(setName, transform=t)
        dataloader = DataLoader(dataset, batch_size=args.batch, shuffle=False, num_workers=1)

        # evalSet(dataloader, model, args.batch, device)
        detector.eval_set(dataloader, loader, batch, device)

        # Save the results
        loader.save(setName)



def select_device(device):
    '''Select the device to use in next steps.
    '''

    if device == 'Auto':   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:                  device = args.device

    return device




if __name__ == '__main__':


    # Read configuration.
    list_detectors = readConfigFile()
    args = parseInput(list_detectors)


    # Select the device where it supposed to run.
    device = select_device(args.device)


    # Import the selected detector.
    module = importlib.import_module(args.detector)
    class_ = getattr(module, args.detector)
    detector = class_()                              # Create detector object


    # Set up the dataset.
    loader = DatasetLoader(path=args.path, set_data=args.set_data, savePath=args.detc_path, detectorName=detector.detector_name(args.name))

    
    # Run evaluation
    eval_sets(detector, loader, args.batch, device)


