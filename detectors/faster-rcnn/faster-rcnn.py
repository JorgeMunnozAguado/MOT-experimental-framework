
import torch
import argparse
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

from ImagesLoader import DatasetLoader

device = 'cpu'


def parseInput():
    '''Parse input of the script.
    '''

    parser = argparse.ArgumentParser(description='Faster-RCNN demo')

    parser.add_argument("-dvc", "--device", help="Device where is suposed to execute.", default='Auto')
    parser.add_argument("--path", help="Path were to find the data.", default='dataset')
    parser.add_argument("--set_data", help="Name of set of data.", default='MOT20')
    parser.add_argument("--detc_path", help="Name of folder used to store detections.", default='outputs/detections')
    parser.add_argument("--batch", help="Size of the batch.", default='10', type=int)
    parser.add_argument("--name", help="Add a postfix to 'faster-rcnn' detector name.")
    parser.add_argument("-v", "--verbose", help="Print debug information.", default=1, type=int)

    return parser.parse_args()





def evalSet(dataloader, model, batch_size, device):
    '''Run evaluation over loaded data.
    '''

    for i_batch, (images, y) in enumerate(dataloader):

        images = images.to(device)

        output = model(images)

        for i, frame in enumerate(output):

            i_frame = (i_batch * batch_size) + i + 1

            loader.saveResult(setName, i_frame, frame['boxes'], frame['scores'], frame['labels'])





if __name__ == '__main__':

    # Parse input
    args = parseInput()


    # Select device where is supose to run
    if args.device == 'Auto':    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:                        device = args.device


    # Name of detector
    detectorName = 'faster-rcnn'

    if args.name:   detectorName = detectorName + '-' + args.name


    ######################
    # Load data
    loader = DatasetLoader(path=args.path, mot=args.set_data, savePath=args.detc_path, detectorName=detectorName)
    t = transforms.Compose([transforms.ToTensor()])


    ######################
    # Load model (pretrained)
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()



    ######################
    # Predict
    for setName in loader.listData():

        if args.verbose: print('Processing set', setName, '...')

        dataset = loader.loadData(setName, transform=t)
        dataloader = DataLoader(dataset, batch_size=args.batch, shuffle=False, num_workers=1)

        # evalSet(dataloader, model, args.batch, device)

