
import argparse
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

from ImagesLoader import DatasetLoader

device = 'cpu'


parser = argparse.ArgumentParser(description='Faster-RCNN demo')

parser.add_argument("-dvc", "--device", help="Device where is suposed to execute.", default='cpu')
parser.add_argument("--path", help="Path were to find the data.", default='data/images')
parser.add_argument("--set_data", help="Name of set of data.", default='MOT20')
parser.add_argument("--detc_path", help="Name of folder used to store detections.", default='data/detections')
parser.add_argument("--batch", help="Size of the batch.", default='10', type=int)

args = parser.parse_args()


def evalSet(dataloader, model, batch_size):


    for i_batch, (images, y) in enumerate(dataloader):

        images = images.to(device)

        output = model(images)

        for i, frame in enumerate(output):

            i_frame = (i_batch * batch_size) + i + 1

            print(i_frame, i_batch, i)

            loader.saveResult(setName, i_frame, frame['boxes'], frame['scores'], frame['labels'])





if __name__ == '__main__':

    ######################
    # Load data
    ######################

    loader = DatasetLoader(path=args.path, mot=args.set_data, savePath=args.detc_path, detectorName='faster-rcnn')

    t = transforms.Compose([transforms.ToTensor()])


    ######################
    # Load model (pretrained)
    ######################

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    model.eval()



    ######################
    # Predict
    ######################

    for setName in loader.listData():

        print('Processing set', setName, '...')

        dataset = loader.loadData(setName, transform=t)
        dataloader = DataLoader(dataset, batch_size=args.batch, shuffle=False, num_workers=1)

        evalSet(dataloader, model, args.batch)

