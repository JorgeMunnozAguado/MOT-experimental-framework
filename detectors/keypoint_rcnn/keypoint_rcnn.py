
import torchvision
from torch.utils.data import DataLoader

from Detector import Detector


class keypoint_rcnn(Detector):

    def __init__(self, batch_size):

        super().__init__('keypoint_rcnn', batch_size)


        # Load model (pretrained)
        self.model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True)
        self.model.eval()

        self.label_permited = [1]



    def eval_set(self, dataset, loader, device, verbose=0):
        '''Run evaluation over loaded data.
        '''

        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=1)
        self.model.to(device)


        for i_batch, (images, y) in enumerate(dataloader):

            images = images.to(device)
            output = self.model(images)

            if verbose: print('Frame:', i_batch * self.batch_size)

            for i, frame in enumerate(output):

                i_frame = (i_batch * self.batch_size) + i + 1

                loader.update(i_frame, frame['boxes'].cpu().detach().numpy(), frame['scores'].cpu().detach().numpy(), frame['labels'].cpu(), label_permited=self.label_permited)
