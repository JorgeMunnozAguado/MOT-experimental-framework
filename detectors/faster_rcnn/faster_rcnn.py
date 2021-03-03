
import torchvision

from Detector import Detector


class faster_rcnn(Detector):

    def __init__(self):

        super().__init__('faster_rcnn')


        # Load model (pretrained)
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        self.model.eval()



    def eval_set(self, dataloader, loader, batch_size, device, verbose=0):
        '''Run evaluation over loaded data.
        '''

        for i_batch, (images, y) in enumerate(dataloader):

            images = images.to(device)
            output = self.model(images)

            if verbose: print('Frame:', i_batch * batch_size)

            for i, frame in enumerate(output):

                i_frame = (i_batch * batch_size) + i + 1

                loader.update(i_frame, frame['boxes'], frame['scores'], frame['labels'])

