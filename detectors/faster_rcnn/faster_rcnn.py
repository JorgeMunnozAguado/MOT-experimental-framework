
import torchvision
import torch.optim as optim

from torch import nn
from torch.utils.data import DataLoader

from Detector import Detector


class faster_rcnn(Detector):

    def __init__(self, batch_size, trained=False):

        super().__init__('faster_rcnn', batch_size)


        # Load model (pretrained)
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        #self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, box_score_thresh=0, rpn_score_thresh=0, box_nms_thresh=0)
        #self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, box_score_thresh=0.5, rpn_score_thresh=0.5)
        #self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, box_score_thresh=0.9, rpn_score_thresh=0.9, box_nms_thresh=0.9)

        #self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, box_score_thresh=0.09, box_nms_thresh=0.35)

        if trained: self.model = Detector.load_checkpoint(None, self.model, 'detectors/faster_rcnn/checkpoints/faster_rcnn.pkl')

        self.model.eval()

        # Classes:  https://pytorch.org/vision/stable/models.html#object-detection-instance-segmentation-and-person-keypoint-detection
        self.label_permited = [1, 2, 3, 4, 6, 7, 8]



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


    def train_model(self, dataloader, epochs, device='cpu'):


        # print(self.model.backbone.parameters())

        for param in self.model.backbone.parameters():

            param.requires_grad = False

        # self.model.backbone
        # self.model.rpn
        # self.model.roi_heads

        criterion = nn.BCELoss()
        #optimizer = optim.Adam(self.model.parameters(), lr=0.003, weight_decay=0.0018)
        #optimizer = optim.Adam(self.model.parameters(), lr=0.025, weight_decay=0.0018)
        #optimizer = optim.Adam(self.model.parameters(), lr=0.0055, weight_decay=0.0018)
        #optimizer = optim.Adam(self.model.parameters(), lr=0.0045, weight_decay=0.0025)

        optimizer = optim.SGD(self.model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)


        self.train(dataloader, None, criterion, optimizer, epochs, validation=False, device=device)
