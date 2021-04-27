
import torch

from abc import ABC, abstractmethod

from torch.autograd import Variable


class Detector(ABC):

    def __init__(self, name, batch_size):
        
        self.name = name
        self.batch_size = batch_size


    def detector_name(self, extension=None):

        if extension: return self.name + '-' + extension

        return self.name


    @abstractmethod
    def eval_set(self, dataloader, loader, device, verbose=0):
        pass



    def train_model(self, data):
        pass




    def train_epoch(self, train_loader, criterion, optimizer, device):

        # put model in train mode
        self.model.train()

        # keep track of the training losses during the epoch
        train_losses = []

        for inputs, labels in train_loader:

            # print('VUELTA')
            labels = [{k: v.to(device) for k, v in labels.items()}]
            inputs = list([inputs.to(device)])

            # clear previous gradient computation
            optimizer.zero_grad()

            # forward propagation
            loss_dict = self.model(inputs, labels)

            losses = sum(loss for loss in loss_dict.values())

            losses.backward()
            optimizer.step()

            # update the array of batch losses
            train_losses.append(losses.item())

        # calculate average training loss of the epch and return
        return sum(train_losses) / len(train_losses)



    def train(self, train_loader, valid_loader, criterion, optimizer, epochs, first_epoch=0, validation=True, device='cuda'):
        '''Train the model for a specific number of epochs.
        '''
        train_losses, valid_losses = [],  []

        for epoch in range(first_epoch, first_epoch + epochs):

            # training phase
            train_loss = self.train_epoch(train_loader, criterion, optimizer, device)

            if validation:
                # validation phase
                valid_loss, valid_acc = validate(self.model, valid_loader, criterion, device)

                print(f'[{epoch:03d}] train loss: {train_loss:04f}  '
                      f'val loss: {valid_loss:04f}  '
                      f'val acc: {valid_acc*100:.4f}%')

                valid_losses.append(valid_loss)

            else:

                print(f'[{epoch:03d}] train loss: {train_loss:04f}  ')


            train_losses.append(train_loss)

        # Save a checkpoint
        checkpoint_filename = f'detectors/faster_rcnn/checkpoints/faster_rcnn-{epoch:03d}.pkl'
        Detector.save_checkpoint(optimizer, self.model, epoch, checkpoint_filename)

        return train_losses, valid_losses



    @staticmethod
    def save_checkpoint(optimizer, model, epoch, filename):

        checkpoint_dict = {
            'optimizer': optimizer.state_dict(),
            'model': model.state_dict(),
            'epoch': epoch
        }

        torch.save(checkpoint_dict, filename)