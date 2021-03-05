
from abc import ABC, abstractmethod


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


