
from abc import ABC, abstractmethod


class Detector(ABC):

    def __init__(self, name):
        
        self.name = name


    def detector_name(self, extension=None):

        if extension: return self.name + '-' + extension

        return self.name


    @abstractmethod
    def eval_set(self, dataloader, loader, batch_size, device, verbose=0):
        pass


