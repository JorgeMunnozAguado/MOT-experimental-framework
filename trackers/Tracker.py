
import os

from abc import ABC, abstractmethod


class Tracker(ABC):

    def __init__(self, name, batch_size):
        
        self.name = name
        self.batch_size = batch_size


    def tracker_name(self, extension=None):

        if extension: return self.name + '-' + extension

        return self.name


    @abstractmethod
    def load_data(self, img_path, detections_path, aux_path):
        pass


    @abstractmethod
    def calculate_tracks(self, aux_path, track_path, verbose=0):
        pass



##################################################################
##################################################################
### STATIC METHODS
###

    @staticmethod
    def create_path(path):
        '''Check if exist path and all subpaths.
        If they do not exist, the method will create them.
        '''

        dirs = path.split('/')

        # If root
        if path[0] == '/':

            dirs = dirs[1:]
            dirs[0] = '/' + dirs[0]


        path_a = ''

        # Check if exist. If not create it.
        for folder in dirs:

            path_a = os.path.join(path_a, folder)

            if folder == '..': continue

            if not os.path.exists(path_a):
                os.makedirs(path_a)



    @staticmethod
    def check_path(actual, new, clean=False):
        '''Check if path exist, if not create it.
        clean = True: will remove the content in the folder.
        '''

        path = os.path.join(actual, new)

        if not os.path.exists(path):
            os.makedirs(path)

        elif os.path.exists(path) and clean:
            shutil.rmtree(path)
            os.makedirs(path)


        return path




