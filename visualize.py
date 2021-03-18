
import os
import cv2
import argparse
import numpy as np

def parseInput(list_detectors):
    '''Parse input of the script.
    '''

    parser = argparse.ArgumentParser(description='Visualize tracks')

    # Select an able detector from list.
    parser.add_argument("--detector", help="Name of detector used.", required=True)
    parser.add_argument("--tracker", help="Name of tracker used.", required=True)
    parser.add_argument("--set", help="Name of set of data.", required=True)

    # Important arguments, but optional.

    # Other optional arguments


    return parser.parse_args()



class Visualize:

    def __init__(self, detector, tracker, set_data, data, verbose=0):

        data, self.extension = os.path.splitext(data)

        self.track_path = os.path.join('outputs/tracks', tracker, detector, set_data, data)

        self.imgs_path  = os.path.join('dataset/', set_data, data, 'img1')

        if verbose:

            print(self.track_path)
            print(self.imgs_path)


        self.readTracks()

        img  = self.readFrame(1)
        size = (img.shape[1], img.shape[0])

        self.videoWritter(size)




    @staticmethod
    def subsets(detector, tracker, set_data):

        path = os.path.join('outputs/tracks', tracker, detector, set_data)

        return os.listdir( path )



    def readTracks(self):

        tracks = np.loadtxt(self.track_path + '.txt', delimiter=',')

        unique = np.unique(tracks[:, 0])


        frame = {}

        for u in unique:

            a = np.where(tracks[:, 0] == u)

            frame[u] = tracks[a][:, 1:]


        self.frames = frame
        


    def readFrame(self, frame):
        '''
        '''

        file_name = os.path.join(self.imgs_path, '%.6d.jpg'%frame)

        return cv2.imread(file_name)


            

    @staticmethod
    def fromXtoWH(x1, y1, x2, y2):

        return x1, y1, x2-x1, y2-y1


    @staticmethod
    def fromWHtoX(x, y, w, h):

        return x, y, x+w, y+h


    def draw_bbox(self, img, thickness=2):

        color = (255, 255, 0)

        for bbox in self.frames[frame]:

            x1, y1, x2, y2 = Visualize.fromWHtoX(int(bbox[1]), int(bbox[2]), int(bbox[3]), int(bbox[4]))

            img = cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)


        return img


    def videoWritter(self, size):

        self.writter = cv2.VideoWriter('out.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 25.0, size)

    def writeVideo(self, img):

        self.writter.write(img)

        # out = cv2.VideoWriter('project.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
    def saveVideo(self):

        self.writter.release()


if __name__ == '__main__':

    detector = 'public'
    tracker  = 'sort'
    set_data = 'MOT20'

    data = Visualize.subsets(detector, tracker, set_data)

    # For each set

    visual_s = Visualize(detector, tracker, set_data, data[0])




    # PLOT ---------------------------------------------

    # x, y, w, h = 10, 10, 300, 300
    x1, y1, x2, y2 = 10, 10, 300, 300

    
    frame = 1

    while True:

        img = visual_s.readFrame(frame)

        if img is None: break

        visual_s.draw_bbox(img)
        visual_s.writeVideo(img)

        frame += 1


    visual_s.saveVideo()
