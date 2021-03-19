
import os
import cv2
import argparse
import numpy as np

from matplotlib.colors import to_rgb


# Need more colors?:
# https://matplotlib.org/2.0.2/examples/color/named_colors.html
COLORS = ['b', 'green', 'r', 'c', 'm', 'y', 'fuchsia', 'lime']




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
        self.processTraces()
        self.listColors()

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
        

    def processTraces(self):

        ids = {}

        for frame in self.frames.values():

            for obj in frame:

                if not obj[0] in ids:

                    ids[obj[0]] = []


                ids[obj[0]].append( Visualize.centerWH(obj[1], obj[2], obj[3], obj[4]) )


        self.ids = ids
        # print(ids)


    def listColors(self):

        list_rgb = {}

        for i, c in enumerate(COLORS):

            rgb = to_rgb(c)
            rgb = tuple( [rgb[0] * 255, rgb[1] * 255, rgb[2] * 255])

            list_rgb[i] = rgb

            # print(rgb)


        self.list_rgb = list_rgb
        self.max_colors = len(COLORS)






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


    @staticmethod
    def centerWH(x, y, w, h):

        return int(x+(w/2)), int(y+(h/2))


    def draw_bbox(self, img, thickness=2):

        # color = (0, 0, 255)

        for bbox in self.frames[frame]:

            x1, y1, x2, y2 = Visualize.fromWHtoX(int(bbox[1]), int(bbox[2]), int(bbox[3]), int(bbox[4]))

            path = self.ids[bbox[0]]
            color = self.list_rgb[ bbox[0] % self.max_colors ]

            # print(color, (x1, y1), (x2, y2))

            img = cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
            img = Visualize.drawPath(img, path, color, 1)


        return img


    @staticmethod
    def drawPath(img, path, color, thickness=2):

        old = ()

        for idx, coord in enumerate(path):

            if idx == 0:

                old = coord
                continue


            img = cv2.line(img, old, coord, color, thickness=thickness)

            old = coord


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

    frame = 1

    while True:
    # while frame < 30:

        img = visual_s.readFrame(frame)

        if img is None: break

        visual_s.draw_bbox(img)
        visual_s.writeVideo(img)

        frame += 1


    visual_s.saveVideo()
