
import numpy as np

from abc import ABC, abstractmethod

class Efforce(ABC):

    def __init__(self):

        self.prev_gt = None
        self.prev_tr = None
        self.traces  = {}


    def evaluate(self, v, ud, ut, detector, tracker):

        self.v  = v
        self.ud = ud
        self.ut = ut
        
        self.K  = len(v)

        self.detector = detector
        self.tracker  = tracker


        intra = self.intra_frame()
        inter = self.inter_frame()

        values = self.join_metrics(intra, inter)

        return values



    @abstractmethod
    def cost_matrix(self, v1, v2):
        pass

    @abstractmethod
    def names(self):
        pass

    @abstractmethod
    def intra_frame(self):
        pass

    @abstractmethod
    def inter_frame(self):
        pass

    @abstractmethod
    def join_metrics(self, intra, inter):
        pass



    def IDSW(self, tr, gt):

        idws_c = 0


        cost, row, col, _ = self.cost_matrix(tr[:, 1:], gt[:, 1:])

        # print('------------------------')

        for r, c in zip(row, col):

            # print(r, c)
            if tr[r, 0] not in self.traces:

                self.traces[tr[r, 0]] = gt[c, 0]

            elif self.traces[tr[r, 0]] != gt[c, 0]:

                # print('ER ', tr[r, 0], gt[c, 0], self.traces[tr[r, 0]])
                self.traces[tr[r, 0]] = gt[c, 0]

                idws_c += 1

            

        # self.prev_gt = gt.copy()
        # self.prev_tr = tr.copy()

        # print(idws_c)

        return idws_c, cost, row, col



        # print(a)


    @staticmethod
    def coord2center(v):

        v2 = v.copy()

        v2[:, 0] = v2[:, 0] + v2[:, 2] / 2
        v2[:, 1] = v2[:, 1] + v2[:, 3] / 2

        return v2[:, :2]


    @staticmethod
    def coord2corner(v):

        v2 = v.copy()

        v2[:, 2] = v2[:, 0] + v2[:, 2]
        v2[:, 3] = v2[:, 1] + v2[:, 3]

        return v2


    @staticmethod
    def get_iou(v1, v2):
    

        xA = max(v1[0], v2[0])
        yA = max(v1[1], v2[1])
        xB = min(v1[2], v2[2])
        yB = min(v1[3], v2[3])

        interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))

        boxAArea = abs((v1[2] - v1[0]) * (v1[3] - v1[1]))
        boxBArea = abs((v2[2] - v2[0]) * (v2[3] - v2[1]))

        iou = interArea / float(boxAArea + boxBArea - interArea)

        return iou


    @staticmethod
    def iou(v1, v2):

        matrix = np.zeros((len(v1), len(v2)))

        for i, v1_bbx in enumerate(v1):

            for j, v2_bbx in enumerate(v2):

                matrix[i, j] = 1 - Efforce.get_iou(v1_bbx, v2_bbx)


        return matrix


        