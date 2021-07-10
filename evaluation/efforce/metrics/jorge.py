
import numpy as np

from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment

from metrics.Efforce import Efforce


class Jorge(Efforce):

    def __init__(self):

        super().__init__()


    def cost_matrix(self, v1, v2):

        # # x, y, w, h -> cx, cy
        # v1 = self.coord2center(v1)
        # v2 = self.coord2center(v2)

        # matrix = distance.cdist(v1, v2, 'cosine')

        # row, col = linear_sum_assignment(matrix)
        
        # cost = matrix[row, col].sum()

        # return cost



        # x, y, w, h ---> x1, y1, x2, y2
        v1 = self.coord2corner(v1)
        v2 = self.coord2corner(v2)

        # Cost matrix based on IoU
        matrix = self.iou(v1, v2)


        row, col = linear_sum_assignment(matrix)
        
        cost = matrix[row, col].sum()

        return cost, row, col, matrix



    def names(self):
        return ['Det. Efforce', 'Trk. Efforce', 'Trk. over Det.', 'MOT Acc']


    def intra_frame(self):

        Ed = np.zeros((self.K))
        Et = np.zeros((self.K))
        E  = np.zeros((self.K))
        Y  = np.zeros((self.K))

        for k in range(self.K):

            # Check values.
            if not k+1 in self.ut:  self.ut[k+1] = np.zeros((0, 5))
            if not k+1 in self.ud:  self.ud[k+1] = np.zeros((0, 5))

            # Variables.
            V  = len(self.v[k + 1])
            Ud = len(self.ud[k + 1])
            Ut = len(self.ut[k + 1])

            # Calculate association costs and IDSW
            Ad, _, _, _    = self.cost_matrix(self.ud[k + 1][:, 1:], self.v[k + 1][:, 1:])
            idws, At, _, _ = self.IDSW(self.ut[k + 1], self.v[k + 1])


            # Detection metric.
            if V == 0:  Ed[k] = 0
            else:       Ed[k] = 1 - (Ad / V) - (abs(Ud - V) / max(V, Ud))
            # print(V, Ud)


            # Tracking metrics.
            if Ut == 0: E[k] = 0
            else:       E[k] = (Ut - idws) / Ut
            
            E[k] += 1 - (abs(Ut - V) / (Ut + V))

            if V == 0:  Y[k] = 0
            else:       Y[k]  = Ed[k] + (At / V) - 1

            Et[k] = E[k] - Y[k]


        
        Sd = sum(Ed) / self.K
        St = sum(Et) / self.K
        S  = sum(E) / self.K
        Ya = - sum(Y) / self.K


        # print('det    track  TOT    err(track)')
        # print('%.2f,  %.2f,  %.2f,  %.2f' % (Sd, St, S, Ya))


        return Sd, St, Ya, S




    def inter_frame(self):
        return None

    def join_metrics(self, intra, inter):

        return intra