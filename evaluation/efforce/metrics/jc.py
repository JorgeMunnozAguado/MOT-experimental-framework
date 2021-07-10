
import numpy as np


from scipy.optimize import linear_sum_assignment

from metrics.Efforce import Efforce


class JC(Efforce):

    def __init__(self):

        pass


    def cost_matrix(self, v1, v2):
        
        v1 = self.coord2corner(v1)
        v2 = self.coord2corner(v2)

        matrix = self.iou(v1, v2)

        row, col = linear_sum_assignment(matrix)
        
        cost = matrix[row, col].sum()

        return cost



    def tracking_efforce(self):
        pass


    def detection_efforce(self):
        pass


    def intra_frame(self):

        # Create variables.
        E  = np.zeros((self.K))
        Ed = np.zeros((self.K))
        Et = np.zeros((self.K))


        for k in range(self.K):


            Ed[k] = self.cost_matrix(self.ud[k + 1][:, 1:], self.v[k + 1][:, 1:])
            Et[k] = self.cost_matrix(self.ut[k + 1][:, 1:], self.v[k + 1][:, 1:])


        E = sum(abs(Ed - Et))


        # for k in range(self.K):

        #     print(E, Ed[k])

        S = (1 / self.K) * sum([(1 - ((E / Ed[k]) if Ed[k] != 0 else 0)) for k in range(self.K)])


        return S


    def inter_frame(self):
        
        # Create variables.
        E  = np.zeros((self.K))
        Ed = np.zeros((self.K))
        Et = np.zeros((self.K))


        for k in range(self.K - 1):


            Ed[k] = self.cost_matrix(self.ud[k + 1], self.v[k + 2])
            Et[k] = self.cost_matrix(self.ut[k + 1], self.v[k + 2])


        E = sum(abs(Ed - Et))


        S = (1 / self.K) * sum([(1 - ((E / Ed[k]) if Ed[k] != 0 else 0)) for k in range(self.K)])


        return S


    def join_metrics(self, intra, inter, alfa=0.5):

        return (alfa * intra) + ((1 - alfa) * inter)


