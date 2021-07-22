
import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment

from metrics.Efforce import Efforce


class Test_eff(Efforce):

    def __init__(self):

        super().__init__()

        self.alfa = 0.5
        self.beta = 0.5


    def cost_matrix(self, v1, v2):


        # x, y, w, h ---> x1, y1, x2, y2
        v1m = self.coord2corner(v1)
        v2m = self.coord2corner(v2)

        # Cost matrix based on IoU
        matrix = self.iou(v1m, v2m)


        row, col = linear_sum_assignment(matrix)
        
        cost = matrix[row, col].sum()

        return cost, row, col, matrix



    def names(self):
        # return ['Det. Efforce', 'Trk. Efforce', 'Trk. over Det.', 'MOT Acc', 'IDs Efforce']
        return ['S', 'Sf', 'Sd', 'St', 'Y', 'Nd', 'Nt', 'Id', 'It']


    def intra_frame(self):

        Id = np.zeros((self.K))
        It = np.zeros((self.K))
        Nd = np.zeros((self.K))
        Nt = np.zeros((self.K))
        Qd = np.zeros((self.K))
        Qt = np.zeros((self.K))
        Ef = np.zeros((self.K))
        E  = np.zeros((self.K))
        Y  = np.zeros((self.K))

        V  = np.zeros((self.K))
        Ud = np.zeros((self.K))
        Ut = np.zeros((self.K))
        Ad = np.zeros((self.K))
        At = np.zeros((self.K))

        for k in range(self.K):

            kp = k + 1

            # Check values.
            if not kp in self.ut:  self.ut[kp] = np.zeros((0, 5))
            if not kp in self.ud:  self.ud[kp] = np.zeros((0, 5))

            # Variables.
            V[k]  = len(self.v[kp])
            Ud[k] = len(self.ud[kp])
            Ut[k] = len(self.ut[kp])


            # Calculate association costs and IDSW
            Ad[k], row_d, _, _       = self.cost_matrix(self.ud[kp][:, 1:], self.v[kp][:, 1:])
            idsw, At[k], row_t, _ = self.IDSW(self.ut[kp], self.v[kp])


            len_d = len(row_d)
            len_t = len(row_t)

            

            if len_d > 0:  Id[k] = 1 - (Ad[k] / len_d)
            elif V[k] == 0:   Id[k] = 1
            else:             Id[k] = 0

            if len_t > 0:  It[k] = 1 - (At[k] / len_t)
            elif V[k] == 0:   Id[k] = 1
            else:             It[k] = 0


            # Cardinality comparision
            Nd[k] = 1 - (abs(Ud[k] - V[k]) / max(V[k], Ud[k]))
            Nt[k] = 1 - (abs(Ut[k] - V[k]) / max(V[k], Ut[k]))



            # Quality of bounding boxes metric
            Qd[k] = (self.alfa * Id[k]) + ((1 - self.alfa) * Nd[k])
            Qt[k] = (self.alfa * It[k]) + ((1 - self.alfa) * Nt[k])
            Y[k]  = Qt[k] - Qd[k]



            # Tracking metric
            # Ef[k] = 1 - idsw[k] / V[k]

            if len_t > 0:
                Ef[k] = (1 - idsw / len_t) * 0.5

            elif V[k] == 0:
                Ef[k] = 1

            else:
                Ef[k] = 0

            E[k] = Ef[k] + Y[k]
            


        
        Sd = sum(Qd) / self.K
        St = sum(Qt) / self.K
        Sf = sum(Ef) / self.K
        S  = sum(E) / self.K

        Y_a  = sum(Y) / self.K

        Id_a = sum(Id) / self.K
        It_a = sum(It) / self.K
        Nd_a = sum(Nd) / self.K
        Nt_a = sum(Nt) / self.K


        # V_a  = sum(V) / self.K
        # Ud_a = sum(Ud) / self.K
        # Ut_a = sum(Ut) / self.K
        # Ad_a = sum(Ad) / self.K
        # At_a = sum(At) / self.K




        # self.plotDet(Qd, Qt, Y)
        # self.plotDet_basic(Nd, Nt, Id, It)


        # return Sd, St, Y_a, S, idsw_a, It_a, Nt_a
        return S, Sf, Sd, St, Y_a, Nd_a, Nt_a, Id_a, It_a




    def inter_frame(self):
        return None

    def join_metrics(self, intra, inter):

        return intra



    def plotDet(self, Qd, Qt, Y):

        plt.figure(figsize=(10, 6))

        # Detection compare
        plt.plot(Qd,  label='Ed (Detection Efforce BBoxes)')
        plt.plot(Qt,  label='Et (Tracking Efforce BBoxes)')
        plt.plot(Y,  label='Y (Efforce Difference)')


        plt.ylabel('Score')
        plt.xlabel('Frame number')
        plt.title('%s / %s performance bounding boxes' % (self.detector, self.tracker))
        plt.legend()
        # plt.show()

        plt.savefig('outputs/figs/det_%s_%s.jpg' % (self.detector, self.tracker))
        plt.clf()
        plt.close()


    def plotDet_basic(self, Nd, Nt, Id, It):

        plt.figure(figsize=(10, 6))

        # Detection compare
        plt.plot(Nd,  label='Nd (Cardinality BBoxes)')
        plt.plot(Nt,  label='Nt (Cardinality BBoxes)')
        plt.plot(Id,  label='Id (Overlap with BBoxes)')
        plt.plot(It,  label='It (Overlap with BBoxes)')


        plt.ylabel('Score')
        plt.xlabel('Frame number')
        plt.title('%s / %s performance bounding boxes' % (self.detector, self.tracker))
        plt.legend()
        # plt.show()

        plt.savefig('outputs/figs/det_basic_%s_%s.jpg' % (self.detector, self.tracker))
        plt.clf()
        plt.close()
        # plt.close('all')