
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

        self.e = 0.0000001


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

        # Association scores
        Id = np.zeros((self.K))
        It = np.zeros((self.K))
        Ad = np.zeros((self.K))
        At = np.zeros((self.K))

        # Cardinality scores
        Nd = np.zeros((self.K))
        Nt = np.zeros((self.K))
        Ud = np.zeros((self.K))
        Ut = np.zeros((self.K))
        Ld = np.zeros((self.K))
        Lt = np.zeros((self.K))
        V  = np.zeros((self.K))

        # Final scores
        Qd = np.zeros((self.K))
        Qt = np.zeros((self.K))
        Y  = np.zeros((self.K))

        Eintra = np.zeros((self.K))



        for k in range(self.K):

            kp = k + 1

            # Check values.
            if not kp in self.ut:  self.ut[kp] = np.zeros((0, 5))
            if not kp in self.ud:  self.ud[kp] = np.zeros((0, 5))


            # Calculate association costs and IDSW
            Ad[k], row_d, _, _    = self.cost_matrix(self.ud[kp][:, 1:], self.v[kp][:, 1:])
            At[k], row_t, _, _    = self.cost_matrix(self.ut[kp][:, 1:], self.v[kp][:, 1:])


            # Cardinaliry variables
            V[k]  = len(self.v[kp])
            Ud[k] = len(self.ud[kp])
            Ut[k] = len(self.ut[kp])
            Ld[k] = len(row_d)
            Lt[k] = len(row_t)
            
            # Association scores
            Id[k] = 1 - (Ad[k] / (Ld[k] + self.e))
            It[k] = 1 - (At[k] / (Lt[k] + self.e))

            # Cardinality comparision
            Nd[k] = 1 - (abs(Ud[k] - V[k]) / max(V[k], Ud[k]))
            Nt[k] = 1 - (abs(Ut[k] - V[k]) / max(V[k], Ut[k]))

            # Quality of bounding boxes metric
            Qd[k] = (self.alfa * Id[k]) + ((1 - self.alfa) * Nd[k])
            Qt[k] = (self.beta * It[k]) + ((1 - self.beta) * Nt[k])
            Y[k]  = Qt[k] - Qd[k]


            Eintra[k] = Qd[k] + Y[k]
            

        # Association scores
        Id = sum(Id) / self.K
        It = sum(It) / self.K
        Ad = sum(Ad) / self.K
        At = sum(At) / self.K

        # Cardinality scores
        Nd = sum(Nd) / self.K
        Nt = sum(Nt) / self.K
        Ud = sum(Ud) / self.K
        Ut = sum(Ut) / self.K
        Ld = sum(Ld) / self.K
        Lt = sum(Lt) / self.K
        V  = sum(V) / self.K
        
        # Final scores
        Qd = sum(Qd) / self.K
        Qt = sum(Qt) / self.K
        Y  = sum(Y) / self.K

        Eintra  = sum(Eintra) / self.K


        # self.plotDet(Qd, Qt, Y)
        # self.plotDet_basic(Nd, Nt, Id, It)


        return Eintra, Qd, Qt, Y, Nd, Nt, Id, It




    def inter_frame(self):


        gt_ids = self.gt_ids_counter()
        self.prev_traces = np.nan * np.zeros(len(gt_ids))

        for k in range(self.K):

            kp = k + 1

            idsw, _, row_t, _ = self.IDSW(self.ut[kp], self.v[kp])


        print('out/')


        return None


    def gt_ids_counter(self):

        list_ids = []

        for k in range(self.K):

            k_list = self.v[k + 1]

            list_ids = np.union1d(list_ids, k_list)

        return list_ids




    def calc_score_trace(self):

        for v in self.traces_id:
            pass



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