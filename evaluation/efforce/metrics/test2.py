
import numpy as np

from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment

from metrics.Efforce import Efforce


class Test2(Efforce):

    def __init__(self):

        super().__init__()

        self.alfa = 0.5
        self.beta = 0.5


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
        # return ['Det. Efforce', 'Trk. Efforce', 'Trk. over Det.', 'MOT Acc', 'N Det', 'N Track', 'U Det', 'U Track', 'V']


    def intra_frame(self):

        Id = np.zeros((self.K))
        It = np.zeros((self.K))
        I  = np.zeros((self.K))
        Nd = np.zeros((self.K))
        Nt = np.zeros((self.K))
        N  = np.zeros((self.K))
        Ed = np.zeros((self.K))
        Et = np.zeros((self.K))
        Ef = np.zeros((self.K))
        E  = np.zeros((self.K))
        Y  = np.zeros((self.K))


        V  = np.zeros((self.K))
        Ud = np.zeros((self.K))
        Ut = np.zeros((self.K))
        Ad = np.zeros((self.K))
        At = np.zeros((self.K))

        len_d = np.zeros((self.K))
        len_t = np.zeros((self.K))

        idsw  = np.zeros((self.K))

        for k in range(self.K):

            # Check values.
            if not k+1 in self.ut:  self.ut[k+1] = np.zeros((0, 5))
            if not k+1 in self.ud:  self.ud[k+1] = np.zeros((0, 5))

            # Variables.
            V[k]  = len(self.v[k + 1])
            Ud[k] = len(self.ud[k + 1])
            Ut[k] = len(self.ut[k + 1])


            # Calculate association costs and IDSW
            Ad[k], row_d, _, _       = self.cost_matrix(self.ud[k + 1][:, 1:], self.v[k + 1][:, 1:])
            idsw[k], At[k], row_t, _ = self.IDSW(self.ut[k + 1], self.v[k + 1])


            len_d[k] = len(row_d)
            len_t[k] = len(row_t)

            

            if len_d[k] > 0:  Id[k] = 1 - (Ad[k] / len_d[k])
            else:             Id[k] = 0

            if len_t[k] > 0:  It[k] = 1 - (At[k] / len_t[k])
            else:             It[k] = 0
            
            I[k]  = It[k] - Id[k]


            # Number comparision
            Nd[k] = 1 - (abs(Ud[k] - V[k]) / max(V[k], Ud[k]))
            Nt[k] = 1 - (abs(Ut[k] - V[k]) / max(V[k], Ut[k]))
            N[k]  = Nt[k] - Nd[k]



            # Detection metric
            Ed[k] = (self.alfa * Id[k]) + ((1 - self.alfa) * Nd[k])
            Et[k] = (self.alfa * It[k]) + ((1 - self.alfa) * Nt[k])
            Y[k]  = Et[k] - Ed[k]
            # Ed[k] = Id[k] * 0.75 + Nd[k] * 0.25



            # Tracking metric
            # Ef[k] = 1 - idsw[k] / V[k]

            if len_t[k] > 0:
                Ef[k] = (1 - idsw[k] / len_t[k]) * 0.5
                Ef[k] +=  (len_t[k] / V[k]) * 0.5

            elif V[k] == 0:
                Ef[k] = 1
            else:
                Ef[k] = 0

            # -> GT objects: V
            # -> Track objects: Ut




            # E[k] = self.beta * Ef[k] + (1 - self.beta) * Et[k]
            # E[k] = self.beta * Ef[k] + (1 - self.beta) * Y[k]
            E[k] = Ef[k] + Y[k]
            # Y[k] = (Ed[k] - It[k]) + N[k]


            # Y[k] = (Ed[k] - It[k])
            # Y[k] = It[k] - Id[k]


            # Y[k] = (N[k] + I[k]) * 0.5


            # E[k] = (1 - idsw[k] / V[k]) * 0.75 + (At[k] / V[k]) * 0.25
            # E[k] = (1 - idsw[k] / V[k]) * 0.5 + It[k] * 0.25 + Nt[k] * 0.25
            # E[k] = (V[k] - idsw[k]) / V[k]
            # E[k] = (E[k] + Nt[k]) * 0.5
            # Et[k] = ((E[k] + Nt[k]) * 0.5) - Y[k]
            # Et[k] = E[k] + Y[k]














            # # Tracking metrics.
            # if Ut == 0: E[k] = 0
            # else:       E[k] = (Ut - idsw) / Ut
            
            # E[k] += 1 - (abs(Ut - V) / max(Ut, V))

            # if V == 0:  Y[k] = 0
            # else:       Y[k] = Ed[k] + (At / V) - 1

            # Et[k] = E[k] - Y[k]


        
        Sd = sum(Ed) / self.K
        St = sum(Et) / self.K
        Sf = sum(Ef) / self.K
        S  = sum(E) / self.K

        Y_a  = sum(Y) / self.K

        Id_a = sum(Id) / self.K
        It_a = sum(It) / self.K
        Nd_a = sum(Nd) / self.K
        Nt_a = sum(Nt) / self.K
        Ed_a = sum(Ed) / self.K


        V_a  = sum(V) / self.K
        Ud_a = sum(Ud) / self.K
        Ut_a = sum(Ut) / self.K
        Ad_a = sum(Ad) / self.K
        At_a = sum(At) / self.K

        idsw_a = sum(At) / self.K


        import matplotlib.pyplot as plt

        # Detection compare
        plt.plot(Ed,  label='Ed (Detection Efforce BBoxes)')
        plt.plot(Et,  label='Et (Tracking Efforce BBoxes)')
        plt.plot(Y,  label='Y (Efforce Difference)')


        # Detection data
        # plt.plot(Ed,  label='Ed (Detection Efforce)')
        # plt.plot(Id,  label='Id (Min. IoU)')
        # plt.plot(Nd,  label='Nd (Diff. Detection numb.)')

        # Detection detail
        # # plt.plot(Nd * 10,  label='Nd (Diff. Detection numb.) x10')        
        # plt.plot(Ud,  label='Ud')
        # # plt.plot(V,  label='V')

        # Tracking
        # plt.plot(Nt,  label='Nt')
        # # plt.plot(It,  label='It')
        # # plt.plot(idsw,  label='IDSW')

        # # plt.plot(Nt * 10,  label='Nt (Diff. Detection numb.) x10')        
        # plt.plot(Ut,  label='Ut')
        # plt.plot(V,  label='V')
        # # plt.plot(idsw,  label='IDSW')


        # # plt.plot(Et,  label='Et')
        # # plt.plot(E,  label='E')

        # # plt.plot(Nd,  label='Nd')
        # # plt.plot(Id,  label='Id')
        # # plt.plot(It,  label='It')
        # # plt.plot(Y,  label='Y')
        # # plt.plot(V,  label='V')


        plt.ylabel('Score')
        plt.xlabel('Frame number')
        plt.title('faster_rcnn / sort detection performance')
        plt.legend()
        plt.show()

        print(asdf)


        # return Sd, St, Y_a, S, idsw_a, It_a, Nt_a
        return Sd, St, Y_a, S, Sf




    def inter_frame(self):
        return None

    def join_metrics(self, intra, inter):

        return intra