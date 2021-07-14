
import numpy as np

from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment

from metrics.Efforce import Efforce


class Test(Efforce):

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
        E  = np.zeros((self.K))
        Y  = np.zeros((self.K))


        V  = np.zeros((self.K))
        Ud  = np.zeros((self.K))
        Ut  = np.zeros((self.K))
        Ad  = np.zeros((self.K))
        At  = np.zeros((self.K))

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
            Ad[k], row_d, _, _    = self.cost_matrix(self.ud[k + 1][:, 1:], self.v[k + 1][:, 1:])
            idsw[k], At[k], row_t, _ = self.IDSW(self.ut[k + 1], self.v[k + 1])




            # Detection metric.
            # if V == 0:  Ed[k] = 0
            # else:       Ed[k] = (2 - (Ad / V) - (abs(Ud - V) / max(V, Ud))) * 0.5
            # print(V, Ud)


            # IoU correlation
            if V[k] == 0:
                Id[k] = 0
                It[k] = 0
                I[k]  = 0

            else:
                Id[k] = 1 - (Ad[k] / V[k])
                It[k] = 1 - (At[k] / V[k])
                I[k]  = It[k] - Id[k]

            # Number comparision
            Nd[k] = 1 - (abs(Ud[k] - V[k]) / max(V[k], Ud[k]))
            Nt[k] = 1 - (abs(Ut[k] - V[k]) / max(V[k], Ut[k]))
            N[k]  = Nt[k] - Nd[k]

            # Detection metric
            # Ed[k] = (Id[k] + Nd[k]) * 0.5
            Ed[k] = Id[k] * 0.75 + Nd[k] * 0.25



            # Tracking metric
            # Y[k] = (Ed[k] - It[k]) + N[k]


            # Y[k] = (Ed[k] - It[k])
            # Y[k] = It[k] - Id[k]


            Y[k] = (N[k] + I[k]) * 0.5


            E[k] = (1 - idsw[k] / V[k]) * 0.75 + (At[k] / V[k]) * 0.25
            # E[k] = (V[k] - idsw[k]) / V[k]
            E[k] = (E[k] + Nt[k]) * 0.5
            # Et[k] = ((E[k] + Nt[k]) * 0.5) - Y[k]
            Et[k] = E[k] + Y[k]














            # # Tracking metrics.
            # if Ut == 0: E[k] = 0
            # else:       E[k] = (Ut - idsw) / Ut
            
            # E[k] += 1 - (abs(Ut - V) / max(Ut, V))

            # if V == 0:  Y[k] = 0
            # else:       Y[k] = Ed[k] + (At / V) - 1

            # Et[k] = E[k] - Y[k]


        
        Sd = sum(Ed) / self.K
        St = sum(Et) / self.K
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
        plt.plot(Ed,  label='Ed (Detection Efforce)')
        plt.plot(Id,  label='Id (Min. IoU)')
        plt.plot(Nd,  label='Nd (Diff. Detection numb.)')
        # plt.plot(Ud,  label='Ud')
        # plt.plot(V,  label='V')

        # plt.plot(Et,  label='Et')
        # plt.plot(E,  label='E')

        # plt.plot(Nd,  label='Nd')
        # plt.plot(Nt,  label='Nt')
        # plt.plot(Id,  label='Id')
        # plt.plot(It,  label='It')
        # plt.plot(Y,  label='Y')
        # plt.plot(V,  label='V')
        plt.ylabel('some numbers')
        plt.legend()
        plt.show()

        # print(Id_a, It_a, Y_a)

        print(asdf)


        # print('det    track  TOT    err(track)')
        # print('%.2f,  %.2f,  %.2f,  %.2f' % (Sd, St, S, Ya))


        # return Sd, St, Y_a, S, idsw_a, V_a, At_a
        return Sd, St, Y_a, S




    def inter_frame(self):
        return None

    def join_metrics(self, intra, inter):

        return intra