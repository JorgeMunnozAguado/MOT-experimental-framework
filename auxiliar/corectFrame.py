
import os
import numpy as np

if __name__ == '__main__':

    detector = 'efficientdet-d7x'

    path = os.path.join('outputs/detections', detector, 'MOT17/MOT17-02/det/det.txt')

    file = np.loadtxt(path, delimiter=',')

    print(file.shape)
    file[:, 0] = file[:, 0] + 1
    print(file[0])


    np.savetxt(path, file, delimiter=',', fmt=['%d', '%d', '%.0f', '%.0f', '%.0f', '%.0f', '%.4f', '%d', '%d', '%d'])