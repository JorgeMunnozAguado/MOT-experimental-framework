
import os
import numpy as np


from metrics.fabio import Fabio
from metrics.jorge import Jorge
from metrics.jc import JC



def load_file(path, type, labels=[1, 2, 3, 4, 5, 6, 7]):
    '''
    Load tracking file from path.

    Inputs:
        - path : path were the file is stored.
        - type : select from list: ['gt', 'det', 'public', 'trc']

    Outputs:
        - dict where key is the frame number and value detections
          in the frame.
    '''


    file = np.loadtxt(path, delimiter=',')

    unique = np.unique(file[:, 0])


    frame = {}

    for u in unique:

        a = np.where(file[:, 0] == u)

        # frame[u] = file[a][:, 1:]
        aux_frame = file[a][:, 1:]


        if type == 'gt':

            idx = np.where(np.isin(aux_frame[:, 6], labels))
            aux_frame = aux_frame[idx]

            aux_frame = aux_frame[:, :5]


        elif type == 'det':
            
            aux_frame = aux_frame[:, :5]


        elif type == 'public':
            
            aux_frame = aux_frame[:, :5]


        elif type == 'trc':

            aux_frame = aux_frame[:, :5]


        else: assert "Error, incorrect type"


        frame[u] = aux_frame

    return frame, len(unique)



def run_metrics(gt_file, det_file, track_file, K):

    # print(track_file.keys())


    ##########################################
    # FABIO
    ##########################################

    # intra_frame_f1(gt_file, det_file, track_file, K)

    # fabio_metric = Fabio()

    # fabio_metric.evaluate(gt_file, det_file, track_file)


    ##########################################
    # JORGE
    ##########################################

    jorge_metric = Jorge()

    values = jorge_metric.evaluate(gt_file, det_file, track_file)




    ##########################################
    # JUAN CARLOS
    ##########################################

    # jc = JC()

    # v = jc.evaluate(gt_file, det_file, track_file)

    # print(v)




    return values


def pretty_print(content, type, header=None):

    if header:

        for el in header:

            print('%-18.15s' % el, end='')


    for el in content:

        if   type == 'str': print('%-18.15s' % el, end='')
        elif type == 'flt': print('%-18.2f' % el, end='')

    print()




if __name__ == '__main__':


    # trackers  = ['sort', 'deep_sort', 'uma', 'sst']
    trackers  = ['sort']
    # trackers  = ['deep_sort']
    # trackers  = ['uma']
    # trackers  = ['sst']
    # detectors = ['yolo3', 'faster_rcnn', 'faster_rcnn-fine-tune', 'gt']
    # detectors = ['yolo3']
    detectors = ['gt']
    # detectors = ['public']
    datasets  = ['MOT17']



    
    jorge_metric = Jorge()
    pretty_print(jorge_metric.names(), 'str', header=['Detector', 'Tracker', 'Sequence'])
    print('-------------------------------------------------------------------------------------------------------------------')


    for tck in trackers:

        values_avg = []

        for det in detectors:

            for sets in datasets:


                # subsets = ['MOT17-02']
                subsets = ['MOT17-05']

                for s_name in subsets:


                    gt_path    = os.path.join('dataset', sets, s_name, 'gt/gt.txt')
                    det_path   = os.path.join('outputs/detections', det, sets, s_name, 'det/det.txt')
                    track_path = os.path.join('outputs/tracks', tck, det, sets, s_name + '.txt')


                    # print(gt_path)
                    # print(det_path)
                    # print(track_path)


                    gt_file, K    = load_file(gt_path, 'gt')
                    track_file, _ = load_file(track_path, 'trc')

                    if det == 'public':
                        det_file, _ = load_file(det_path, 'public')

                    else:
                        det_file, _ = load_file(det_path, 'det')



                    values = run_metrics(gt_file, det_file, track_file, K)

                    values_avg.append(list(values))

                    pretty_print(values, 'flt', header=[det, tck, s_name])


        values_avg = np.asarray(values_avg)

        values_avg = np.sum(values_avg, axis=0) / len(detectors)

        pretty_print(values_avg, 'flt', header=['AVG', tck, 'AVG'])
        print('-------------------------------------------------------------------------------------------------------------------')