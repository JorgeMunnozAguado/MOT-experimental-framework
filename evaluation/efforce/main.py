
import os
import numpy as np

from multiprocessing import Pool
from functools import partial

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



def run_metrics(metric_obj, gt_file, det_file, track_file, K, detector, tracker):

    values = metric_obj.evaluate(gt_file, det_file, track_file, detector, tracker)

    return values


def pretty_print(content, type, header=None, f=None):

    if header:

        for el in header:

            print('%-16.15s' % el, end='')
            if not f is None:  f.write('%s,' % el)


    for el in content[:-1]:

        if type == 'str':
            print('%-16.15s' % el, end='')
            if not f is None:  f.write('%s,' % el)

        elif type == 'flt':
            print('%-16.2f' % el, end='')
            if not f is None:  f.write('%.2f,' % el)


    # Print file
    if type == 'str':
        print('%-16.15s' % content[-1], end='')
        if not f is None:  f.write('%s' % content[-1])

    elif type == 'flt':
        print('%-16.2f' % content[-1], end='')
        if not f is None:  f.write('%.2f' % content[-1])


    print()
    if not f is None:  f.write('\n')



# def eval_parallel(s_name, metric_obj, det, tck):
def eval_parallel(det, metric_obj, s_name, tck):

    sets = s_name.split('-')[0]

    gt_path    = os.path.join('dataset', sets, s_name, 'gt/gt.txt')
    det_path   = os.path.join('outputs/detections', det, sets, s_name, 'det/det.txt')
    track_path = os.path.join('outputs/tracks', tck, det, sets, s_name + '.txt')


    gt_file, K    = load_file(gt_path, 'gt')
    track_file, _ = load_file(track_path, 'trc')

    if det == 'public':
        det_file, _ = load_file(det_path, 'public')

    else:
        det_file, _ = load_file(det_path, 'det')



    values = run_metrics(metric_obj, gt_file, det_file, track_file, K, det, tck)

    return det, values



if __name__ == '__main__':

    f = open("outputs/evaluation/own.csv", "w")
    # f = None

    num_proc = 8
    # num_proc = 1


    trackers  = ['sort', 'deep_sort', 'uma', 'sst']
    # trackers  = ['sort']
    # trackers  = ['deep_sort']
    # trackers  = ['uma']
    # trackers  = ['sst']
    # detectors = ['yolo3', 'efficientdet-d7x', 'faster_rcnn', 'faster_rcnn-fine-tune', 'gt']
    # detectors = ['yolo3', 'faster_rcnn', 'faster_rcnn-fine-tune', 'gt']
    # detectors = ['yolo3', 'public', 'faster_rcnn', 'faster_rcnn-mod-1', 'faster_rcnn-mod-2', 'faster_rcnn-mod-3', 'faster_rcnn-mod-4', 'faster_rcnn-fine-tune', 'gt']
    detectors = ['public', 'faster_rcnn', 'faster_rcnn-mod-1', 'faster_rcnn-mod-2', 'faster_rcnn-mod-3', 'faster_rcnn-mod-4', 'faster_rcnn-fine-tune', 'gt']
    # detectors = ['yolo3']
    # detectors = ['faster_rcnn']
    # detectors = ['faster_rcnn-mod-2']
    # detectors = ['faster_rcnn-mod-4']
    # detectors = ['gt']
    # detectors = ['public']
    # detectors = ['faster_rcnn-fine-tune']
    # datasets  = ['MOT20', 'MOT17']
    datasets  = ['MOT17']



    
    # from metrics.test import Test
    # from metrics.test2 import Test2
    from metrics.test_eff import Test_eff
    # metric_obj = Test()
    # metric_obj = Test2()
    metric_obj = Test_eff()
    # metric_obj = Jorge()
    # metric_obj = JC()
    # metric_obj = Fabio()


    pretty_print(metric_obj.names(), 'str', header=['Detector', 'Tracker', 'Sequence'], f=f)
    print('-------------------------------------------------------------------------------------------------------------------')


    for tck in trackers:

        values_avg = []

        # for det in detectors:
        for sets in datasets:

            subsets = os.listdir(os.path.join('dataset', sets))
            # subsets = ['MOT17-05']
            # subsets = ['MOT17-11']

            for s_name in subsets:

                # Parallel pool
                pool = Pool(num_proc)                

                func_charged = partial(eval_parallel, metric_obj=metric_obj, s_name=s_name, tck=tck)

                out = zip(pool.map(func_charged, detectors))

                for (values,) in out:

                    det = values[0]
                    values = values[1]

                    pretty_print(values, 'flt', header=[det, tck, s_name], f=f)


        print('-------------------------------------------------------------------------------------------------------------------')


    f.close()