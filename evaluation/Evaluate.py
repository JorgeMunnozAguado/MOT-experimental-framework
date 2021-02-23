
import os
import numpy as np

import motmetrics as mm



def readFile(path):

    file = np.loadtxt(path, delimiter=',')

    unique = np.unique(file[:, 0])


    frame = {}

    for u in unique:

        a = np.where(file[:, 0] == u)

        frame[u] = file[a][:, 1:]


    return frame



def extractFromFrame(frame, process=False):

    ids = frame[:, 0]
    boxes = frame[:, 1:5]


    # Ground Truth is already processed
    if process:

        boxes = np.stack(boxes, axis=0)
        # x1, y1, x2, y2 --> x1, y1, width, height
        boxes = np.stack((boxes[:, 0],
                          boxes[:, 1],
                          boxes[:, 2] - boxes[:, 0],
                          boxes[:, 3] - boxes[:, 1]),
                          axis=1)

    return ids, boxes


def updateMetric(acc, gt_id, out_id, distance):

    acc.update(
        gt_id,      # Ground truth objects in this frame
        out_id,     # Detector hypotheses in this frame
        distance    # Distances from object to hypotheses
    )


def evaluate_mot_accums(accums, names, generate_overall=True):

    mh = mm.metrics.create()

    summary = mh.compute_many(
        accums,
        metrics=mm.metrics.motchallenge_metrics,
        names=names,
        generate_overall=generate_overall)

    str_summary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names,
    )

    return str_summary



def evaluateData(tracker, detector, db, data):

    acc = mm.MOTAccumulator(auto_id=True)

    path_out = os.path.join('data/predictions/', tracker, detector, db, data)
    path_gt  = os.path.join('data/images/', db, data.split('.')[0] + '/gt/gt.txt')

    frame_out = readFile(path_out)
    frame_gt  = readFile(path_gt)

    for key in frame_out.keys():

        out_id, out_box = extractFromFrame(frame_out[key])
        gt_id,  gt_box  = extractFromFrame(frame_gt[key])


        distance = mm.distances.iou_matrix(gt_box, out_box, max_iou=0.5)


        updateMetric(acc, gt_id, out_id, distance)


    return acc




def listFiles(*args, path='data/predictions'):

    path = os.path.join(path, *args)

    return os.listdir(path)




if __name__ == '__main__':


    for tracker in listFiles():

        print(tracker)

        for detector in listFiles(tracker):

            print('--', detector)
            accums = []
            data_names = []

            for db in listFiles(tracker, detector):

                for data in listFiles(tracker, detector, db):

                    data_names.append(data)

                    accums.append( evaluateData(tracker, detector, db, data) )


            summary = evaluate_mot_accums(accums, data_names)
            print(summary)

