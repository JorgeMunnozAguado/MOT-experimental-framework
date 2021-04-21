
import os
import numpy as np



classes_dict = {1 : 'Pedestrian',
                2 : 'Person_on_vehicle',
                3 : 'Car',
                4 : 'Bicycle',
                5 : 'Motorbike',
                6 : 'Non_motorized_vehicle',
                7 : 'Static_person',
                8 : 'Distractor',
                9 : 'Occluder',
                10 : 'Occluder_on_the_ground',
                11 : 'Occluder_full',
                12 : 'Reflection',
                13 : 'Crowd'}



def readFile(path):
    '''Read a GT or predictions file in the indicated
    path. Return a dict were the keys are frames.
    '''

    file = np.loadtxt(path, delimiter=',')

    unique = np.unique(file[:, 0])


    frame = {}

    for u in unique:

        a = np.where(file[:, 0] == u)

        frame[u] = file[a][:, 1:]


    return frame



def int2class(frame, classes_dict, permited=[1, 2, 3, 4, 5, 6, 7]):

    out_array = []

    for i, label in enumerate(frame[:, 6]):

        if not label in permited: continue

        # out_list.append( classes_dict[label] )

        aux_frame = np.full((6), "", dtype=object)

        # Label string
        # aux_frame[0] = classes_dict[label]
        aux_frame[0] = 'object'

        # Confidence
        aux_frame[1] = frame[i, 5]

        # Bounding box
        aux_frame[2:6] = frame[i, 1:5]


        # width, height -> x2, y2
        aux_frame[4] = aux_frame[2] + aux_frame[4]
        aux_frame[5] = aux_frame[3] + aux_frame[5]


        out_array.append( aux_frame )


    return np.asarray(out_array)



def processFrame(frame, filename, gt=False):

    # <class_name> <left> <top> <right> <bottom>
    final_frame = np.zeros( (frame.shape[0], 6))

    final_frame = int2class(frame, classes_dict)


    if gt: np.savetxt(filename, final_frame[:, [0, 2, 3, 4, 5]], delimiter=' ', fmt=['%s', '%.0f', '%.0f', '%.0f', '%.0f'])
    else:  np.savetxt(filename, final_frame, delimiter=' ', fmt=['%s', '%.2f', '%.0f', '%.0f', '%.0f', '%.0f'])


def cleanFile(path):

    os.system("rm -r %s/*" %(path))



def processSequence(gt_path, det_path, img_path, gt_auxiliar='evaluation/mAP/auxiliar/GT', det_auxiliar='evaluation/mAP/auxiliar/DET'):

    # Clean axiliar folder
    cleanFile(gt_auxiliar)
    cleanFile(det_auxiliar)


    # Read files with data
    gt_file  = readFile(gt_path)
    det_file = readFile(det_path)


    # For each frame (create aux file)
    for key, gt_frame in gt_file.items():

        gt_filename  = os.path.join(gt_auxiliar, '%.6d.txt'%key)
        det_filename = os.path.join(det_auxiliar, '%.6d.txt'%key)


        processFrame(gt_frame, gt_filename, gt=True)
        processFrame(det_file[key], det_filename)


    os.system("python evaluation/mAP/mAP.py --img_path " + img_path)



if __name__ == '__main__':

    list_detectors = os.listdir('outputs/detections')
    # list_sets = ['MOT17', 'MOT20']
    list_sets = ['MOT17']

    output_file = os.path.join('outputs/evaluation', 'mAP.txt')
    
    verbose = True

    with open(output_file, "w") as file:

        if verbose: print('File open')

        file.write('| Detector | Subset name | mAP |\n')
        file.write('--------------------------------\n')


        for detector in list_detectors:

            # list_sets = os.listdir('dataset')
            if verbose: print('DETECTOR:  ', detector)

            for set_name in list_sets:

                list_subsets = os.listdir( os.path.join('dataset', set_name) )

                for subset in list_subsets:

                    if verbose: print('->', set_name, subset)

                    if detector in ['public']: continue

                    gt_path  = os.path.join('dataset/', set_name, subset, 'gt/gt.txt')
                    img_path = os.path.join('dataset/', set_name, subset, 'img1')
                    det_path = os.path.join('outputs/detections', detector, set_name, subset, 'det/det.txt')


                    processSequence(gt_path, det_path, img_path)


                    with open('evaluation/mAP/auxiliar.txt', 'r') as f:

                        mAP = f.read()

                        if verbose: print(mAP)


                    file.write('| ' + detector + ' | ' + set_name + '/' + subset + ' | ' + mAP + ' | \n')







    # detector = 'yolo3'
    # det_path = os.path.join('outputs/detections', detector, 'MOT17/MOT17-02/det/det.txt')

    # print('DETECTOR:  ', detector)
    # processSequence(gt_path, det_path)

