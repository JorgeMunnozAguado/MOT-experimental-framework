
import os
import shutil



def move_public_detections(path, end):

    for folder in os.listdir(path):

        new_f  = os.path.join(end, folder)
        origin = os.path.join(path, folder, 'det')

        print(new_f)
        print(origin)

        os.mkdir(new_f)
        shutil.move(origin, new_f)



move_public_detections('dataset/MOT20', 'outputs/detections/public/MOT20')
move_public_detections('dataset/MOT17', 'outputs/detections/public/MOT17')
