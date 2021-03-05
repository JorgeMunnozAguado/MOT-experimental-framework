
import os
import shutil


path = 'dataset/MOT20'
end = 'outputs/public/MOT20'


for folder in os.listdir(path):

    new_f  = os.path.join(end, folder)
    origin = os.path.join(path, folder, 'det')

    print(new_f)
    print(origin)

    os.mkdir(new_f)
    shutil.move(origin, new_f)


