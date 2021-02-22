
import os
import shutil


path = 'images/MOT20'
end = 'detections/default/MOT20'


for folder in os.listdir(path):

	new_f  = os.path.join(end, folder)
	origin = os.path.join(path, folder, 'det')

	os.mkdir(new_f)
	shutil.move(origin, new_f)


