import cv2
import os
import numpy as np

data_dir = './testdata/'
output_dir = './output/'

if not os.path.exists(output_dir):
	os.makedirs(output_dir)

group = '0'

config = {
	'a': [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
	'b': [[1, 0, 0], [0, 0, 1]],
	'c': [[0, 1, 0], [0, 0, 1], [0.7, 0.3, 0]],
}

for tag in config:
	img = cv2.imread(data_dir+group+tag+'.png')
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	cv2.imwrite(output_dir+group+tag+'_gray.png', gray)
	for w in config[tag]:
		weights = np.array(w)
		cov_img = (img @ weights).astype(np.uint8)
		cv2.imwrite(output_dir+group+tag+'_'+str(w)+'.png', cov_img)
