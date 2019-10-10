import numpy as np
from joint_bilateral_filter import Joint_bilateral_filter
import cv2
import json
import sys

if len(sys.argv) != 2:
    print("USAGE: python3 hw1.py [GROUP]")
    exit(0)

img_dir = './testdata/'
group = sys.argv[1]

f = open('error'+group+'.csv', 'w')
print('task,sigma_s,sigma_r,wb,wg,wr,error', file=f)

for alpha in ['a', 'b', 'c']:

    img = cv2.imread(img_dir+group+alpha+'.png')

    for sigma_s in [1, 2, 3]:
        for sigma_r in [0.05, 0.1, 0.2]:
            
            JBF = Joint_bilateral_filter(sigma_s, sigma_r)

            for i in range(11):
                for j in range(10-i, -1, -1):
                    ws = np.array([i, j, 10-i-j])/10

                    guidance = (img @ ws).astype(np.uint8)

                    jbf_out = JBF.joint_bilateral_filter(img, guidance).astype(np.uint8)
                    bf_out = JBF.joint_bilateral_filter(img, img).astype(np.uint8)
        
                    error = np.sum(np.abs(bf_out-jbf_out))

                    print(alpha,sigma_s,sigma_r,ws[0],ws[1],ws[2],error, file = f, sep=',')
                    print(alpha,sigma_s,sigma_r,ws[0],ws[1],ws[2],error)
