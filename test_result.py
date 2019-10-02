import numpy as np
import cv2

from joint_bilateral_filter import Joint_bilateral_filter

sigma_s = 3
sigma_r = 0.2


img = cv2.imread('testdata/0b.png')
guidance = (img.astype(float) @ np.array([0, 0.1, 0.9])).astype(np.uint8)

# create JBF class
JBF = Joint_bilateral_filter(sigma_s, sigma_r)

jbf_out = JBF.joint_bilateral_filter(img, guidance).astype(np.uint8)
bf_out = JBF.joint_bilateral_filter(img, img).astype(np.uint8)

error = np.sum(np.abs(bf_out-jbf_out))
print(error)

cv2.imshow('img', img)
cv2.imshow('gui', guidance)
cv2.imshow('jbf', jbf_out)
cv2.imshow('bf', bf_out)

cv2.waitKey(0)
cv2.destroyAllWindows()