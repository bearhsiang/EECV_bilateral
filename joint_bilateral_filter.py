import numpy as np
import cv2


class Joint_bilateral_filter(object):
    
    gr_cache = {}

    def __init__(self, sigma_s, sigma_r, border_type='reflect'):
        
        self.border_type = border_type
        self.sigma_r = sigma_r
        self.sigma_s = sigma_s
        self.r = 3*self.sigma_s
        self.s_kernel = Joint_bilateral_filter.create_kernel(self.sigma_s, self.r)

    def joint_bilateral_filter(self, input, guidance):
        ## TODO
        pad_img = np.pad(input, ((self.r, self.r), (self.r, self.r), (0, 0)), self.border_type)
        pass
        return output

    @staticmethod
    def create_kernel(sigma, r):
        k = np.zeros((2*r+1, 2*r+1))
        for i in range(r+1):
            for j in range(r+1):
                t = np.exp(-(i**2+j**2)/(2*sigma**2))
                k[r+i][r+j] = t
                k[r-i][r+j] = t
                k[r+i][r-j] = t
                k[r-i][r-j] = t
        return k