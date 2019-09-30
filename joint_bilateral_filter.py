import numpy as np
import cv2


class Joint_bilateral_filter(object):
    
    gr_cache = [-1]*(255**3)

    def __init__(self, sigma_s, sigma_r, border_type='reflect'):
        
        self.border_type = border_type
        self.sigma_r = sigma_r
        self.sigma_s = sigma_s
        self.r = 3*self.sigma_s
        self.s_kernel = Joint_bilateral_filter.create_skernel(self.sigma_s, self.r)

    def joint_bilateral_filter(self, input, guidance):
        ## TODO
        pad_img = np.pad(input, ((self.r, self.r), (self.r, self.r), (0, 0)), self.border_type)
        output = np.zeros(input.shape)
        for i in range(output.shape[0]):
            for j in range(output.shape[1]):
                print(i, j)
                patch = pad_img[self.r+i-self.r:self.r+i+self.r+1, self.r+j-self.r:self.r+j+self.r+1]
                r_kernel = Joint_bilateral_filter.create_rkernel(self.sigma_r, self.r, patch)
                w = self.s_kernel*r_kernel
                v = np.sum([w[i] @ patch[i] for i in range(patch.shape[0])], axis = 0)
                v /= np.sum(w)
                output[i, j] = v
        output = output.astype(np.uint8)
        cv2.imshow('test', output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return output

    @staticmethod
    def create_skernel(sigma, r):
        k = np.zeros((2*r+1, 2*r+1))
        for i in range(r+1):
            for j in range(r+1):
                t = np.exp(-(i**2+j**2)/(2*sigma**2))
                k[r+i][r+j] = t
                k[r-i][r+j] = t
                k[r+i][r-j] = t
                k[r-i][r-j] = t
        return k

    @staticmethod
    def create_rkernel(sigma, r, patch):
        
        kernel = np.zeros((2*r+1, 2*r+1))
        
        for i in range(patch.shape[0]):
            for j in range(patch.shape[1]):
                
                dis = np.sum(np.square(patch[i][j]-patch[r][r]))
            
                w = Joint_bilateral_filter.gr_cache[dis]
                if  w < 0:
                    w = np.exp(-float(dis)/(255.0**2)/(2*sigma**2))
                    Joint_bilateral_filter.gr_cache[dis] = w

                kernel[i][j] = w


        return kernel