import numpy as np
import cv2


class Joint_bilateral_filter(object):
    

    def __init__(self, sigma_s, sigma_r, border_type='reflect'):
        
        self.border_type = border_type
        self.sigma_r = sigma_r
        self.sigma_s = sigma_s
        self.r = 3*self.sigma_s
        self.s_kernel = Joint_bilateral_filter.create_skernel(self.sigma_s, self.r).reshape(-1)


    def create_w_cache(self):
        d = 255*255*2*self.sigma_r**2
        return [np.exp(-float(i)/d) for i in range(255**3)]

    def joint_bilateral_filter(self, input_img, guidance):

        pad_img = np.pad(input_img, [(self.r, self.r), (self.r, self.r), (0 , 0)], 'symmetric').astype(float)
        guidance_padlist = [(self.r, self.r)]*2 + [(0, 0)] * (len(guidance.shape)-2)
        pad_guidance = np.pad(guidance, guidance_padlist, 'symmetric').astype(float)

        output = np.zeros(input_img.shape)
        
        for i in range(output.shape[0]):
            for j in range(output.shape[1]):
                print(i, j)
                patch_g = pad_guidance[i:i+2*self.r+1, j:j+2*self.r+1]
                patch_i = pad_img[i:i+2*self.r+1, j:j+2*self.r+1]

                r_kernel = self.create_rkernel(patch_g, i+self.r, j+self.r)

                w = self.s_kernel*r_kernel
                patch_i = np.reshape(patch_i, (-1, 3))
                
                w = w.reshape((1, -1))
                v = (w @ patch_i)/np.sum(w)

                output[i, j] = v

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

    def create_rkernel(self, patch, x, y):
        
        diff = patch - patch[self.r, self.r]
        diff = np.square(diff)

        if len(diff.shape) > 2:
            diff = np.sum(diff, -1)

        diff = np.reshape(diff, -1)


        diff /= 255**2*2*self.sigma_r**2
        diff *= -1
        kernel = np.exp(diff)

        return kernel
        