from typing import final
import cv2
from icecream import ic
import numpy as np
import glob
#-- timer for timing things yay --
import time



class TimerError(Exception):
    """A custom exception used to report errors in use of Timer class"""

class Timer:
    def __init__(self):
        self._start_time = None

    def start(self):
        """Start a new timer"""
        if self._start_time is not None:
            raise TimerError(f"Timer is running. Use .stop() to stop it")

        self._start_time = time.perf_counter()

    def stop(self):
        """Stop the timer, and report the elapsed time"""
        if self._start_time is None:
            raise TimerError(f"Timer is not running. Use .start() to start it")

        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None
        return '{:.6f}'.format(elapsed_time)
        #print(f"Elapsed time: {elapsed_time:0.6f} seconds")
# -- -- 


t = Timer()

class CNN:
    def __init__(self, dir, blueprint):
        self.log = []
        self.dir = dir
        self.blueprint = blueprint
        self.finalWeights = None
        filenames = glob.glob(dir) #get all filenames in the specfied directory
        filenames.sort() #sort by alpha order
        self.images = np.array([cv2.imread(img) for img in filenames]) / 255 #read each file, constrain to interval (0,1)
        ic(np.shape(self.images))
        

        print(f'Loaded {len(filenames)} images from "{dir}"')

        #read the blueprint given and feedforward with it
        currentImageSet = self.images #initialize the first image set to work with as the images given
        for operation in blueprint:
            operationType = operation['operationType']
            if operationType == 'convolve':
                kernel = operation['kernel']
                t.start()
                bias = operation['bias']
                operation['outputImageSet'] = self.convolve(kernel, bias, currentImageSet)
                currentImageSet = operation['outputImageSet']
                print(f'Completed convolution in {t.stop()} seconds, image ID {np.sum(currentImageSet / 10000)}')
            if operationType == 'ReLU':
                t.start()
                operation['outputImageSet'] = np.maximum(0, currentImageSet)
                ic(operation['outputImageSet'].shape)
                currentImageSet = operation['outputImageSet']
                print(f'Completed ReLU in {t.stop()} seconds, image ID {np.sum(currentImageSet / 10000)}')
        ic(self.blueprint)
                

    def convolve(self, kernel, bias, images, train=False):
        kernelShape = (kernel.shape[0], kernel.shape[1])
        
        #ensure that the views don't look anywhere except for up and down on the images with axis=(1,2)
        views = np.lib.stride_tricks.sliding_window_view(images, kernelShape, axis=(1, 2))
        kernel = kernel[np.newaxis, np.newaxis, np.newaxis, ...] #modify kernel shape for broadcasting

        #element wise multiply the views with the kernels, then sum over a LOT of axes to get the final result
        #einsum is so smart it even broadcasts the element wise multiplication O_O
        finalConvolution = np.einsum('ijklmn,ijklmn->ijk', views, kernel) 
        finalConvolution += bias 

        ic(finalConvolution.shape)

        return finalConvolution



rng = np.random.default_rng()
first = CNN('data/Training-Set-10/*.png', [{
    'operationType': 'convolve', 
    'kernel': (np.random.random_sample((3, 3, 3)) - 0.5), 
    'bias': 0.05,
    'outputImageSet': None
}, 
{
    'operationType': 'ReLU',
    'outputImageSet': None
}])

cv2.waitKey(0)
cv2.destroyAllWindows()


