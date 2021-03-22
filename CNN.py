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
        print(f"Elapsed time: {elapsed_time:0.6f} seconds")
# -- -- 


t = Timer()

class CNN:
    def __init__(self, dir):
        self.log = []
        self.dir = dir
        self.finalWeights = None
        filenames = glob.glob(dir) #get all filenames in the specfied directory
        filenames.sort() #sort by alpha order
        self.images = np.array([cv2.imread(img) for img in filenames]) #read each file
        ic(np.shape(self.images))

        print(f'Loaded {len(filenames)} images from "{dir}"')

    def convolve(self, kernel, bias, images, train=False):
        kernelShape = (3,3)
        
        views = np.lib.stride_tricks.sliding_window_view(images, kernelShape, axis=(1, 2))
        kernel = kernel[None, None, None, ...] #modify kernel shape for broadcasting

        #element wise multiply the views with the kernels, then sum over a LOT of axes to get the final result
        #einsum is so smart it even broadcasts the element wise multiplication O_O
        finalConvolution = np.einsum('ijklmn,ijklmn->ijk', views, kernel) 
  
        ic(finalConvolution.shape)
        
        

        
        if not train: 
            self.log.append({
                'type': 'convolution',
                'kernel': kernel,
                'inputs': images, #contains all of the images that were convoluted 
                #'outputs': convolved #contains all of the now convoluted images
            })
        

        
    def ReLU(self, matrix):
        return np.maximum(0, matrix)


first = CNN('data/Training-Set-10/*.png')
first.convolve(np.arange(27).reshape(3, 3, 3), 10, first.images)

cv2.waitKey(0)
cv2.destroyAllWindows()


