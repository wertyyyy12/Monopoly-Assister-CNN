import cv2
from icecream import ic
import numpy as np
from scipy import signal
import glob

# img = cv2.imread('Illinois1.png')
class CNN:
    def __init__(self, dir, log=[]):
        self.log = log
        self.dir = dir
        filenames = glob.glob(dir) #get all filenames in the specfied directory
        filenames.sort() #sort by alpha order
        self.images = [cv2.imread(img) for img in filenames] #read each file
        self.images = [np.squeeze(np.dsplit(img, 3)) for img in self.images] #split the tensor into individual channels 
        ic(np.shape(self.images))

        print(f'Loaded {len(filenames)} images from "{dir}"')

    def convolve(self, tensorImg, kernel):

        #pads the image so that the input is the same as the output (to avoid losing spatial data)
        #only convolve about the 1 and 2 axes b/c those are the axes parallel to the image plane
        convolved = signal.oaconvolve(tensorImg, kernel, 'same', axes=[1, 2]) 

        self.log.append({
            'type': 'convolution',
            'input': tensorImg,
            'kernel': kernel
        })

        return convolved
        





first = CNN("data/Training-Set-10/*.png")
first.convolve(first.images[0], np.random.rand(3, 3, 3))
ic(first.log)


cv2.waitKey(0)
cv2.destroyAllWindows()
# ic(split)