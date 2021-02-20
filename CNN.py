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
        # tensorImg = np.pad(tensorImg, 1, pad_with)
        kernelSize = np.shape(kernel)[0]
        convolved = []
        kernelNumber = 0
        for imgChannel in tensorImg:
            convolved.append(signal.oaconvolve(np.pad(imgChannel, int((kernelSize-1)/2), lambda vector, pad_width, iaxis, kwargs: None), kernel[kernelNumber], 'valid'))
            kernelNumber = kernelNumber + 1
            
        
        convolved = signal.oaconvolve(tensorImg, kernel, axes=[1, 2])
        ic(np.shape(convolved))
        ic(np.sum(convolved))
        return convolved
        





first = CNN("Training Set 10/*.png")
first.convolve(first.images[0], np.random.rand(3, 3, 3))


# print('hi')


cv2.waitKey(0)
cv2.destroyAllWindows()
# ic(split)