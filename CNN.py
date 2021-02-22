import cv2
from icecream import ic
import numpy as np
from pprint import pprint as p
# import math
import glob

# img = cv2.imread('Illinois1.png')
# def s(array):
#     print(f'size was {np.shape(array)}')

class CNN:
    def __init__(self, dir):
        self.log = []
        self.dir = dir
        self.finalWeights = None
        filenames = glob.glob(dir) #get all filenames in the specfied directory
        filenames.sort() #sort by alpha order
        self.images = [cv2.imread(img) for img in filenames] #read each file
        # self.images = [np.squeeze(np.dsplit(img, 3)) for img in self.images] #split the tensor into individual channels 
        ic(np.shape(self.images))

        print(f'Loaded {len(filenames)} images from "{dir}"')

    def convolve(self, images, kernel, train=False):

        # convolved = signal.correlate(images, kernel, 'same')

        
                

        #pads the image so that the input is the same as the output (to avoid losing spatial data)
        #only convolve about the 1 and 2 axes b/c those are the axes parallel to the image plane

        if not train: 
            self.log.append({
                'type': 'convolution',
                'kernel': kernel,
                'inputs': images, #contains all of the images that were convoluted 
                'outputs': convolved #contains all of the now convoluted images
            })

            # print(f'convolved {np.shape(convolved)[0]} images with size {np.shape(kernel)} kernel')
        
        
    def flatten(self, train=False):
        lastOutputs = np.array(self.log[-1]['outputs'])
        flattenedOutputsTwo = lastOutputs.reshape(3000, -1)
        ic(np.shape(flattenedOutputsTwo))
        

    # def backprop(self):






# first = CNN("data/Training-Set-10/*.png")
# first.convolve(first.images, np.random.rand(3000, 3, 3, 3))
# first.flatten()

def slidingWindow(imageSize, kernelSize, depth):
    kernelSize = kernelSize[0]
    kernelCellSize = pow(kernelSize, 2)
    
    imageHeight = imageSize[0]
    imageWidth = imageSize[1]
    imageCellSize = imageHeight * imageWidth
    image = np.arange(imageHeight * imageWidth).reshape(imageHeight, imageWidth)
    image3D = np.arange(imageHeight * imageWidth * depth).reshape(imageHeight, imageWidth, depth)
    

    numberSlidingWindows = (imageHeight - kernelSize + 1) * (imageWidth - kernelSize + 1)


    offsets = np.zeros((kernelSize, kernelSize), dtype=int)
    i = np.arange(kernelSize)
    offsets[i, :] = (imageWidth - kernelSize) * i

    offsets = offsets.flatten(order='F')
    wastedWindows = (imageWidth - kernelSize - 1) * imageHeight
    invalidImagePortionH = np.delete(image, np.arange((imageWidth - kernelSize)+1), axis=1).flatten() #remove parts of the image that cant be convoluted
    invalidImagePortionW = np.delete(image, np.arange((imageHeight - kernelSize)+1), axis=0).flatten()

    invalidImagePortion = np.concatenate((invalidImagePortionH, invalidImagePortionW))
    invalidImagePortion = invalidImagePortion[invalidImagePortion < (numberSlidingWindows + wastedWindows)]

    
    # ic(numberSlidingWindows + wastedWindows, invalidImagePortionW, invalidImagePortionH)


    indexer = np.arange(kernelCellSize).reshape(1, -1) + (np.arange(numberSlidingWindows + wastedWindows).reshape(-1, 1)) + offsets

    
    indexer = np.delete(indexer, invalidImagePortion, axis=0)  
    # indexer = indexer[:-(imageHeight - kernelSize + 1), :]
    # indexer = np.expand_dims(indexer, axis=2)
    # indexer = np.moveaxis(indexer, 1, 2)

    ic(np.shape(indexer))


    newIndexer = []
    for i in indexer:
        new = []
        for d in range(0, depth):
            new.append(i+(d*imageCellSize))

        new = np.array(new).flatten()

        newIndexer.append(new)
        # new = np.concatenate(new).flatten()
        # y = np.resize(indexer[i], (1, 27))
        # y = new

    

    ic(newIndexer)

    ic(image3D.flatten())
    # ic(indexer)

    return image3D.flatten()[np.array(newIndexer)]



a = np.reshape(np.arange(36*3), (6, 6, 3))
b = np.reshape(np.arange(9), (3, 3))

ic(slidingWindow((6, 6), (3, 3), 3))








cv2.waitKey(0)
cv2.destroyAllWindows()
# ic(split)


