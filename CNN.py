import cv2
from icecream import ic
import numpy as np
import glob


def slidingWindowIndices(imageSize, kernelSize, depth): #gets the relevant sliding window indices so that a kernel can be easily convolved/cross-correlated
    kernelSize = kernelSize[0]
    kernelCellSize = pow(kernelSize, 2)
    
    imageHeight = imageSize[0]
    imageWidth = imageSize[1]
    imageCellSize = imageHeight * imageWidth
    indices = np.arange(imageHeight * imageWidth).reshape(imageHeight, imageWidth)
    indices3D = np.arange(imageHeight * imageWidth * depth)
    

    numberSlidingWindows = (imageHeight - kernelSize + 1) * (imageWidth - kernelSize + 1)


    offsets = np.zeros((kernelSize, kernelSize), dtype=int)
    i = np.arange(kernelSize)
    offsets[i, :] = (imageWidth - kernelSize) * i

    offsets = offsets.flatten(order='F')
    wastedWindows = (imageWidth - kernelSize - 1) * imageHeight
    invalidImagePortionH = np.delete(indices, np.arange((imageWidth - kernelSize)+1), axis=1).flatten() #remove parts of the image that cant be convoluted
    invalidImagePortionW = np.delete(indices, np.arange((imageHeight - kernelSize)+1), axis=0).flatten()

    invalidImagePortion = np.concatenate((invalidImagePortionH, invalidImagePortionW))
    invalidImagePortion = invalidImagePortion[invalidImagePortion < (numberSlidingWindows + wastedWindows)]

    indexer = np.arange(kernelCellSize).reshape(1, -1) + (np.arange(numberSlidingWindows + wastedWindows).reshape(-1, 1)) + offsets

    
    indexer = np.delete(indexer, invalidImagePortion, axis=0)  

    d = np.arange(depth)
    newIndexer = np.zeros((np.shape(indexer)[0], (kernelCellSize * depth)))

    
    newPeek = np.zeros(depth)

    newPeek[d]= d * imageCellSize #add the offsets to peek depth-wise


    oldIndexShape = indexer.shape
    indexerCellSize = oldIndexShape[0] * oldIndexShape[1]
    
    
    indexer = np.broadcast_to(indexer, (depth, indexer.shape[0], indexer.shape[1])) #extend indexer array depth-wise 
    newPeek = np.broadcast_to(newPeek[..., None], (newPeek.shape[0], indexerCellSize)) #repeat each number of newPeek (indexer.shape[0] * indexer.shape[1]) times (assume indexer isn't flattened yet)
    newPeek = np.resize(newPeek, (depth, oldIndexShape[0], oldIndexShape[1])) #resize the repeated newPeek array so that each number of the old newPeek is now a matrix of old indexer's shape 
    
    
    newIndexer = indexer + newPeek
    newIndexer = np.concatenate(newIndexer, axis=1)
    



    return indices3D.flatten()[np.array(newIndexer, dtype=int)]

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




a = np.reshape(np.arange(36*3), (6, 6, 3))
b = np.reshape(np.arange(9), (3, 3))

ic(slidingWindowIndices((6, 6), (3, 4), 3))

cv2.waitKey(0)
cv2.destroyAllWindows()


