import cv2
from icecream import ic
import numpy as np
import glob


def slidingWindowIndices(flatImageSize, kernelSize, depth): #gets the relevant sliding window indices so that a kernel can be easily convolved/cross-correlated
    kernelSize = kernelSize[0]
    kernelCellSize = pow(kernelSize, 2)
    
    imageHeight = flatImageSize[0]
    imageWidth = flatImageSize[1]
    imageCellSize = imageHeight * imageWidth
    indices = np.arange(imageCellSize).reshape(imageHeight, imageWidth)
    

    numberSlidingWindows = (imageHeight - kernelSize + 1) * (imageWidth - kernelSize + 1)


    offsets = np.zeros((kernelSize, kernelSize), dtype=int)
    i = np.arange(kernelSize)
    offsets[i, :] = (imageWidth - kernelSize) * i

    offsets = offsets.flatten(order='F')

    numberInvalidsOnEdge = kernelSize - 1     
    imageWithoutInvalidPortions = indices[:-numberInvalidsOnEdge, :]
    imageWithoutInvalidPortions = imageWithoutInvalidPortions[:, :-numberInvalidsOnEdge]

    indexer = np.arange(kernelCellSize).reshape(1, -1) + imageWithoutInvalidPortions.reshape(-1, 1) + offsets

    d = np.arange(depth)
    newIndexer = np.zeros((np.shape(indexer)[0], (kernelCellSize * depth)))

    
    newPeek = np.zeros(depth)

    newPeek[d]= d * imageCellSize #add the offsets to peek depth-wise


    oldIndexShape = indexer.shape
    indexerCellSize = oldIndexShape[0] * oldIndexShape[1]
     
    # ic(indexer)
    indexer = np.broadcast_to(indexer, (depth, indexer.shape[0], indexer.shape[1])) #extend indexer array depth-wise 
    newPeek = np.broadcast_to(newPeek[..., None], (newPeek.shape[0], indexerCellSize)) #repeat each number of newPeek (indexer.shape[0] * indexer.shape[1]) times (assume indexer isn't flattened yet)
    newPeek = np.resize(newPeek, (depth, oldIndexShape[0], oldIndexShape[1])) #resize the repeated newPeek array so that each number of the old newPeek is now a matrix of old indexer's shape 
    
   
    newIndexer = (indexer + newPeek).astype('int')
    
    newIndexer = np.concatenate(newIndexer, axis=1)



    return newIndexer

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

    def convolve(self, kernel, bias, train=False):
        images = np.array(self.images)
        #imageSize = images[0].shape
        #print(f'flat image size was {imageSize[0]} by {imageSize[1]},', f'kernel size was {kernel.shape},', f'depth was {imageSize[2]}')
        indices = slidingWindowIndices((36, 64), (3, 3), 3)
        
        for image in images:
            pass
        extendedIndices = np.broadcast_to(indices, (images.shape[0], indices.shape[0], indices.shape[1])) #extend indices array depth-wise 
        windows = image.flatten()[indices]
        dotProducts = np.einsum('ij,j->i', windows, kernel.flatten()) #take dot product of each sliding window with the kernel
        dotProducts += bias
        ic(windows.shape)
            



        if not train: 
            self.log.append({
                'type': 'convolution',
                'kernel': kernel,
                'inputs': images, #contains all of the images that were convoluted 
                #'outputs': convolved #contains all of the now convoluted images
            })

            # print(f'convolved {np.shape(convolved)[0]} images with size {np.shape(kernel)} kernel')
        
        
    def flatten(self, train=False):
        lastOutputs = np.array(self.log[-1]['outputs'])
        flattenedOutputsTwo = lastOutputs.reshape(3000, -1)
        ic(np.shape(flattenedOutputsTwo))



first = CNN('data/Training-Set-10/*.png')
first.convolve(np.arange(27).reshape(3, 3, 3), 10)


# ic(slidingWindowIndices((36, 6), (3, 3), 3))

cv2.waitKey(0)
cv2.destroyAllWindows()


