import cv2
from icecream import ic
import numpy as np
import glob

np.set_printoptions(threshold=1000)

def slidingWindowIndices(flatImageSize, kernelSize, depth): #gets the relevant sliding window indices so that a kernel can be easily convolved/cross-correlated
    kernelSize = kernelSize[0]
    kernelCellSize = pow(kernelSize, 2)
    
    imageHeight = flatImageSize[0]
    imageWidth = flatImageSize[1]
    fullImageCellSize = imageHeight * imageWidth
    indices = np.arange(fullImageCellSize).reshape(imageHeight, imageWidth)



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

    newPeek = d * fullImageCellSize #add the offsets to peek depth-wise


    oldIndexShape = indexer.shape
    indexerCellSize = oldIndexShape[0] * oldIndexShape[1]
     
    # ic(indexer)
    indexer = np.broadcast_to(indexer, (depth, indexer.shape[0], indexer.shape[1])) #extend indexer array depth-wise 
    # newPeek = np.broadcast_to(newPeek[..., None], (newPeek.shape[0], indexerCellSize)) #repeat each number of newPeek (indexer.shape[0] * indexer.shape[1]) times (assume indexer isn't flattened yet)
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
        self.images = np.array([cv2.imread(img) for img in filenames]) #read each file
        # self.images = [np.squeeze(np.dsplit(img, 3)) for img in self.images] #split the tensor into individual channels 
        ic(np.shape(self.images))

        print(f'Loaded {len(filenames)} images from "{dir}"')

    def convolve(self, kernel, bias, depth=3, train=False):
        images = self.images
        #imageSize = images[0].shape
        #print(f'flat image size was {imageSize[0]} by {imageSize[1]},', f'kernel size was {kernel.shape},', f'depth was {imageSize[2]}')
        indices = slidingWindowIndices((36, 64), (3, 3), 3)
        fullImageCellSize = images.shape[1] * images.shape[2] * images.shape[3] #the number of indices in a full 3d image
        numImages = images.shape[0] #number of images
        # for image in images:
        #     pass
            
        newIndices = np.broadcast_to(indices, (numImages, indices.shape[0], indices.shape[1])) #extend indices array depth-wise
        imageWisePeek = np.arange(numImages) * (fullImageCellSize)
        # imageWisePeek = np.broadcast_to(imageWisePeek[..., None], (numImages, fullImageCellSize)) #repeat the peek so that it has enough elements to be the size of the full image matrix
        imageWisePeek = np.resize(imageWisePeek, (numImages, indices.shape[0], indices.shape[1])) #reshape the size of the peek to the size of the full image matrix
    

        newIndices = (newIndices + imageWisePeek).astype('int')

     
        # newIndices = np.concatenate(newIndices, axis=1)
        ic(newIndices.shape)

        windows = images.flatten()[newIndices]
        ic(windows.shape)
        dotProducts = np.einsum('ijk,k->ij', windows, kernel.flatten()) #take dot product of each sliding window with the kernel
        dotProducts += bias #add bias to all dot products of all images at once (numpy broadcasting is going in like 3 dimensions!)
        old = dotProducts
        ic(old.shape)

        dotProducts = np.resize(dotProducts, (numImages, images.shape[1]-2, images.shape[2]-2))
        ic(dotProducts.shape)
        ic(old[0].shape, dotProducts[0].shape)
        



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


# ic(slidingWindowIndices((6, 6), (3, 3), 3))

cv2.waitKey(0)
cv2.destroyAllWindows()


