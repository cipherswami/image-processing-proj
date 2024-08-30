import cv2
import numpy as np
import math

def bilateralFilter(imageMatrix, windowLength=7, sigmaColor=25, sigmaSpace=9, maskImageMatrix=None):
    maskImageMatrix = np.zeros(
        (imageMatrix.shape[0], imageMatrix.shape[1])) if maskImageMatrix is None else maskImageMatrix
    imageMatrix = imageMatrix.astype(np.int32)
    def limit(x):
        x = 0 if x < 0 else x
        x = 255 if x > 255 else x
        return x
    limitUfun = np.vectorize(limit, otypes=[np.uint8])
    def lookForGaussianTable(delta):
        return deltaGaussianDict[delta]
    def generateBilateralFilterDistanceMatrix(windowLength, sigma):
        distanceMatrix = np.zeros((windowLength, windowLength, 3))
        leftBias = int(math.floor(-(windowLength - 1) / 2))
        rightBias = int(math.floor((windowLength - 1) / 2))
        for i in range(leftBias, rightBias + 1):
            for j in range(leftBias, rightBias + 1):
                distanceMatrix[i - leftBias][j - leftBias] = math.exp(-(i ** 2 + j ** 2) / (2 * (sigma ** 2)))
        return distanceMatrix
    deltaGaussianDict = {i: math.exp(-i ** 2 / (2 * (sigmaColor ** 2))) for i in range(256)}
    lookForGaussianTableUfun = np.vectorize(lookForGaussianTable, otypes=[np.float64])
    bilateralFilterDistanceMatrix = generateBilateralFilterDistanceMatrix(windowLength, sigmaSpace)
    margin = int(windowLength / 2)
    leftBias = math.floor(-(windowLength - 1) / 2)
    rightBias = math.floor((windowLength - 1) / 2)
    filterImageMatrix = imageMatrix.astype(np.float64)
    for i in range(0 + margin, imageMatrix.shape[0] - margin):
        for j in range(0 + margin, imageMatrix.shape[1] - margin):
            if maskImageMatrix[i][j] == 0:
                filterInput = imageMatrix[i + leftBias:i + rightBias + 1,
                                           j + leftBias:j + rightBias + 1]
                bilateralFilterValueMatrix = lookForGaussianTableUfun(np.abs(filterInput - imageMatrix[i][j]))
                bilateralFilterMatrix = np.multiply(bilateralFilterValueMatrix, bilateralFilterDistanceMatrix)
                bilateralFilterMatrix = bilateralFilterMatrix / np.sum(bilateralFilterMatrix, keepdims=False, axis=(0, 1))
                filterOutput = np.sum(np.multiply(bilateralFilterMatrix, filterInput), axis=(0, 1))
                filterImageMatrix[i][j] = filterOutput
    filterImageMatrix = limitUfun(filterImageMatrix)
    return filterImageMatrix

def solution(image_path_a, image_path_b):
    '''Function to get the prolight image using bilateral filter'''
    ######################################################################
    ## image_path_a is path to the non-flash high ISO image
    ## image_path_b is path to the flash low ISO image
    ######################################################################
    # Authour       : Aravind Potluri <aravindswami135@gmail.com>
    ######################################################################

    # Load the Images
    flashImage = cv2.imread(image_path_b)
    noFlashImage = cv2.imread(image_path_a)

    # Bilateral Filter Parameters
    diameter = 1
    sigmaColor = 4
    sigmaSpace = 0.05

    # Applying Cross Bilateral Filter
    noFlashImage = bilateralFilter(noFlashImage, diameter, sigmaColor, sigmaSpace)

    # Combine images based on flash information
    alpha = 0.5
    proLightImage = cv2.addWeighted(noFlashImage, alpha, flashImage, 1 - alpha, 0)

    ############################
    return proLightImage

