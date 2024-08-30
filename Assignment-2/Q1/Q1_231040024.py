import cv2
import numpy as np

def isLavaImage(image):
    return True

def hsvThreshMask(image):
    hsvImage = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lowerBound = np.array([0, 120, 70])  # Lower bound for orange-red color in HSV
    upperBound = np.array([20, 255, 255])  # Upper bound for orange-red color in HSV
    mask = cv2.inRange(hsvImage, lowerBound, upperBound)
    return mask

def solution(image_path):
    '''
    This function generates mask for image containing lava.
    '''
    ######################################################################
    # Authour       : Aravind Potluri <aravindswami135@gmail.com>
    ######################################################################

    # Load the Image
    image = cv2.imread(image_path)

    # Detect non Lava Img
    if isLavaImage(image):
        # HSV Thresholding to generate Mask
        lavaMask = hsvThreshMask(image)
    else:
        lavaMask = np.zeros_like(image)

    ######################################################################      
    ######################################################################  
    return lavaMask
