import cv2
import numpy as np

def houghLineAngle(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 20, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 20)
    if lines is not None:
        for rho, theta in lines[0]:
            angle = np.degrees(theta) - 90
    return angle

def contourRectAngle(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binarized = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    contours = cv2.findContours(binarized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    largest_contour = max(contours, key=cv2.contourArea)
    return cv2.minAreaRect(largest_contour)[-1]

def solution(image_path):
    #############################################
    #############################################
    # Author: CIPH3R <aravindswami135@gmail.com>
    #############################################

    # Load the image
    img = cv2.imread(image_path)

    # HoughLines rotation to change it to clockwise 
    # tilted image
    if houghLineAngle(img) < 0:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    # Angle calculation of anti-clockwise tilted images
    # using Canny
    angle = contourRectAngle(img)
    print("Roation Angle: " + str(angle))
    
    # General Affine Rotation
    center = tuple(np.array(img.shape[1::-1]) / 2)
    rMatrix = cv2.getRotationMatrix2D(center, angle, 0.8)
    rotatedImage =  cv2.warpAffine(img, rMatrix, img.shape[1::-1], flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    ############################
    ############################

    return rotatedImage