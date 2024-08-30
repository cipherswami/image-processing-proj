import cv2
import numpy as np

def detect_sun(image_path):
    # Load image
    print(image_path)
    image = cv2.imread(image_path)
    # Looking for shadows
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=10, minLineLength=20, maxLineGap=100)
    shadows_found = False
    if lines is not None:
        for line in lines:
            for x1,y1,x2,y2 in line:
                if abs(x2-x1) < 3 and y2 < y1:
                    shadows_found = True
                    break
    print(shadows_found)
    # Convert the image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Define a range of yellow color in HSV
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    # Threshold the image to get only yellow pixels
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    # Count the number of yellow pixels
    yellow_pixel_count = cv2.countNonZero(mask)
    print(yellow_pixel_count)
    if (yellow_pct > 9000) and shadows_found:
        print("Sun is likely present") 
    else:
        print("No sun detected")

def hsvThreshMask(imagePath):
    image = cv2.imread(imagePath)
    hsvImage = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Define lower and upper bounds for lava color in HSV
    lowerBound = np.array([0, 120, 70])  # Lower bound for orange-red color in HSV
    upperBound = np.array([20, 255, 255])  # Upper bound for orange-red color in HSV
    # Threshold the image to create a binary mask
    mask = cv2.inRange(hsvImage, lowerBound, upperBound)
    return mask

def lumaMask(imagePath):
    # Read the input image
    image = cv2.imread(imagePath)
    # Convert the image from BGR to YCbCr color space
    ycbcr_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    # Define lower and upper bounds for lava color in YCbCr
    lower_bound = np.array([0, 133, 77])  # Lower bound for YCbCr lava color
    upper_bound = np.array([255, 173, 127])  # Upper bound for YCbCr lava color
    # Threshold the image to create a binary mask
    mask = cv2.inRange(ycbcr_image, lower_bound, upper_bound)
    # Find contours in the mask (optional)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Draw contours on the original image (optional)
    cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
    # # Display the result
    cv2.imshow('Lava and Stones Detection', image)
    # cv2.imshow('Lava Mask', mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return mask

def watershedMask(imagePath):
    '''
    This function uses watershed algorithm generate mask for dominating segment.
    '''
    image = cv2.imread(imagePath)
    # Image preprocessing
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binaryImage = cv2.threshold(grayImage, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]  
    # Perform morphological operations
    kernel = np.ones((3, 3), np.uint8)
    sureBG = cv2.dilate(binaryImage, kernel, iterations=3)
    # Distance transform to create distance markers
    disTransform = cv2.distanceTransform(binaryImage, cv2.DIST_L2, 5)
    sureFG = cv2.threshold(disTransform, 0.7 * disTransform.max(), 255, 0)[1]
    # Sure foreground area
    sureFG = np.uint8(sureFG)
    unknown = cv2.subtract(sureBG, sureFG)
    # Marker labeling
    markers = cv2.connectedComponents(sureFG)[1]
    markers = markers + 1
    # Mark the region of unknown with 0
    markers[unknown == 255] = 0
    markers = cv2.watershed(image, markers)
    image[markers == -1] = [0, 0, 255]  # Mark the boundary of the lava regions with red color
    # Create a binary mask for lava regions
    lavaMask = np.zeros_like(image)
    lavaMask[markers == 1] = [255, 255, 255]
    # # Display the lava mask
    # cv2.imshow('Lava Mask', lavaMask)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return lavaMask

# Usage
def solution(image_path):
    '''
    This function generates mask for image containing lava.
    '''
    ######################################################################
    # Authour       : Aravind Potluri <aravindswami135@gmail.com>
    ######################################################################

    # Load the image to Algorithm
    # lavaMask = watershedMask(image_path)
    # lavaMask = lumaMask(image_path)
    # detect_sun(image_path)
    lavaMask = hsvThreshMask(image_path)
    ######################################################################      ######################################################################  
    return lavaMask
