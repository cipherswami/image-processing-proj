import cv2
import numpy as np

def euclideanDistance(point1, point2):
    """Function to get min norm distance"""
    return (np.abs(point1[0] - point2[0]) + np.abs((point1[1] - point2[1])))

def getPerspective(img, points, width, height):
    """FUnction to perspectivly wrap image"""
    points1 = np.float32(points[0:4])
    points2 = np.float32([[0,0], [width, 0], [0, height], [width, height]])
    matrix = cv2.getPerspectiveTransform(points1, points2)
    return cv2.warpPerspective(img, matrix, [height, width])

def getCorners(contours, width, height):
    """Function to get the 4 corners of any rectangular contours"""
    closest_points_to_corners = {
        "top_left": None,
        "top_right": None,
        "bottom_left": None,
        "bottom_right": None
    }
    for contour in contours:
        for point in contour:
            # Calculate distances to each corner
            dist_to_top_left = euclideanDistance(point[0], (0, 0))
            dist_to_top_right = euclideanDistance(point[0], (width, 0))
            dist_to_bottom_left = euclideanDistance(point[0], (0, height))
            dist_to_bottom_right = euclideanDistance(point[0], (width, height))
            # Update the closest points for each corner
            if closest_points_to_corners["top_left"] is None or dist_to_top_left < closest_points_to_corners["top_left"][1]:
                closest_points_to_corners["top_left"] = (point[0], dist_to_top_left)
            if closest_points_to_corners["top_right"] is None or dist_to_top_right < closest_points_to_corners["top_right"][1]:
                closest_points_to_corners["top_right"] = (point[0], dist_to_top_right)
            if closest_points_to_corners["bottom_left"] is None or dist_to_bottom_left < closest_points_to_corners["bottom_left"][1]:
                closest_points_to_corners["bottom_left"] = (point[0], dist_to_bottom_left)
            if closest_points_to_corners["bottom_right"] is None or dist_to_bottom_right < closest_points_to_corners["bottom_right"][1]:
                closest_points_to_corners["bottom_right"] = (point[0], dist_to_bottom_right)
    corner_points = [
        closest_points_to_corners["bottom_left"][0],
        closest_points_to_corners["bottom_right"][0],
        closest_points_to_corners["top_left"][0],
        closest_points_to_corners["top_right"][0]
        ]
    return corner_points

def solution(image_path):
    image = cv2.imread(image_path)
    ######################################################################
    ######################################################################
    #####  WRITE YOUR CODE BELOW THIS LINE ###############################
    ######################################################################
    # Author: CIPH3R <aravindswami135@gmail.com>
    ######################################################################

    # Pre Processing of the image
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img =  cv2.GaussianBlur(img, (5, 5), 1)
    img = cv2.Canny(img, 50, 100)

    # Find the contours
    contourImg = image.copy()
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get image dimensions
    height, width = image.shape[:2]

    # Get the 4 corners of flag
    corner_points = getCorners(contours, height, width)

    # # Testing Code
    # for corner_point in corner_points:
    #     cv2.circle(contourImg, tuple(corner_point), 5, (0, 255, 0), -1)
    # cv2.imshow('Corners Img', contourImg)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    final = getPerspective(image, corner_points, 600, 600)

    # # Testing Code
    # cv2.imshow('Final Img', final)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    print("Done : " + image_path[-5:])

    ######################################################################

    return final
