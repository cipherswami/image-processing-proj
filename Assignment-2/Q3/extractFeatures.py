import cv2
import numpy as np

# # Load the image
# image_path = 'test/r1.jpg'
# image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# # Initialize SIFT detector
# sift = cv2.SIFT_create()

# # Detect keypoints and compute descriptors
# keypoints, descriptors = sift.detectAndCompute(image, None)

# # Convert keypoints to a NumPy array
# keypoints_data = np.array([[kp.pt[0], kp.pt[1], kp.size, kp.angle, kp.response, kp.octave, kp.class_id] for kp in keypoints])

# # Store descriptors and keypoints inside the code
# descriptors_code = descriptors.tolist()
# keypoints_code = keypoints_data.tolist()

# with open("descriptor.log", 'w') as fd:
#     print(descriptors_code, file=fd)

# with open("keypoints.log", 'w') as fd:
#     print(keypoints_code, file=fd)

# Load the image
image_path = 'test/r4.jpg'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Initialize SIFT detector
sift = cv2.SIFT_create()

# Detect keypoints and compute descriptors
descriptors = sift.detectAndCompute(image, None)[1]

stored_descriptors = descriptors.tolist()

with open('descriptors.log', 'w') as fp:
    fp.write(str(stored_descriptors))

