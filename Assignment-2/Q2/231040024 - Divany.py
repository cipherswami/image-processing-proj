import cv2
import numpy as np

def cross_bilateral_filter(source, guide, diameter, sigma_color, sigma_space):
    # Padding for convolution
    pad_width = diameter // 2
    padded_source = np.pad(source, [(pad_width, pad_width), (pad_width, pad_width), (0, 0)], mode='reflect')
    padded_guide = np.pad(guide, [(pad_width, pad_width), (pad_width, pad_width), (0, 0)], mode='reflect')

    # Output image
    output = np.zeros_like(source)

    # Pre-compute Gaussian spatial weights
    y, x = np.mgrid[-pad_width:pad_width+1, -pad_width:pad_width+1]
    g_space = np.exp(-(x*2 + y*2) / (2 * sigma_space*2))

    # Reshape g_space to (diameter, diameter, 1) for broadcasting
    g_space = g_space[:, :, np.newaxis]

    # Iterate over each pixel in the source image
    for i in range(pad_width, padded_source.shape[0] - pad_width):
        for j in range(pad_width, padded_source.shape[1] - pad_width):
            # Extract local regions
            source_region = padded_source[i - pad_width:i + pad_width + 1, j - pad_width:j + pad_width + 1]
            guide_region = padded_guide[i - pad_width:i + pad_width + 1, j - pad_width:j + pad_width + 1]

            # Compute Gaussian range weights
            g_range = np.exp(-((guide_region - guide[i-pad_width, j-pad_width])*2) / (2 * sigma_color*2))

            # Combine space and range weights
            weights = g_space * g_range
            weights /= weights.sum(axis=(0, 1), keepdims=True)

            # Apply weights to the source region
            filtered_value = (weights * source_region).sum(axis=(0, 1))
            output[i-pad_width, j-pad_width] = filtered_value
    return output

def solution(image_path_a, image_path_b):
    '''Function to get the prolight image using cross bilateral filter'''
    ############################
    ############################
    ## image_path_a is path to the non-flash high ISO image
    ## image_path_b is path to the flash low ISO image
    ############################
    ############################

    # Load the Images
    flashImage = cv2.imread(image_path_b)
    noFlashImage = cv2.imread(image_path_a)

    # Bilateral Filter Parameters
    diameter = 5
    sigmaColor = 30
    sigmaSpace = 27

    # Applying Cross Bilateral Filter
    proLightImage = cross_bilateral_filter(flashImage, noFlashImage, diameter, sigmaColor, sigmaSpace)

    ############################
    return proLightImage

