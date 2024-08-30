import cv2
import numpy as np
import math

def distance(x, y, p, q): # Distance Function between two points- Euclidean
    return np.sqrt((x-p)**2 + (y-q)**2)

def gaussian(x, sigma): # Gaussian Function with zero mean and variance sigma
    return (1.0 /np.sqrt(2 * math.pi * (sigma ** 2)))* math.exp(- (x ** 2) / (2 * sigma ** 2))

def weight(image,i,j,k,l,sigma_c,sigma_d): # Weight function to calculate the gaussians across variation in pixel positions and intensities
	fg = gaussian(image[k][l] - image[i][j], sigma_c) # Function to spread the image intensities across a window in a gaussian curve
	gs = gaussian(distance(i, j, k, l), sigma_d) # Function to spread the varaition in pixel positions from a particular pixel across a window in a gaussian curve
	w = fg * gs	# Dot product of the two functions
	return w

def designed_bilateral_filter(image,d,sigma_c,sigma_d): 
	b,g,r = cv2.split(image)
	bilatb=np.zeros(b.shape)
	bilatg=np.zeros(g.shape)
	bilatr=np.zeros(r.shape)

	m= int((d-1)/2) # Radius calulation, ie, distance form the center pixel to the corresponding window
	for i in range(m,b.shape[0]-m): # Starting window at pixel position (m,m) so as to avoid corners
		for j in range(m,b.shape[1]-m):
			Wp1=0 # Total weight initlaized to zero every-time the window moves to next pixel
			Wp2=0
			Wp3=0		
			for k in range(i-m,m+i+1):		# Window function of diameter 'd'
				for l in range(j-m,m+j+1):		
					#print(weight(i,j,k,l))				
					Wp1+=weight(b,i,j,k,l,sigma_c,sigma_d) # Calculation of weight function across each channel and adding all of them up (In a particular Window)
					Wp2+=weight(g,i,j,k,l,sigma_c,sigma_d)
					Wp3+=weight(r,i,j,k,l,sigma_c,sigma_d)				
					bilatb[i,j]+=b[k,l]*weight(b,i,j,k,l,sigma_c,sigma_d) # Calculation of denoised pixel at position (i,j) accross every channel 
					bilatg[i,j]+=g[k,l]*weight(g,i,j,k,l,sigma_c,sigma_d)
					bilatr[i,j]+=r[k,l]*weight(r,i,j,k,l,sigma_c,sigma_d)
					if (k==m+i) and (l==m+j):
						bilatb[i,j]=int(round(bilatb[i,j]/Wp1)) # At the end of the window, we must divide by the total wweight function across every pixel in that window.
						bilatg[i,j]=int(round(bilatg[i,j]/Wp2))
						bilatr[i,j]=int(round(bilatr[i,j]/Wp3))
	for i in range(0,m):
		for j in range(0,b.shape[1]):
			bilatb[i,j]=b[i,j]
		bilatg[i,j]=g[i,j]
		bilatr[i,j]=r[i,j]
	for i in range(b.shape[0]-m-1,b.shape[0]):
		for j in range(0,b.shape[1]):
			bilatb[i,j]=b[i,j]
			bilatg[i,j]=g[i,j]
			bilatr[i,j]=r[i,j]
	for i in range(0,b.shape[0]):
		for j in range(0,m):
			bilatb[i,j]=b[i,j]
			bilatg[i,j]=g[i,j]
			bilatr[i,j]=r[i,j]
	for i in range(0,b.shape[0]):
		for j in range(b.shape[1]-m-1,b.shape[1]):
			bilatb[i,j]=b[i,j]
			bilatg[i,j]=g[i,j]
			bilatr[i,j]=r[i,j]

	bilat1 = np.zeros(image.shape) # Stacking all the channels in matrix
	bilat1[..., 0] = bilatb # Red Matrix
	bilat1[..., 1] = bilatg # Green Matrix
	bilat1[..., 2] = bilatr # Red Matrix
	return bilat1

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
    diameter = 1
    sigmaColor = 4
    sigmaSpace = 0.05

    # Applying Cross Bilateral Filter
    proLightImage = designed_bilateral_filter(flashImage, diameter, sigmaColor, sigmaSpace)

    ############################
    return proLightImage

