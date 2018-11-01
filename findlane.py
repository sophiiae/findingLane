'''
    project for finding lane on input image.
'''
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

# read input image
img = mpimg.imread('test_images/solidWhiteRight.jpg')

# pirnt out info of input image and plot input image
print('This image is: ', type(img), 'with dimensions: ', img.shape)

# get x and y value of image 
y = img.shape[0]
x = img.shape[1]

imCopy = np.copy(img)

# set threshold to extract the lines
th = [200, 200, 200]
thresholds = (img[:, :, 0] < th[0]) | (img[:, :, 1] < th[1]) | (img[:, :, 2] < th[2])
imCopy[thresholds] = [0, 0, 0]

#plot both original image and line extracted image
plt.figure(figsize=(6, 4))
plt.subplot(211)
plt.imshow(img)
plt.subplot(212)
plt.imshow(imCopy)
plt.show()