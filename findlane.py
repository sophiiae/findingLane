import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

#read input image
img = mpimg.imread('test_images/solidWhiteRight.jpg')

#pirnt out info of input image and plot input image
print('This image is: ', type(img), 'with dimensions: ', img.shape)
plt.imshow(img)
plt.show()
