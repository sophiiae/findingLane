'''
    Find lane in input image
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

gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
blur = cv2.GaussianBlur(gray, (3, 3), 0)

edges = cv2.Canny(blur, 50, 100)

rho = 1 
theta = np.pi / 180
th = 1
min_length = 10 
max_gap = 2

lineCopy = np.copy(img)

lines = cv2.HoughLinesP(edges, rho, theta, th, np.array([]), min_length, max_gap)

for line in lines:
    for x1, y1, x2, y2 in line: 
        cv2.line(lineCopy, (x1, y1), (x2, y2), (255, 0, 0), 10)

color_edges = np.dstack((edges, edges, edges))
combo = cv2.addWeighted(color_edges, 1, lineCopy, 1, 1)

regionCopy = np.copy(img)

left_bot = [0, 539]
left_top = [400, 350]
right_bot = [959, 539]
right_top = [600, 350]

fit_left = np.polyfit((left_bot[0], left_top[0]), (left_bot[1], left_top[1]), 1)
fit_right = np.polyfit((right_bot[0], right_top[0]), (right_bot[1], right_top[1]), 1)
fit_top = np.polyfit((left_top[0], right_top[0]), (left_top[1], right_top[1]), 1)
fit_bottom = np.polyfit((left_bot[0], right_bot[0]), (left_bot[1], right_bot[1]), 1)

X, Y = np.meshgrid(np.arange(0, x), np.arange(0, y))
region_th = (Y > (X*fit_left[0] + fit_left[1])) & \
         (Y > (X*fit_right[0] + fit_right[1])) & \
         (Y < (X*fit_bottom[0] + fit_bottom[1])) & \
         (Y > (X*fit_top[0]) + fit_top[1])
regionCopy[region_th] = [0, 255, 0]

colorLines = np.copy(lineCopy)
color_th = (lineCopy[:,:,0] > 250) & (lineCopy[:,:,1] < 10) & (lineCopy[:,:,2] < 10)

colorLines[color_th & region_th] = [0, 0, 255]

#plot both original image and line extracted images
plt.figure(figsize=(6, 10))
plt.subplot(311)
plt.imshow(combo)
plt.subplot(312)
plt.imshow(regionCopy)
plt.subplot(313)
plt.imshow(edges, cmap='Greys_r')
# plt.imshow(colorLines)
plt.show()
