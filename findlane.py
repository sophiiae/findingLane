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

# convert image to grayscale and use Gaussian blur to smooth the edges. 
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
blur = cv2.GaussianBlur(gray, (3, 3), 0)
edges = cv2.Canny(gray, 100, 200)

# define region of interest and make cropped image
cropMask = np.zeros_like(edges)
pts = np.array([[(0, 539), (959, 539),(520, 325), (445, 325)]], np.int32)
colorMatch = 255
cv2.fillPoly(cropMask, pts, colorMatch)
cropImg = cv2.bitwise_and(edges, cropMask)


# use Hough transform to extract lines
rho = 1
theta = np.pi / 180
th = 1
min_length = 40
max_gap = 25

lines = cv2.HoughLinesP(cropImg, rho, theta, th, np.array([]), min_length, max_gap)
print("Hough output shape: ", lines.shape)

# draw lines on cropped image
lineCopy = np.copy(img)
lineImg = np.zeros((y, x, 3), dtype=np.uint8)

laneLeft = []
laneRight = []

for line in lines:
    for x1, y1, x2, y2 in line: 
        slope = (y2 - y1) / (x2 - x1)
        if slope < 0: 
            laneRight.extend([(x1, y1)])
            laneRight.extend([(x2, y2)])
        elif slope > 0:
            laneLeft.extend([(x1, y1)])
            laneLeft.extend([(x2, y2)])
        cv2.line(lineImg, (x1, y1), (x2, y2), (255, 0, 0), 4)
        
combo = cv2.addWeighted(lineCopy, 0.8, lineImg, 1, 0)

# plot both original image and line extracted images
fig = plt.figure(figsize=(12, 6))

plt.subplot(231)
plt.imshow(img)
plt.title("Original image")
plt.subplot(232)
plt.imshow(gray, cmap='gray')
plt.title("Grayscale image")
plt.subplot(233)
plt.imshow(edges, cmap='Greys_r')
plt.title("Canny Edge Detection")

plt.subplot(234)
plt.imshow(cropImg)
plt.title("Region of interest")
plt.subplot(235)
plt.imshow(lineImg)
plt.title("Extract lines")
plt.subplot(236)
plt.imshow(combo)
plt.title("Line segments in ROI")

plt.show()
