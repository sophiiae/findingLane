import os
import scipy.misc as spmisc
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from helper import *
from moviepy.editor import VideoFileClip

os.listdir("test_images/")

def draw_lane_lines(image):
    # edge detection
    result = grayscale(image)
    result = gaussian_blur(result, 5)
    result = canny(result, 50, 150)
    
    # mask
    imshape = image.shape
    vertices = np.array([[
        (50,imshape[0]),
        (imshape[1] / 2 - 60, imshape[0] / 2 + 60), 
        (imshape[1] / 2 + 60, imshape[0] / 2 + 60), 
        (imshape[1] - 50, imshape[0])
    ]], dtype=np.int32)
    result = region_of_interest(result, vertices)

    # line detection
    rho = 2 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 15     # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 30 #minimum number of pixels making up a line
    max_line_gap = 20    # maximum gap in pixels between connectable line segments
    result = hough_lines(result, rho, theta, threshold, min_line_length, max_line_gap)
    
    result = weighted_img(result, image)
    return result

input_path = "test_images/"
output_path = "test_images_output/"

# for file in os.listdir(input_path):
#     image = mpimg.imread(os.path.join(input_path, file))
#     result = draw_lane_lines(image)
    
#     plt.figure()
#     plt.imshow(result)
#     plt.show()

#     spmisc.imsave(os.path.join(output_path, file), result)

def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)
    result = draw_lane_lines(image)
    return result

white_output = 'test_videos_output/solidWhiteRight.mp4'
##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!

yellow_output = 'test_videos_output/solidYellowLeft.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip2 = VideoFileClip('test_videos/solidYellowLeft.mp4').subclip(0,5)
clip2 = VideoFileClip('test_videos/solidYellowLeft.mp4')
yellow_clip = clip2.fl_image(process_image)
yellow_clip.write_videofile(yellow_output, audio=False)

challenge_output = 'test_videos_output/challenge.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip3 = VideoFileClip('test_videos/challenge.mp4').subclip(0,5)
clip3 = VideoFileClip('test_videos/challenge.mp4')
challenge_clip = clip3.fl_image(process_image)
challenge_clip.write_videofile(challenge_output, audio=False)