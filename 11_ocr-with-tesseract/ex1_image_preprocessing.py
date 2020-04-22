import re
import cv2 
import numpy as np
import pytesseract
from pytesseract import Output
from image_preprocessing import get_grayscale, remove_noise, thresholding, dilate, erode, opening, canny, deskew, match_template

IMG_DIR = 'images/'

# Plot original image

image = cv2.imread(IMG_DIR + 'aurebesh.jpg')
b,g,r = cv2.split(image)
rgb_img = cv2.merge([r,g,b])
cv2.imshow('Original image', rgb_img)

# Plot preprocessed image

gray = get_grayscale(image)
cv2.imshow('Gray', gray)

thresh = thresholding(gray)
cv2.imshow('thresh', thresh)

opening = opening(gray)
cv2.imshow('opening', opening)

canny = canny(gray)
cv2.imshow('canny', canny)


cv2.waitKey(0)
cv2.destroyAllWindows()
