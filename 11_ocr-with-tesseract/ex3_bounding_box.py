import re
import cv2 
import numpy as np
import pytesseract
from pytesseract import Output
from image_preprocessing import get_grayscale, remove_noise, thresholding, dilate, erode, opening, canny, deskew, match_template

IMG_DIR = 'images/'

# Plot original image

image = cv2.imread(IMG_DIR + 'invoice-sample.jpg')
b,g,r = cv2.split(image)
rgb_img = cv2.merge([r,g,b])
cv2.imshow('Original image', rgb_img)


# Plot character boxes on image using pytesseract.image_to_boxes() function

image = cv2.imread(IMG_DIR + 'invoice-sample.jpg')
h, w, c = image.shape
boxes = pytesseract.image_to_boxes(image) 
for b in boxes.splitlines():
    b = b.split(' ')
    image = cv2.rectangle(image, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (0, 255, 0), 2)

b,g,r = cv2.split(image)
rgb_img2 = cv2.merge([r,g,b])
cv2.imshow('SAMPLE INVOICE WITH CHARACTER LEVEL BOXES', rgb_img2)

cv2.waitKey(0)
cv2.destroyAllWindows()
