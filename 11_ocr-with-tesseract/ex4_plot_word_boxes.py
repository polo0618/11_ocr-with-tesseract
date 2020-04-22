import re
import cv2 
import numpy as np
import pytesseract
from pytesseract import Output
from image_preprocessing import get_grayscale, remove_noise, thresholding, dilate, erode, opening, canny, deskew, match_template

IMG_DIR = 'images/'

image = cv2.imread(IMG_DIR + 'invoice-sample.jpg')
d = pytesseract.image_to_data(image, output_type=Output.DICT)
print('DATA KEYS: \n', d.keys())

n_boxes = len(d['text'])
for i in range(n_boxes):
    # condition to only pick boxes with a confidence > 60%
    if int(d['conf'][i]) > 60:
        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
        image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

b,g,r = cv2.split(image)
rgb_img = cv2.merge([r,g,b])
cv2.imshow('SAMPLE INVOICE WITH WORD LEVEL BOXES', rgb_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
