import re
import cv2 
import numpy as np
import pytesseract
from pytesseract import Output
from image_preprocessing import get_grayscale, remove_noise, thresholding, dilate, erode, opening, canny, deskew, match_template

IMG_DIR = 'images/'

# Plot boxes around text that matches a certain regex template
# In this example we will extract the date from the sample invoice

image = cv2.imread(IMG_DIR + 'invoice-sample.jpg')
d = pytesseract.image_to_data(image, output_type=Output.DICT)
date_pattern = '^(0[1-9]|[12][0-9]|3[01])/(0[1-9]|1[012])/(19|20)\d\d$'

n_boxes = len(d['text'])
for i in range(n_boxes):
    if int(d['conf'][i]) > 60:
        if re.match(date_pattern, d['text'][i]):
            (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
            image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

b,g,r = cv2.split(image)
rgb_img = cv2.merge([r,g,b])
cv2.imshow('SAMPLE INVOICE WITH BOXES FOR DATES', rgb_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
