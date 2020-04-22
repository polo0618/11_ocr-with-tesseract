import re
import cv2 
import numpy as np
import pytesseract
from pytesseract import Output
from image_preprocessing import get_grayscale, remove_noise, thresholding, dilate, erode, opening, canny, deskew, match_template

IMG_DIR = 'images/'


image = cv2.imread(IMG_DIR + '3.png')
b,g,r = cv2.split(image)
rgb_img = cv2.merge([r,g,b])

gray = get_grayscale(image)

thresh = thresholding(gray)

opening = opening(gray)

canny = canny(gray)

# Get OCR output using Pytesseract

custom_config = r'--oem 3 --psm 13'
print('-----------------------------------------')
print('TESSERACT OUTPUT --> GRAY IMAGE')
print('-----------------------------------------')
print(pytesseract.image_to_string(gray, config=custom_config))
print('\n-----------------------------------------')
print('TESSERACT OUTPUT --> THRESHOLDED IMAGE')
print('-----------------------------------------')
print(pytesseract.image_to_string(thresh, config=custom_config))
print('\n-----------------------------------------')
print('TESSERACT OUTPUT --> OPENED IMAGE')
print('-----------------------------------------')
print(pytesseract.image_to_string(opening, config=custom_config))
print('\n-----------------------------------------')
print('TESSERACT OUTPUT --> CANNY EDGE IMAGE')
print('-----------------------------------------')
print(pytesseract.image_to_string(canny, config=custom_config))
