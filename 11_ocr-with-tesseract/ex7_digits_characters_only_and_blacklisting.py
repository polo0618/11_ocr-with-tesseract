import re
import cv2 
import numpy as np
import pytesseract
from pytesseract import Output
from image_preprocessing import get_grayscale, remove_noise, thresholding, dilate, erode, opening, canny, deskew, match_template

IMG_DIR = 'images/'


# Plot original image

image = cv2.imread(IMG_DIR + 'digits-task.jpg')
b,g,r = cv2.split(image)
rgb_img = cv2.merge([r,g,b])
cv2.imshow('Original image', rgb_img)


#Output with outputbase digits

print('-----------------------------------------')
print('Digits only')
print('-----------------------------------------')
custom_config = r'--oem 3 --psm 6 outputbase digits'
print(pytesseract.image_to_string(image, config=custom_config))


# Output with a whitelist of characters (here, we have used all the lowercase characters from a to z only)

print('-----------------------------------------')
print('Whitelisted charaters only')
print('-----------------------------------------')
custom_config2 = r'-c tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyz --psm 6'
print(pytesseract.image_to_string(image, config=custom_config2))


# Output without the blacklisted characters (here, we have removed all digits)

print('-----------------------------------------')
print('Blacklist characters')
print('-----------------------------------------')
custom_config3 = r'-c tessedit_char_blacklist=0123456789 --psm 6'
print(pytesseract.image_to_string(image, config=custom_config3))

cv2.waitKey(0)
cv2.destroyAllWindows()


