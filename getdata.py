import numpy as np
import cv2
import os
import sys

# Run using: python getdata.py {label} {startIndex} {endIndex}

# Create location for image directory
cap = cv2.VideoCapture(0)
label = sys.argv[1]
PATH = os.getcwd() + '/data/'
SAVE_PATH = os.path.join(PATH, label)

# Create directory for images
try:
    os.mkdir(SAVE_PATH)
except FileExistsError:
    pass

# Process arguments for number of images
start_index = int(sys.argv[2])
final_index = int(sys.argv[3])+1
print("Hit Space to Capture Image")

# Take and save photos
while True:
    ret, frame = cap.read()
    cv2.imshow('Get Data : ' + label, frame[50:350, 150:450])
    if cv2.waitKey(1) & 0xFF == ord(' '):
        cv2.imwrite(SAVE_PATH + '/' + label +
                    '{}.jpg'.format(start_index), frame[50:350, 150:450])
        print(SAVE_PATH + label + '{}.jpg Captured'.format(start_index))
        start_index += 1
    if start_index >= final_index:
        break

cap.release()
cv2.destroyAllWindows()
