import cv2
import numpy as np

frame = cv2.imread("CV/test_imgs/4.png")

# Convert BGR to HSV
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

# define range of blue color in HSV
lower_red = np.array([161, 155, 84]) - (np.array([1, 1, 1]) * 1)
upper_red = np.array([179, 255, 255])

# Threshold the HSV image to get only blue colors
mask = cv2.inRange(hsv, lower_red, upper_red)

# Bitwise-AND mask and original image
res = cv2.bitwise_and(frame, frame, mask=mask)

cv2.imshow('frame', frame)
cv2.imshow('mask', mask)
cv2.imshow('res', res)
cv2.waitKey(0)

cv2.destroyAllWindows()
