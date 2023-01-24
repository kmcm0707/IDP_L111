import cv2
import numpy as np


def detect_line(frame):
    # print(repr(frame))
    "colour_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)"

    # setting the colour threshold
    #   if in range shows up white and if not black
    lower_lim = np.array([0, 0, 0])
    upper_lim = np.array([1, 1, 1]) * 200

    # black & white processed img
    line = cv2.inRange(frame, lower_lim, upper_lim)

    cv2.imshow("Detect Line", line)
    cv2.waitKey(0)  # press w key to close

    cv2.destroyAllWindows()


def detect_red(frame):

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define range of blue color in HSV

    lower_red = np.array([120, 90, 80])

    # lower_red = cv2.cvtColor([192, 107, 117])

    upper_red = np.array([180, 255, 255])

    # Threshold the HSV image to get only blue colours
    mask = cv2.inRange(hsv, lower_red, upper_red)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame, frame, mask=mask)

    # cv2.imshow('frame', frame)
    cv2.imshow('mask', mask)
    # cv2.imshow('res', res)
    cv2.waitKey(0)

    cv2.destroyAllWindows()


if __name__ == "__main__":

    """# For lines
    for i in range(1, 9):
        img = cv2.imread(f"../CV/test_imgs/{i}.png")
        detect_line(img)
"""
    # For blocks
    for i in range(1, 9):
        img = cv2.imread(f"../CV/test_imgs/{i}.png")
        detect_red(img)
