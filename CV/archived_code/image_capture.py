#!../env/bin/python
# coding: utf-8

"""Script to capture images and save them to a folder.

used for collecting images for calibration and other purposes.
"""
import cv2
import numpy as np
import time


def process(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (7, 6), None)
    # If found, add object points, image points (after refining them)

    if ret == True:
        print("corners found")
        # good_one = fname

        # objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        # imgpoints.append(corners2)
        # Draw and display the corners
        cv2.drawChessboardCorners(img, (7, 6), corners2, ret)

    return img


video = cv2.VideoCapture("http://localhost:8081/stream/video.mjpeg")
# video = cv2.VideoCapture(0)

# stored_frame = []
count = 1
while True:

    check, img = video.read()
    key = cv2.waitKey(1)
    if key == ord("q"):
        break
    # if key == ord("w"):
    if (count % 20) == 0:
        print("captured")
        cv2.imwrite(f"calib_imgs/img_dump_manual_table_2/{int(count/20)}.jpg", img)

    count += 1

    cv2.imshow("img", img)


video.release()

"""for img in stored_frame:
    cv2.imshow("img", img)
    cv2.waitKey()"""

cv2.destroyAllWindows()
