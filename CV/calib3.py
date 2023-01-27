import cv2
import numpy as np
import time

CHECKERBOARD = (7, 6)

subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)

# calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_CHECK_COND+cv2.fisheye.CALIB_FIX_SKEW

objp = np.zeros((6 * 7, 3), np.float32)
objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2) * 33  # dimention of cube

objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

video = cv2.VideoCapture("http://localhost:8081/stream/video.mjpeg")

while True:
    print("frame")

    check, img = video.read()
    if _img_shape == None:
        _img_shape = img.shape[:2]
    else:
        assert _img_shape == img.shape[:2], "All images must share the same size."

    gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(
        gray,
        CHECKERBOARD,
        cv2.CALIB_CB_ADAPTIVE_THRESH
        + cv2.CALIB_CB_FAST_CHECK
        + cv2.CALIB_CB_NORMALIZE_IMAGE,
    )

    if ret == True:
        print("corner found")
        objpoints.append(objp)
        cv2.cornerSubPix(gray, corners, (3, 3), (-1, -1), subpix_criteria)
        imgpoints.append(corners)

        cv2.drawChessboardCorners(img, (7, 6), corners, ret)
        # time.sleep(2)

        cv2.imshow("Calibration sequence", img)
        key = cv2.waitKey()
        if key == ord("q"):
            break
        elif key == ord("w"):
            continue

    cv2.imshow("Calibration sequence", img)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break

video.release()
