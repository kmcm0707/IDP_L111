import numpy as np
import cv2
import time
import glob

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)

# calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_CHECK_COND+cv2.fisheye.CALIB_FIX_SKEW

objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

# video = cv2.VideoCapture("http://localhost:8081/stream/video.mjpeg")
video = cv2.VideoCapture(0)
# images = glob.glob('CV/calib_imgs/mac_webcam/*.jpg')
# print("glob working")
# print(images)

# for fname in images:
while True:
    # print(fname)
    check, img = video.read()
    print("frame")
    # img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # cv2.imshow("gray", gray)
    # cv2.waitKey()

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (7, 6), None)
    # If found, add object points, image points (after refining them)

    if ret == True:
        print("corners found")
        # good_one = fname

        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1), criteria)

        imgpoints.append(corners2)
        # Draw and display the corners
        cv2.drawChessboardCorners(img, (7, 6), corners2, ret)
      
    cv2.imshow('img', img)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break

# video.release()

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print(ret, mtx, dist, rvecs, tvecs)

"""img = cv2.imread(good_one)
h,  w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

# undistort
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv2.imshow("calib_img", dst)
cv2.waitKey()
cv2.imwrite('calibresult.png', dst)

mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    mean_error += error
print( "total error: {}".format(mean_error/len(objpoints)) )
"""
cv2.destroyAllWindows()
