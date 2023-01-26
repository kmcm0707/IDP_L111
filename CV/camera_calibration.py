import numpy as np
import cv2
import time
import glob


def undistort(img_path, K, D, balance=0.0, dim2=None, dim3=None):    
    img = cv2.imread(img_path)
    dim1 = img.shape[:2][::-1]  #dim1 is the dimension of input image to un-distort    
    DIM = dim1
    assert dim1[0]/dim1[1] == DIM[0]/DIM[1], "Image to undistort needs to have same aspect ratio as the ones used in calibration"    
    if not dim2:
        dim2 = dim1    
    if not dim3:
        dim3 = dim1   

    scaled_K = K * dim1[0] / DIM[0]  # The values of K is to scale with image dimension.
    scaled_K[2][2] = 1.0  # Except that K[2][2] is always 1.0    
    # This is how scaled_K, dim2 and balance are used to determine the final K used to un-distort image. OpenCV document failed to make this clear!
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_K, D, dim2, np.eye(3), balance=balance)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(scaled_K, D, np.eye(3), new_K, dim3, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)    
    cv2.imshow("undistorted", undistorted_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)

# calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_CHECK_COND+cv2.fisheye.CALIB_FIX_SKEW

objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)  *33 # dimention of cube
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

# video = cv2.VideoCapture("http://localhost:8081/stream/video.mjpeg")
# video = cv2.VideoCapture(0)
# images = glob.glob('CV/calib_imgs/mac_webcam/*.jpg')
images = glob.glob('img_dump_manual/*.jpg')
# print("glob working")
# print(images)

for fname in images:
# while True:
    #print(fname)
    # check, img = video.read()
    # print("frame")
    img = cv2.imread(fname)
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

images = glob.glob('img_dump_manual/*.jpg')

"""ret = 3.8486752727448055
mtx =  np.array([[777.86119377,   0. ,        501.7690592 ],
 [  0.   ,      643.66737472 ,378.41289206],
 [  0.       ,    0.   ,        1.        ]])

dist = np.array([[-0.01085549,  0.20750795, -0.03204646,  0.06345628 , 0.82946194]])
rvecs = np.array([np.array([[-0.46971024],
       [-0.37911188],
       [ 1.6037184 ]]), np.array([[-0.42701863],
       [-0.08007203],
       [-2.18732118]]), np.array([[-0.58179822],
       [-0.21304824],
       [ 3.06882188]]), np.array([[0.73288275],
       [1.79273191],
       [1.86721026]])]) 
tvecs = np.array([np.array([[24.00533581],
       [-8.45793763],
       [86.85165589]]), np.array([[-23.12160639],
       [-19.86868153],
       [ 56.95755337]]), np.array([[28.16456112],
       [48.64594752],
       [95.78887674]]), np.array([[23.66790113],
       [40.38851859],
       [80.37845642]])])"""

print(ret, mtx, dist, rvecs, tvecs)

# video = cv2.VideoCapture(0)
# video = cv2.VideoCapture("http://localhost:8081/stream/video.mjpeg")

# while True:
for fname in images:
    img = cv2.imread(fname)
    # check, img = video.read()
    h,  w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    # undistort
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    # crop the image"""
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    cv2.imshow("calib_img", dst)
    key = cv2.waitKey()
    if key == ord("q"):
        break
# cv2.imwrite('calibresult.png', dst)


mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    mean_error += error
print( "total error: {}".format(mean_error/len(objpoints)) )
"""
while True:
    balance = float(input("balnce : "))
    undistort(fname, mtx, dist, 0.8)

    if balance == 0:
        break
"""
cv2.destroyAllWindows()
