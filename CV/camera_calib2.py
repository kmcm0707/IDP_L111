import cv2
import numpy as np
import time
import glob

CHECKERBOARD = (7, 6)

subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)

calibration_flags = (
    cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_FIX_SKEW
)  # +cv2.fisheye.CALIB_CHECK_COND

objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)

objp[0, :, :2] = (
    np.mgrid[0 : CHECKERBOARD[0], 0 : CHECKERBOARD[1]].T.reshape(-1, 2) * 33
)
_img_shape = None

# 3d point in real world space
objpoints = [] 

# 2d points in image plane.
imgpoints = []  

# images = glob.glob('*.jpg')for fname in images:

# video = cv2.VideoCapture("http://localhost:8081/stream/video.mjpeg")
# video = cv2.VideoCapture(0)
images = glob.glob("img_dump_manual/*.jpg")
for fname in images:

    # while True:
    img = cv2.imread(fname)
    print("frame")
    # video.set(cv2.CAP_PROP_FRAME_COUNT, int(t*20))
    # check, img = video.read()
    if _img_shape == None:
        _img_shape = img.shape[:2]
    else:
        assert _img_shape == img.shape[:2], "All images must share the same size."

    gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)

    mask = cv2.inRange(gray, 0, 200)
    mask = cv2.bitwise_not(mask)
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    cv2.imshow("mask", mask)
    cv2.waitKey()
    # continue
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(
        gray,
        CHECKERBOARD,
        cv2.CALIB_CB_ADAPTIVE_THRESH
        + cv2.CALIB_CB_FAST_CHECK
        + cv2.CALIB_CB_NORMALIZE_IMAGE,
    )
    # If found, add object points, image points (after refining them)
    if ret == True:
        print("corner found")
        objpoints.append(objp)
        cv2.cornerSubPix(gray, corners, (3, 3), (-1, -1), subpix_criteria)
        imgpoints.append(corners)

        cv2.drawChessboardCorners(mask, (7, 6), corners, ret)
        # time.sleep(2)

        """cv2.imshow("Calibration sequence", img)
        key = cv2.waitKey()
        if key == ord("q"):
            break
        elif key == ord("w"):
            continue"""

    cv2.imshow("Calibration sequence", mask)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break

# video.release()
# exit()
N_OK = len(objpoints)
K = np.zeros((3, 3))
D = np.zeros((4, 1))
rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
rms, _, _, _, _ = cv2.fisheye.calibrate(
    objpoints,
    imgpoints,
    gray.shape[::-1],
    # gray.shape,
    K,
    D,
    rvecs,
    tvecs,
    calibration_flags,
    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6),
)
print("Found " + str(N_OK) + " valid images for calibration")
print("DIM=" + str(_img_shape[::-1]))
print("K=np.array(" + str(K.tolist()) + ")")
print("D=np.array(" + str(D.tolist()) + ")")

video = cv2.VideoCapture(0)
# video = cv2.VideoCapture("http://localhost:8081/stream/video.mjpeg")
check, img = video.read()
h, w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 1, (w, h))

while True:
    # img = cv2.imread(good_one)
    check, img = video.read()
    if check:
        # undistort
        dst = cv2.undistort(img, K, D, None, newcameramtx)
        # crop the image
        x, y, w, h = roi
        # dst = dst[y:y+h, x:x+w]
        # print(dst)
        cv2.imshow("calib_img", dst)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break
# cv2.imwrite('calibresult.png', dst)

mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, D)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    mean_error += error
print("total error: {}".format(mean_error / len(objpoints)))

cv2.destroyAllWindows()


def check_pattern(img):
    pass
