import cv2
import numpy as np
import time
import glob

CHECKERBOARD = (7, 6)

objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)

objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)

flags = cv2.CALIB_CB_ADAPTIVE_THRESH \
            + cv2.CALIB_CB_FAST_CHECK \
            + cv2.CALIB_CB_NORMALIZE_IMAGE

fish_eyed_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC \
                    + cv2.fisheye.CALIB_FIX_SKEW \
                    # + cv2.fisheye.CALIB_CHECK_COND


imgpoints = []
objpoints = []


def process(img, criteria, flags=None):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    """mask = cv2.inRange(gray, 0, 200)
    mask = cv2.bitwise_not(mask)
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    cv2.imshow("mask", mask)
    cv2.waitKey()"""

    # cv2.imshow("gray", gray)
    # cv2.waitKey()

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (7, 6), flags)
    # If found, add object points, image points (after refining them)
    corners2 = None

    if ret == True:
        print("corners found")
        # good_one = fname

        corners2 = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1), criteria)

    return ret, objp, corners2


def fish_eye_calibration(objpoints, imgpoints, shape):
    N_OK = len(objpoints)
    mtx = np.zeros((3, 3))
    dist = np.zeros((4, 1))
    rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
    tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
    rms, _, _, _, _ = \
        cv2.fisheye.calibrate(
            objpoints,
            imgpoints,
            shape[::-1],
            # gray.shape,
            mtx,
            dist,
            rvecs,
            tvecs,
            fish_eyed_flags,
            (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
        )
    print("Found " + str(N_OK) + " valid images for calibration")
    print("DIM=" + str(shape[::-1]))
    print("K=np.array(" + str(K.tolist()) + ")")
    print("D=np.array(" + str(D.tolist()) + ")")

    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(
            objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        mean_error += error
        
    print("total error: {}".format(mean_error/len(objpoints)))


def general_calibration(objpoints, imgpoints, shape):
    ret, mtx, dist, rvecs, tvecs = \
        cv2.calibrateCamera(objpoints, imgpoints, shape[::-1], None, None)

    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(
            objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        mean_error += error

    print("total error: {}".format(mean_error/len(objpoints)))

    print(ret)

    return ret, mtx, dist, rvecs, tvecs


def calib_from_img_dir(dirpath, file_ext=".jpg", silent=False):
    objpoints = []
    imgpoints = []

    images = glob.glob(f"{dirpath}/*{file_ext}")
    for fname in images:
        img = cv2.imread(fname)
        print("frame")
        cv2.imshow("img", img)
        img_dimension = img.shape[:-1]
        ret, each_objpoints, each_imgpoints = process(img, criteria)

        if ret == True:
            print(fname)

            if silent:
                continue

            cv2.drawChessboardCorners(img, (7, 6), each_imgpoints, ret)
            cv2.imshow("image", img)
            key = cv2.waitKey(1)
            if key == ord("q"):
                return
            elif key == ord("y"):
                objpoints.append(each_objpoints)
                imgpoints.append(each_imgpoints)
            else:
                continue

    ret, mtx, dist, rvecs, tvecs = \
        general_calibration(objpoints, imgpoints, img_dimension)


    # if ret == True:
    print("ret true")
    h,  w = img.shape[:2]
    newcameramtx, roi = \
        cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    img = cv2.undistort(img, mtx, dist, None, newcameramtx)
    number = 1
    mtx.tofile(f"tiral_data/mtx_{number}.dat")
    dist.tofile(f"tiral_data/dist_{number}.dat")
    newcameramtx.tofile(f"trial_data/opt_mtx_{number}.dat")

    cv2.imshow("calib_img", img)

    cv2.waitKey()


if __name__ == "__main__":
    calib_from_img_dir("img_dump_manual_table3_2")
