#!../env/bin/python
# coding: utf-8
# python 3.9.16

"""This file contains functions that are used to calibrate the camera"""

try:
    import cv2
    import os
    import numpy as np
except ImportError:
    print("dependencies not installed")
    print("run 'pip install -r requirements.txt' to install dependencies")
    exit()

import glob
import shutil

# initialising the parameter for calibration
CHECKERBOARD = (7, 6)
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0, :, :2] = np.mgrid[0 : CHECKERBOARD[0], 0 : CHECKERBOARD[1]].T.reshape(-1, 2)

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# flags for chessboard detection
flags = (
    cv2.CALIB_CB_ADAPTIVE_THRESH
    + cv2.CALIB_CB_FAST_CHECK
    + cv2.CALIB_CB_NORMALIZE_IMAGE
)

# flags for fisheye calibration
fish_eyed_flags = (
    cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_FIX_SKEW
)  # + cv2.fisheye.CALIB_CHECK_COND

# coordinate on the image
imgpoints = []

# coordinate on the 3d space
objpoints = []


def process(img, criteria, flags=None):
    """finds corners of chessboard in image, returns the object points and
    images points along with if the corners were found or not
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (7, 6), flags)
    # If found, add object points, image points (after refining them)
    corners2 = None

    if ret == True:  # noqa: E712
        print("corners found")
        # good_one = fname

        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    return ret, objp, corners2


def fish_eye_calibration(objpoints, imgpoints, shape):
    "not curretly in use, fixes distortion for fisheyed lens"

    # initialising values used for calibration
    N_OK = len(objpoints)
    mtx = np.zeros((3, 3))
    dist = np.zeros((4, 1))
    rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
    tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]

    rms, _, _, _, _ = cv2.fisheye.calibrate(
        objpoints,
        imgpoints,
        shape[::-1],
        # gray.shape,
        mtx,
        dist,
        rvecs,
        tvecs,
        fish_eyed_flags,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6),
    )

    print("Found " + str(N_OK) + " valid images for calibration")
    print("DIM=" + str(shape[::-1]))
    print("K=np.array(" + str(mtx.tolist()) + ")")
    print("D=np.array(" + str(dist.tolist()) + ")")

    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error

    print("total error: {}".format(mean_error / len(objpoints)))


def general_calibration(objpoints, imgpoints, shape):
    """returns calibration parameters with given position of points on image
    and in 3d space, can be used for any type of distortion
    """

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, shape[::-1], None, None
    )

    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error

    print("total error: {}".format(mean_error / len(objpoints)))

    print(ret)

    return ret, mtx, dist, rvecs, tvecs


def calib_from_img_dir(
    dirpaths: list or str, file_ext=".jpg", silent=False, number=100
) -> None:
    """calculates calibration parametres (mtr, dist, rvecs, tvecs, optimised_mtx)
    from a directory of images while sorting good ones manually
    """
    # if only one directory is given, convert it to a list
    if isinstance(dirpaths, str):
        dirpaths = [dirpaths]

    objpoints = []
    imgpoints = []

    # going thorught all the images in the directory
    for dirpath in dirpaths:
        images = glob.glob(f"{dirpath}/*{file_ext}")
        for fname in images:
            img = cv2.imread(fname)
            print("frame")
            # cv2.imshow("img", img)
            img_dimension = img.shape[:-1]

            # cheching if the image can be used for calibration
            ret, each_objpoints, each_imgpoints = process(img, criteria, flags=flags)

            if ret == True:
                print(fname)

                if silent:
                    continue

                # manual sorting of good images
                cv2.drawChessboardCorners(img, (7, 6), each_imgpoints, ret)
                cv2.imshow("image", img)
                key = cv2.waitKey()
                if key == ord("q"):
                    return
                elif key == ord("y"):
                    # y to save, any other to skip
                    objpoints.append(each_objpoints)
                    imgpoints.append(each_imgpoints)
                else:
                    continue

    # calculating calibration parameters using the imgaes selected
    ret, mtx, dist, rvecs, tvecs = general_calibration(
        objpoints, imgpoints, img_dimension
    )

    # if ret == True:
    print("ret true")
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    img = cv2.undistort(img, mtx, dist, None, newcameramtx)

    # saving the calibration parameters
    mtx.tofile(f"calibration_data/mtx_{number}.dat")
    dist.tofile(f"calibration_data/dist_{number}.dat")
    newcameramtx.tofile(f"calibration_data/opt_mtx_{number}.dat")

    # preview of the undistorted image
    cv2.imshow("calib_img", img)

    cv2.waitKey()


def sort_img(path_to_imgs, file_ext=".jpg", silent=False):
    "for sorting good images and bad ones from calibration images"

    # going thorugh images
    images = glob.glob(f"{path_to_imgs}/*{file_ext}")
    for fname in images:
        img = cv2.imread(fname)
        print("frame")
        cv2.imshow("img", img)
        img_dimension = img.shape[:-1]

        # cheching if the image can be used for calibration
        ret, each_objpoints, each_imgpoints = process(img, criteria, flags=flags)

        if ret == True:
            print(fname)

            if silent:
                continue

            # manual sorting of good images
            cv2.drawChessboardCorners(img, (7, 6), each_imgpoints, ret)
            cv2.imshow("image", img)
            key = cv2.waitKey()
            if key == ord("q"):
                return
            elif key == ord("y"):
                # y to save, any other to skip
                shutil.copy(fname, "successful_imgs")
            else:
                continue


def test_cal_val(number=6, alpha=1):
    """function for testing privious calibration values

    Parameters
    ----------
    number : int, optional
        number of calibration data, by default 6
        6 is the best one

    alpha : float, optional
        alpha value for undistort function, by default 1
        when 0 shows no void in undistoreted img
    """
    # loaing calibration data
    mtx = np.fromfile(f"calibration_data/mtx_{number}.dat")
    dist = np.fromfile(f"calibration_data/dist_{number}.dat")
    opt_mtx = np.fromfile(f"calibration_data/opt_mtx_{number}.dat")

    mtx = np.reshape(mtx, (3, 3))
    dist = np.reshape(dist, (1, 5))
    opt_mtx = np.reshape(opt_mtx, (3, 3))

    img = cv2.imread("calib_imgs/img_dump_manual_table3_2/1.jpg")
    cv2.imshow("before", img)

    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), alpha, (w, h))

    img = cv2.undistort(img, mtx, dist, None, newcameramtx)

    # preview of the undistorted image
    cv2.imshow("after", img)
    cv2.waitKey()


def undistorted_live_feed(num=2):
    """gives live feed of undistorted image"""
    video = cv2.VideoCapture("http://localhost:8081/stream/video.mjpeg")

    ret = False
    while not ret:
        ret, img = video.read()

    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    while True:
        check, img = video.read()

        # undistort then show
        img = cv2.undistort(img, mtx, dist, None, newcameramtx)
        cv2.imshow("feed", img)

        key = cv2.waitKey(1)
        if key == ord("q"):
            return


def undistort_frame(img, num=6):
    "function to undistort a still frame/image"
    # loading data
    mtx, dist, new_mtx = load_vals(num)
    h, w = img.shape[:2]
    # newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    # undistorting the image
    img = cv2.undistort(img, mtx, dist, None, new_mtx)
    return img


def perspective_transoformation(img, dim):
    """function for manual perspective transformation of an image
    returns the transformation matrix"""
    points = []
    win_name = "select 4 corners of the tables from bottom to top in left right order"

    def click_envent(event, x, y, flags, params):
        "callback function for mouse click event"
        if event == cv2.EVENT_LBUTTONDOWN:
            print(x, y)
            points.append((x, y))

        if len(points) >= 4:
            cv2.destroyAllWindows()
            print(points)

    cv2.imshow(win_name, img)
    cv2.setMouseCallback(win_name, click_envent)
    cv2.waitKey()

    points = np.float32(points)
    new_points = np.float32([(0, 0), (0, dim[1]), (dim[0], 0), dim])

    M = cv2.getPerspectiveTransform(points, new_points)
    return M


def define_line_manually(img, dim, points_len: int = 8, proportional: bool = False):
    """function for manually defineing the line"""

    points = []

    def click_envent(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(x, y)
            points.append((x, y))

        if len(points) >= points_len:
            cv2.destroyAllWindows()
            print(points)

    cv2.imshow("img", img)
    cv2.setMouseCallback("img", click_envent)
    key = cv2.waitKey()
    points = np.int32(points)
    if proportional:
        points = points / dim[0]

    return points


def load_vals(num=6):
    """function for loading calibration values
    returns the matrix, distortion coefficients and optimal matrix

    Parameters
    ----------
    num : int, optional
        number of the calibration values to load, by default 6
    """
    mtx = np.fromfile(f"calibration_data/mtx_{num}.dat")
    dist = np.fromfile(f"calibration_data/dist_{num}.dat")
    opt_mtx = np.fromfile(f"calibration_data/opt_mtx_{num}.dat")

    mtx = np.reshape(mtx, (3, 3))
    dist = np.reshape(dist, (1, 5))
    opt_mtx = np.reshape(opt_mtx, (3, 3))

    return mtx, dist, opt_mtx


if __name__ == "__main__":
    img = cv2.imread("calib_imgs/img_dump_manual/7.jpg")
    mtx, dist, opt_mtx = load_vals(num=6)  # best one is 6
    dim = (800, 800)

    img = cv2.undistort(img, mtx, dist, None, opt_mtx)
    M = perspective_transoformation(img, dim)
    img = cv2.warpPerspective(img, M, dim)
    points = define_line_manually(img, dim, points_len=12)
    img = cv2.polylines(img, [points], True, (0, 255, 0), 2)
    cv2.imshow("img", img)
    cv2.waitKey()
    cv2.destroyAllWindows()

    # cal val 6 is much better

    """ from distortion calval 6
    [(109, 605), (681, 605), (713, 573), (717, 499), (728, 447), (727, 284), (715, 239), (714, 145), (684, 115), (113, 112), (77, 141), (77, 573)]"""

    """from distortion calval 2
    [(97, 614), (683, 616), (726, 576), (727, 503), (740, 448), (740, 278), (728, 232), (723, 145), (689, 107), (102, 105), (66, 135), (62, 573)]
    """
