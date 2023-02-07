import cv2
import numpy as np
import time


def load_vals(num=2):
    mtx = np.fromfile(f"calibration_data/mtx_{num}.dat")
    dist = np.fromfile(f"calibration_data/dist_{num}.dat")
    opt_mtx = np.fromfile(f"calibration_data/opt_mtx_{num}.dat")

    mtx = np.reshape(mtx, (3, 3))
    dist = np.reshape(dist, (1, 5))
    opt_mtx = np.reshape(opt_mtx, (3, 3))

    return mtx, dist, opt_mtx


def get_points(img):
    """function for manual perspective transformation of an image
    returns the transformation matrix"""
    points = []

    def click_envent(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(x, y)
            cv2.circle(img, (x, y), 2, (0, 0, 255), -1)
            points.append((x, y))

        if event == cv2.EVENT_RBUTTONDOWN:
            cv2.destroyAllWindows()
            print(points)

    cv2.imshow("img", img)
    cv2.setMouseCallback("img", click_envent)
    cv2.waitKey()
    points = np.float32(points)
    return points


def click_on_checkpoints(src) -> None:

    mtx, dist, newcameramtx = load_vals(6)

    video = cv2.VideoCapture(src)
    time.sleep(2)
    ret, frame = video.read()
    if not ret:
        print("error with video feed")
        return -1

    frame = cv2.undistort(frame, mtx, dist, None, newcameramtx)

    dim = (810, 810)
    print("click on the corners of the table")
    points = get_points(frame, dim)
    if len(points) != 4:
        print("error with points")
        return -1

    new_points = np.float32([(0, 0), (0, dim[1]), (dim[0], 0), dim])

    M = cv2.getPerspectiveTransform(points, new_points)
    frame = cv2.warpPerspective(frame, M, dim)
    checkpoints = get_points(frame, dim)
    print(checkpoints)
    video.release()
