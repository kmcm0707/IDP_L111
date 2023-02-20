#!../env/bin/python
# coding: utf-8
# python 3.9.16

"""Code to record points of the board"""
import cv2
import numpy as np
import time
import apriltag
import numba


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

    cv2.imshow("img", img)
    cv2.setMouseCallback("img", click_envent)
    cv2.waitKey()
    cv2.destroyAllWindows()
    points = np.int32(points)
    return points


def click_on_checkpoints(src) -> None:

    video = cv2.VideoCapture(src)
    time.sleep(2)
    ret, frame = video.read()
    if not ret:
        print("error with video feed")
        return -1

    mtx, dist, newcameramtx = load_vals(6)
    frame = cv2.undistort(frame, mtx, dist, None, newcameramtx)

    dim = (810, 810)
    print("click on the corners of the table")
    points = get_points(frame)
    if len(points) != 4:
        print("error with points")
        return -1
    # red table
    # points = np.float32([[261, 682], [818, 630], [236, 151], [738, 92]])

    new_points = np.float32([(0, 0), (0, dim[1]), (dim[0], 0), dim])

    M = cv2.getPerspectiveTransform(points.astype("float32"), new_points)
    frame = cv2.warpPerspective(frame, M, dim)
    checkpoints = get_points(frame)
    print(checkpoints)
    video.release()
    return checkpoints  # ,M


@numba.njit
def chech_if_checkpoint(checkpoints, point, r_sqrd=50):

    np.sum(np.square(checkpoints - point), keepdims=True, axis=1)
    return np.sum(np.square(checkpoints - point), keepdims=True, axis=1) < r_sqrd


def main():
    src = "http://localhost:8081/stream/video.mjpeg"
    src = 1
    # checkpoints, M = click_on_checkpoints(src)

    # mtc, dist, newcameramtx = load_vals(6)
    video = cv2.VideoCapture(src)
    time.sleep(2)

    ret, frame = video.read()
    if not ret:
        print("error with video feed")
        return -1
    start = time.time()
    while time.time() - start < 5:
        ret, frame - video.read()
    checkpoints = get_points(frame)

    mask = np.ones((810, 810), np.uint8)
    for point in checkpoints:
        cv2.circle(mask, point, 10, 0, -1)

    option = apriltag.DetectorOptions(families="tag36h11")
    detector = apriltag.Detector(option)

    while True:
        ret, frame = video.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if not ret:
            print("error with video feed")
            continue
        # frame = cv2.undistort(frame, mtc, dist, None, newcameramtx)
        # frame = cv2.warpPerspective(frame, M, (810, 810))
        result = detector.detect(frame)
        if len(result) > 0:
            ##at_checkpoint, i = chech_if_checkpoint(checkpoints, result[0].center)
            print(result[0].center)
            """if at_checkpoint:
                print(f"at checkpoint {i}")"""

        # frame = cv2.bitwise_and(frame, frame, mask=mask)
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


if __name__ == "__main__":
    click_on_checkpoints("http://localhost:8081/stream/video.mjpeg")
    # main()
