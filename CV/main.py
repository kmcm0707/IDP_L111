import cv2
import time
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

    # TODO: do the processing in particular section of the image

    font = cv2.FONT_HERSHEY_COMPLEX

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define range of blue color in HSV

    lower_red = np.array([120, 70, 80])

    # lower_red = cv2.cvtColor([192, 107, 117])

    upper_red = np.array([180, 255, 255])

    # Threshold the HSV image to get only blue colours
    mask = cv2.inRange(hsv, lower_red, upper_red)

    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # segmented_img = cv2.bitwise_and(frame, frame, mask=mask)
    contours, hierarchy = cv2.findContours(
        mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # seg_output = cv2.drawContours(segmented_img, contours, -1, (0, 255, 0),3)
    # output = cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)
    centres = []
    for cont in contours:
        """if cv2.contourArea(cont) <= 20:
            continue"""
        x, y, w, h = cv2.boundingRect(cont)

        centres.append((x, y, w, h))

    dim = frame.shape
    half = (dim[1]*0.5)
    for x, y, w, h in centres:
        if (x < half) and (x > 180):
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0))
            # cv2.putText(frame, f"{x}, {y}, {cv2.contourArea(cont)}", (x, y), font, 0.5, (255, 0, 0))

            # print(f"cubes at : {x, y}")

    # print()

    # cv2.imshow('output', output)

    # Bitwise-AND mask and original image
    # res = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow('frame', frame)
    # cv2.imshow('mask', mask)
    # cv2.imshow('res', res)
    cv2.waitKey(0)

    cv2.destroyAllWindows()


def detect_red_video(video):

    while True:
        check, frame = video.read()
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_red = np.array([120, 70, 80])

        # lower_red = cv2.cvtColor([192, 107, 117])

        upper_red = np.array([180, 255, 255])

        # Threshold the HSV image to get only blue colours
        mask = cv2.inRange(hsv, lower_red, upper_red)

        kernel = np.ones((7, 7), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, hierarchy = cv2.findContours(
            mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        centres = []
        for cont in contours:
            """if cv2.contourArea(cont) <= 20:
                continue"""
            x, y, w, h = cv2.boundingRect(cont)

            centres.append((x, y, w, h))

        dim = frame.shape
        half = (dim[1]*0.5)
        for x, y, w, h in centres:
            if (x < half) and (x > 180):
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0))

        cv2.imshow('frame', frame)

        key = cv2.waitKey(2)
        if key == ord('q'):
            return





if __name__ == "__main__":

    """# For lines
    for i in range(1, 9):
        img = cv2.imread(f"../CV/test_imgs/{i}.png")
        detect_line(img)
"""
    """# For blocks
    for i in range(1, 9):
        img = cv2.imread(f"../CV/test_imgs/{i}.png")
        detect_red(img)"""

    # this tries to apply this object detection with camera

    # for mjpeg stream replace the 0 with the url after connecting through ssh
    video = cv2.VideoCapture(0)

    detect_red_video(video)

    video.release()

    cv2.destroyAllWindows()
