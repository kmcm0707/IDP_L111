import cv2
import numpy as np

img = cv2.imread("CV/test_imgs/vlcsnap-2023-01-23-11h59m44s016.png")
print(repr(img))

colour_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower_lim = np.array([0, 0, 0])
upper_lim = np.array([1, 1, 1])* 200
line = cv2.inRange(img, lower_lim, upper_lim)

cv2.imshow("img", red)
cv2.waitKey(0)




if __name__ == "__main__":
    pass
