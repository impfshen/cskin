import cv2
import numpy as np
from seetaUtil import SampleWapper


def findCircle(white):

    gray = cv2.cvtColor(white, cv2.COLOR_BGR2GRAY)

    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 300, param1=50, param2=30, minRadius=50, maxRadius=150)
    circles = np.uint16(np.around(circles))
    print(len(circles[0, :]))

    return circles[0, :]


def calcAverageLumin(white, circles):

    mask = np.zeros((white.shape[0], white.shape[1])).astype(np.uint8)
    for circle in circles:
        mask = cv2.circle(mask, (circle[0],circle[1]), circle[2], 255, thickness=-1)
        img_circles = cv2.circle(white, (circle[0],circle[1]), circle[2], 255, thickness=2)
    cv2.imwrite("./test/mask.jpg", mask)
    cv2.imwrite("./test/img_circles.jpg", img_circles)

    lab = cv2.cvtColor(white, cv2.COLOR_BGR2LAB)
    lab_l = lab[:, :, 0]

    averge_lumin = cv2.mean(lab_l, mask=mask)

    return int(averge_lumin[0])


def colorIndex(white):
    standard_lumin = 220

    white = white[4200:5200, 1000:2500]  # split color card area

    circles = findCircle(white)

    averge_lumin = calcAverageLumin(white, circles)
    offset_lumin = averge_lumin - standard_lumin
    print("averge_lumin: {}, offset_lumin: {}".format(averge_lumin, offset_lumin))

    return offset_lumin


if __name__ == "__main__":
    white = cv2.imread(r'D:\AIData\cskin\adjust_color\white_standard\1\2.jpg', cv2.IMREAD_COLOR)
    offset_lumin = colorIndex(white)
    print(offset_lumin)