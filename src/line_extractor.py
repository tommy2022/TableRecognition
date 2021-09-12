from sklearn.cluster import MeanShift
import numpy as np
import cv2
from utility import showImage
from matplotlib import pyplot as plt
import math

def getStraightLines(binimage, hv=50, vv=50, iterations=1):
    image = binimage.copy()
    empty = np.zeros_like(image)
    horizontal = morphLine(image, (hv, 1))
    vertical = morphLine(image, (1, vv))
    cv2.drawContours(empty, horizontal, -1, (255,255,255), 2)
    cv2.drawContours(empty, vertical, -1, (255,255,255), 2)
    return empty

def morphLine(image, kernel_size):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    mask = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=1)
    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return cnts[0] if len(cnts) == 2 else cnts[1]


def getHoughLines(grid, theta = 360, thresh = 250):
    # apply canny edge detection
    w, h  = grid.shape
    minLength = min(w, h) / 20
    linesP = np.array(cv2.HoughLinesP(grid, 1, np.pi / theta, thresh, minLineLength=minLength, maxLineGap=100))
    v, h = groupHoughLines(np.squeeze(linesP))
    v, h = calculateLineInfos(v, h)
    v = removeExtraLines(v)
    h = removeExtraLines(h)
    return v, h
    v, h = alignLength(v, h)

    # fix line length
    # calculate each box

def groupHoughLines(linesP):
    dy = linesP[:, 3] - linesP[:, 1]
    dx = linesP[:, 2] - linesP[:, 0]
    angles = np.arctan2(dy, dx)
    coordinates = np.array([np.cos(angles), abs(np.sin(angles))]).T
    coordinates = np.float32(coordinates)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret,label,center=cv2.kmeans(coordinates,2,None,criteria,1,cv2.KMEANS_PP_CENTERS)
    A = linesP[label.ravel()==0]
    B = linesP[label.ravel()==1]
    # Label vertical/horizontal to group
    return (A, B) if abs(A[0,3] - A[0,1]) > abs(B[0,3] - B[0,1]) else (B, A)

def calculateLineInfos(v, h):
    dy = np.array(h[:, 3] - h[:, 1])
    dx = np.array(h[:, 2] - h[:, 0])
    gradient = dy / dx
    intersection = h[:, 1] - gradient * h[:, 0]
    length = np.sqrt((h[:,0] - h[:,2]) ** 2 + (h[:,1] - h[:,3]) ** 2)
    gradient = np.reshape(gradient, (-1, 1))
    intersection = np.reshape(intersection, (-1, 1))
    length = np.reshape(length, (-1, 1))
    h = np.concatenate((gradient, intersection, length, h), axis=1)

    dy = np.array(v[:, 3] - v[:, 1])
    dx = np.array(v[:, 2] - v[:, 0])
    gradient = dx / dy
    intersection = v[:, 0] - gradient * v[:, 1]
    length = np.sqrt((v[:,0] - v[:,2]) ** 2 + (v[:,1] - v[:,3]) ** 2)
    gradient = np.reshape(gradient, (-1, 1))
    intersection = np.reshape(intersection, (-1, 1))
    length = np.reshape(length, (-1, 1))
    v = np.concatenate((gradient, intersection, length, v), axis=1)
    return v, h

def calc_diff_thresh(diff):
    diff_sort = np.sort(diff)
    diff_diff = np.diff(diff_sort)
    max_diff = np.argmax(diff_diff)
    while len(diff) - max_diff < 5:
        diff_diff[max_diff] = 0
        max_diff = np.argmax(diff_diff)
    diff_thresh = diff_sort[max_diff + 1]
    return diff_thresh

def removeExtraLines(lines):
    sorted = lines[lines[:, 1].argsort()]
    intersection = sorted[:, 1]
    diff = np.diff(intersection)
    diff_thresh = calc_diff_thresh(diff)
    cluster_cut = np.argwhere(diff >= diff_thresh).squeeze() + [1]
    groups = np.split(sorted, cluster_cut)
    selected_lines = np.empty((len(groups), len(groups[0][0])))
    for i, lines in enumerate(groups):
        longest = np.argmax(lines.T[2])
        selected_lines[i] = lines[longest]
    return selected_lines

def intersect(v, h):
    # the coordinates of vertical line is flipped so to solve for intersection,
    # We have:
    # x_v = m_v * y_v + c_v
    # y_h = m_h * x_h + c_h
    # solving this we get y = (c_v * c_h) / (c_v - m_h * m_v)
    m_v, c_v, _, _ ,_ ,_ ,_ = v
    m_h, c_h, _, _ ,_ ,_ ,_ = h
    y = (c_v * c_h) / (c_v - m_h * m_v)
    x = m_v * y + c_v
    return int(round(x)), int(round(y))
    

def findIntersections(vertical, horizontal):
    intersections = np.empty((len(vertical), len(horizontal), 2), dtype=np.int32)
    for i, v in enumerate(vertical):
        for j, h in enumerate(horizontal):
            x, y = intersect(v, h)
            intersections[i,j] = [x, y]
    return intersections
            

def drawLines(img, lines):
    if lines is not None:
        for l in lines:
            cv2.line(img, (int(l[3]), int(l[4])), (int(l[5]), int(l[6])), (0,0,255), 2, cv2.LINE_AA)

def showHoughLines(original, v, h):
    img = original.copy()
    drawLines(img, v)
    drawLines(img, h)
    cv2.imshow("img",img)
    cv2.waitKey(0)
