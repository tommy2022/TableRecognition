import numpy as np
import cv2
import re
import os, shutil
import math

from matplotlib import pyplot as plt
from sauvola import sauvola

# **Find the Biggest Contour**
def biggest_contour(contours, min_area):
    max_area = 0
    approx_contour = None
    for n, i in enumerate(contours):
        area = cv2.contourArea(i)
        if area > min_area:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:
                max_area = area
                approx_contour = approx
    if approx_contour is not None:
        approx_contour = approx_contour.reshape((approx_contour.shape[0], -1))
    return approx_contour


# # Transformation the image

# **1. Convert the image to grayscale**

# **2. Remove noise and smoothen out the image by applying blurring and thresholding techniques**

# **3. Use Canny Edge Detection to find the edges**

# **4. Find the biggest contour and crop it out**
def transformation(original_image, intermediate=None):
    binarized = sauvola(original_image)
    image = binarized.copy()
    height, width, channel = original_image.shape
    image_size = binarized.size
    edges = cv2.Canny(image, 100, 200, apertureSize=5)
    contours, hierarchy = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    simplified_contours = []
    for cnt in contours:
        hull = cv2.convexHull(cnt)
        simplified_contours.append(
            cv2.approxPolyDP(hull, 0.02 * cv2.arcLength(hull, True), True)
        )
    simplified_contours = np.array(simplified_contours, dtype=object)

    approx_contour = biggest_contour(simplified_contours, image_size / 4)
    
    if intermediate is not None:
        im_contour = original_image.copy()
        im_simplified = original_image.copy()
        im_approx = original_image.copy()
        im_contour = cv2.drawContours(im_contour, contours, -1, (0, 255, 0), 1)
        im_simplified = cv2.drawContours(im_simplified, simplified_contours, -1, (0, 255, 0), 1)
        cv2.imwrite(str(intermediate/"binarized.png"), binarized)
        cv2.imwrite(str(intermediate/"contours.png"), im_contour)
        cv2.imwrite(str(intermediate/"simplified_contours.png"), im_simplified)
        if approx_contour is not None:
            im_approx = cv2.drawContours(im_approx, [approx_contour], -1, (0, 255, 0), 1)
            for pair in approx_contour:
                im_approx = cv2.circle(im_approx, pair, 10, (0, 0, 255), -1)
            cv2.imwrite(str(intermediate/"approx_contour.png"), im_approx)

    if approx_contour is None:
        return original_image, binarized
    else:
        approx_contour = np.float32(approx_contour)
        return four_point_transform(original_image, binarized, approx_contour)

def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype = "float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect

def is_rectangular(tl, tr, br, bl):
    parr_margin = 0.93969262078 # cosine(+-20 degrees)
    perp_margin = 0.34202014332 # abs(cosine(90 +- 10 degrees))
    def gradient(p1, p2):
        if p1[0] - p2[0] == 0:
            if p1[1] - p2[1] == 0:
                return 0
            else:
                return np.iinfo(np.int32).max
        y_diff = (p1[1] - p2[1])
        x_diff = (p1[0] - p2[0])
        ratio = y_diff / x_diff
        return (p1[1] - p2[1]) / (p1[0] - p2[0])
    
    def cosine_angle(g1, g2):
        # cosine angle similarity: A・B / |A| * |B|
        # A = [1, g1], B = [1, g2]
        return abs(1 + g1 * g2) / math.sqrt((1 + g1**2) * (1 + g2**2))
        
    top = gradient(tl, tr)
    bottom = gradient(bl, br)
    left = gradient(tl, bl)
    right = gradient(tr, br)

    horizontal_cos = cosine_angle(top, bottom)
    vertical_cos = cosine_angle(left, right)
    tl_cos = cosine_angle(top, left)
    br_cos = cosine_angle(bottom, right)

    if (horizontal_cos < parr_margin or vertical_cos < parr_margin
       or tl_cos > perp_margin or br_cos > perp_margin):
        return False
    return True

def four_point_transform(original, binarized, pts):
    # get the 4 edges
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    if not is_rectangular(tl, tr, br, bl):
        return original, binarized

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array(
        [[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]],
        dtype="float32",
    )

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped_org = cv2.warpPerspective(original, M, (maxWidth, maxHeight))
    warped_bin = cv2.warpPerspective(binarized, M, (maxWidth, maxHeight))

    # return the warped image
    return warped_org, warped_bin