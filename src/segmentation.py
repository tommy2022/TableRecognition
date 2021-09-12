import cv2
import numpy as np
import math

from line_extractor import getStraightLines

def scale_img(img, scale_percent=300):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
  
    return cv2.resize(img, dim, interpolation = cv2.INTER_AREA)


def segment_digit(img, colored, i, j):
    img = scale_img(img)
    colored = scale_img(colored)

    skeleton = img.copy()
    skeleton = cv2.ximgproc.thinning(skeleton, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)

    cnts, _ = cv2.findContours(skeleton, cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE)
    
    for cnt in cnts:
        cnt = cv2.approxPolyDP(cnt, 3, True)
        x,y,w,h = cv2.boundingRect(cnt)
        if h < img.shape[0] / 4 or h * 1.5 < w: continue

        cv2.rectangle(colored,(x,y),(x+w,y+h),(0,255,0),2)
        roi = skeleton[y:y+h, x:x+w]

    cv2.imshow(f"img", img)
    cv2.imshow(f"skeleton", skeleton)
    cv2.imshow(f"segmentation", colored)
    cv2.waitKey(0)