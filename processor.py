import argparse
import numpy as np
import cv2
import re
import os
from pathlib import Path
from matplotlib import pyplot as plt
from preprocessor import *
from line_extractor import *
from segmentation import *
from utility import *

src = "images"
dst = "processed"

def get_outfile_name(input_path):
    path = Path(input_path)
    parts = list(path.parts)
    parts[parts.index(src)] = dst
    return Path(*parts)


def main():
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="Path to the image to be scanned")
    ap.add_argument("-v", "--verbose", action="store_true", help="Use to save intermediate images")
    args = vars(ap.parse_args())
    original = cv2.imread(args["image"])

    out_file_name = get_outfile_name(args["image"])
    img_name = out_file_name.stem
    folder_type = out_file_name.parent.name

    out_file_name.parent.mkdir(parents=True, exist_ok=True)

    intermediate = None
    if args["verbose"]:
        intermediate = Path(f"{dst}/intermediate/{img_name}")
        intermediate.mkdir(parents=True, exist_ok=True)

    color, binarized = transformation(original, intermediate)

    if (folder_type == "gridded"):
        grid = getStraightLines(binarized)
        vertical, horizontal = getHoughLines(grid, theta=500)
        # showHoughLines(color, vertical, horizontal)
        intersections = findIntersections(vertical, horizontal)

        grid_removed = binarized.copy()
        grid_removed[np.where(grid==255)] = [0]

        for j in range(intersections.shape[1] - 1):
            for i in range(intersections.shape[0] - 1):
                x_s, y_s = intersections[i,j]
                x_e, y_e = intersections[i + 1, j + 1]
                crop_bin = grid_removed[y_s:y_e, x_s:x_e]
                crop_colored = color[y_s:y_e, x_s:x_e]
                segment_digit(crop_bin, crop_colored, i, j)
        # saveHoughLines(original, v, h, str(out_file_name))
    
    else: cv2.imwrite(str(out_file_name), processed)

if __name__ == "__main__":
    main()