import cv2
import os
from preprocessing import *
def preprocessing():
    dir_name = "images"
    for f in os.listdir(dir_name):
        if f.endswith(".jpg") or f.endswith(".png"):
            img_name = Path(f).stem
            image = cv2.imread(f"{dir_name}/{f}")
            binarized = binarize(image)
            cv2.imwrite(f"binarized/{img_name}.jpg", binarized)
def main():
    if not os.path.exists("binarized"):
        os.makedirs("binarized")
    preprocessing()

if __name__ == "__main__":
    main()