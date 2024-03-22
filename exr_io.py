import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2


def load_depth(depth_path):
    image = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)[:, :, 0]
    image = image / 100  # cm -> m
    return image