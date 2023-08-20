import os
from pathlib import Path
from .file_util import load_pickle, load_json
import copy
import cv2


def get_video_fps(video_path: str):
    """
    Given the video, return fps
    """
    cam = cv2.VideoCapture(video_path)
    fps = cam.get(cv2.CAP_PROP_FPS)
    cam.release()
    return fps