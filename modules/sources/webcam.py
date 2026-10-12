"""
Модуль захоплення відео з вебкамери.
"""
import cv2
from typing import Dict, Any, Tuple
import numpy as np

MODULE_NAME = "webcam"

DEFAULT_CONFIG = {
    "camera_id": 0,
    "width": 640,
    "height": 480,
    "flip_horizontal": True
}


class VideoSource:
    def __init__(self, config: Dict[str, Any]):
        self.config = {**DEFAULT_CONFIG, **config}
        self.camera_id = int(self.config.get("camera_id", 0))
        self.flip_h = bool(self.config.get("flip_horizontal", True))

        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Не вдалося відкрити вебкамеру з ID {self.camera_id}")

        if "width" in self.config and "height" in self.config:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config["width"])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config["height"])

    def read(self) -> Tuple[bool, np.ndarray]:
        ret, frame = self.cap.read()
        if ret and self.flip_h:
            frame = cv2.flip(frame, 1)
        return ret, frame

    def release(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()
