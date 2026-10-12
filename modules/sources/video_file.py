"""
Модуль читання відеопотоку з файлу (.mp4, .avi, .mkv тощо).
"""
import os
import cv2
from typing import Dict, Any, Tuple
import numpy as np

MODULE_NAME = "video_file"

DEFAULT_CONFIG = {
    "filepath": "",
    "loop": True
}


class VideoSource:
    def __init__(self, config: Dict[str, Any]):
        self.config = {**DEFAULT_CONFIG, **config}
        self.filepath = self.config.get("filepath", "")
        self.loop = bool(self.config.get("loop", True))

        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"Відеофайл не знайдено: {self.filepath}")

        self.cap = cv2.VideoCapture(self.filepath)
        if not self.cap.isOpened():
            raise RuntimeError(f"Не вдалося відкрити відеофайл: {self.filepath}")

    def read(self) -> Tuple[bool, np.ndarray]:
        ret, frame = self.cap.read()
        if not ret and self.loop:
            # Перемотування на початок при зацикленні
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
        return ret, frame

    def release(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()
