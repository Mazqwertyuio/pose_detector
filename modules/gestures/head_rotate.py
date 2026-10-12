"""
Модуль детекції повороту голови ліворуч або праворуч (Head Turn / Yaw).
"""
from typing import Dict, Any
from core.models import FrameContext, DetectionResult

MODULE_NAME = "head_rotate"
REQUIRES = ["face"]

DEFAULT_CONFIG = {
    "direction": "left",        # "left" (поворот ліворуч) або "right" (поворот праворуч)
    "threshold": 0.16           # Ступінь відхилення від центру (0.0 .. 0.5)
}


class GestureDetector:
    def __init__(self, config: Dict[str, Any]):
        self.config = {**DEFAULT_CONFIG, **config}

    def process(self, ctx: FrameContext) -> DetectionResult:
        face_result = ctx.landmarks.get("face")
        if not face_result or not face_result.face_landmarks:
            return DetectionResult(triggered=False, value=0.0)

        landmarks = face_result.face_landmarks[0]
        # Ніс (1), Ліва щока (234), Права щока (454)
        nose_x = landmarks[1].x
        left_cheek_x = landmarks[234].x
        right_cheek_x = landmarks[454].x

        face_width = right_cheek_x - left_cheek_x
        if abs(face_width) < 1e-4:
            return DetectionResult(triggered=False, value=0.0)

        # Відносне положення носа: 0.5 - дивиться прямо
        center_ratio = (nose_x - left_cheek_x) / face_width
        deviation = center_ratio - 0.5  # <0 якщо повернуто ліворуч, >0 якщо праворуч

        target_dir = self.config.get("direction", "left").lower()
        thresh = self.config.get("threshold", 0.16)

        if target_dir == "left":
            # При дзеркальному відображенні поворот голови користувача ліворуч
            is_active = deviation < -thresh
            val = abs(deviation)
        else:
            is_active = deviation > thresh
            val = abs(deviation)

        return DetectionResult(
            triggered=bool(is_active),
            value=float(round(val, 3)),
            confidence=0.92,
            extra_data={"deviation": round(deviation, 3), "direction": target_dir}
        )
