"""
Модуль детекції підморгування лівим оком (Wink Left).
"""
import math
from typing import Dict, Any
from core.models import FrameContext, DetectionResult

MODULE_NAME = "wink_left"
REQUIRES = ["face"]

DEFAULT_CONFIG = {
    # Порогове значення заплющення лівого ока (від 0.0 до 1.0)
    "blink_threshold": 0.45,
    # Максимально допустиме заплющення правого ока (щоб не плутати зі звичайним кліпанням обома очима)
    "opposite_eye_max": 0.30
}


class GestureDetector:
    def __init__(self, config: Dict[str, Any]):
        self.config = {**DEFAULT_CONFIG, **config}

    def process(self, ctx: FrameContext) -> DetectionResult:
        blendshapes = ctx.landmarks.get("face_blendshapes")

        # 1. Якщо доступні нейромережеві Blendshapes (MediaPipe)
        if blendshapes:
            left_blink = blendshapes.get("eyeBlinkLeft", 0.0)
            right_blink = blendshapes.get("eyeBlinkRight", 0.0)

            threshold = self.config["blink_threshold"]
            opposite_max = self.config["opposite_eye_max"]

            # Ліве заплющене, праве залишається відкритим
            is_winking = (left_blink >= threshold) and (right_blink <= opposite_max)

            return DetectionResult(
                triggered=bool(is_winking),
                value=float(round(left_blink, 3)),
                confidence=0.95,
                extra_data={
                    "left_blink": round(left_blink, 3),
                    "right_blink": round(right_blink, 3)
                }
            )

        # 2. Якщо раптом Blendshapes недоступні, розраховуємо за геометричними точками
        face_result = ctx.landmarks.get("face")
        if face_result and face_result.face_landmarks:
            landmarks = face_result.face_landmarks[0]
            # Точки лівого ока
            left_ear = self._calculate_ear(landmarks, [33, 160, 158, 133, 153, 144])
            # Точки правого ока
            right_ear = self._calculate_ear(landmarks, [362, 385, 387, 263, 373, 380])

            # Для EAR менше значення означає заплющене око
            is_winking = (left_ear < 0.20) and (right_ear > 0.25)
            return DetectionResult(
                triggered=is_winking,
                value=1.0 if is_winking else 0.0,
                confidence=0.85,
                extra_data={"left_ear": left_ear, "right_ear": right_ear}
            )

        return DetectionResult(triggered=False, value=0.0)

    def _calculate_ear(self, landmarks, indices) -> float:
        """Розрахунок Eye Aspect Ratio за формулою Сукапа/Терка."""
        def dist(i1, i2):
            p1 = landmarks[i1]
            p2 = landmarks[i2]
            return math.hypot(p1.x - p2.x, p1.y - p2.y)

        # Вертикальні відстані
        v1 = dist(indices[1], indices[5])
        v2 = dist(indices[2], indices[4])
        # Горизонтальна відстань
        h = dist(indices[0], indices[3])
        if h == 0:
            return 0.3
        return (v1 + v2) / (2.0 * h)
