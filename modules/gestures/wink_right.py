"""
Модуль детекції підморгування правим оком (Wink Right).
"""
from typing import Dict, Any
from core.models import FrameContext, DetectionResult

MODULE_NAME = "wink_right"
REQUIRES = ["face"]

DEFAULT_CONFIG = {
    "blink_threshold": 0.45,
    "opposite_eye_max": 0.30
}


class GestureDetector:
    def __init__(self, config: Dict[str, Any]):
        self.config = {**DEFAULT_CONFIG, **config}

    def process(self, ctx: FrameContext) -> DetectionResult:
        blendshapes = ctx.landmarks.get("face_blendshapes")

        if blendshapes:
            left_blink = blendshapes.get("eyeBlinkLeft", 0.0)
            right_blink = blendshapes.get("eyeBlinkRight", 0.0)

            threshold = self.config["blink_threshold"]
            opposite_max = self.config["opposite_eye_max"]

            # Праве заплющене, ліве залишається відкритим
            is_winking = (right_blink >= threshold) and (left_blink <= opposite_max)

            return DetectionResult(
                triggered=bool(is_winking),
                value=float(round(right_blink, 3)),
                confidence=0.95,
                extra_data={
                    "left_blink": round(left_blink, 3),
                    "right_blink": round(right_blink, 3)
                }
            )

        return DetectionResult(triggered=False, value=0.0)
