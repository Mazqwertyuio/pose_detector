"""
Модуль детекції синхронного кліпання обома очима (Blink Both).
"""
from typing import Dict, Any
from core.models import FrameContext, DetectionResult

MODULE_NAME = "blink_both"
REQUIRES = ["face"]

DEFAULT_CONFIG = {
    "blink_threshold": 0.45
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
            is_both_blinking = (left_blink >= threshold) and (right_blink >= threshold)

            avg_blink = (left_blink + right_blink) / 2.0

            return DetectionResult(
                triggered=bool(is_both_blinking),
                value=float(round(avg_blink, 3)),
                confidence=0.95,
                extra_data={"avg_blink": round(avg_blink, 3)}
            )

        return DetectionResult(triggered=False, value=0.0)
