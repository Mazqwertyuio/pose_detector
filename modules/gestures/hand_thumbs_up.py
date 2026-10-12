"""
Модуль детекції жесту 'Палець вгору' (Thumbs Up 👍).
Вимагає модель рук: REQUIRES = ["hands"].
"""
from typing import Dict, Any
from core.models import FrameContext, DetectionResult

MODULE_NAME = "hand_thumbs_up"
REQUIRES = ["hands"]

DEFAULT_CONFIG = {
    "min_confidence": 0.85
}


class GestureDetector:
    def __init__(self, config: Dict[str, Any]):
        self.config = {**DEFAULT_CONFIG, **config}

    def process(self, ctx: FrameContext) -> DetectionResult:
        hands_result = ctx.landmarks.get("hands")
        if not hands_result or not hands_result.hand_landmarks:
            return DetectionResult(triggered=False, value=0.0)

        # Перевіряємо кожну знайдену руку
        for hand_lms in hands_result.hand_landmarks:
            # Точки великого пальця
            thumb_tip = hand_lms[4]
            thumb_mcp = hand_lms[2]
            wrist = hand_lms[0]

            # Великий палець направлений вгору відносно зап'ястя та MCP
            thumb_up = (thumb_tip.y < thumb_mcp.y) and (thumb_tip.y < wrist.y)

            # Інші 4 пальці мають бути зігнуті (кінчик нижче суглоба PIP)
            index_folded = hand_lms[8].y > hand_lms[6].y
            middle_folded = hand_lms[12].y > hand_lms[10].y
            ring_folded = hand_lms[16].y > hand_lms[14].y
            pinky_folded = hand_lms[20].y > hand_lms[18].y

            all_other_folded = index_folded and middle_folded and ring_folded and pinky_folded

            if thumb_up and all_other_folded:
                return DetectionResult(
                    triggered=True,
                    value=1.0,
                    confidence=0.95,
                    extra_data={"thumb_y": round(thumb_tip.y, 3)}
                )

        return DetectionResult(triggered=False, value=0.0)
