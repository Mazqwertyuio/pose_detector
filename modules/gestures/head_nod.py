"""
Модуль детекції кивка головою (Head Nod: рух вгору-вниз).
"""
import time
from typing import Dict, Any, List
from core.models import FrameContext, DetectionResult

MODULE_NAME = "head_nod"
REQUIRES = ["face"]

DEFAULT_CONFIG = {
    # Поріг амплітуди вертикального коливання носа
    "nod_threshold": 0.035,
    # Часове вікно аналізу (в секундах)
    "time_window": 0.6
}


class GestureDetector:
    def __init__(self, config: Dict[str, Any]):
        self.config = {**DEFAULT_CONFIG, **config}
        # Історія позицій: [(timestamp, y_norm)]
        self.history: List[tuple] = []
        self.last_nod_time = 0.0

    def process(self, ctx: FrameContext) -> DetectionResult:
        face_result = ctx.landmarks.get("face")
        if not face_result or not face_result.face_landmarks:
            return DetectionResult(triggered=False, value=0.0)

        landmarks = face_result.face_landmarks[0]
        # Кінчик носа (Landmark 1)
        nose_y = landmarks[1].y
        now = ctx.timestamp

        # Зберігаємо історію
        self.history.append((now, nose_y))
        # Очищаємо застарілі кадри
        window = self.config["time_window"]
        self.history = [(t, y) for (t, y) in self.history if (now - t) <= window]

        if len(self.history) < 5:
            return DetectionResult(triggered=False, value=0.0)

        y_values = [y for (_, y) in self.history]
        min_y = min(y_values)
        max_y = max(y_values)
        amplitude = max_y - min_y

        is_nodding = amplitude >= self.config["nod_threshold"]

        return DetectionResult(
            triggered=is_nodding,
            value=float(round(amplitude / self.config["nod_threshold"], 2)) if is_nodding else 0.0,
            confidence=0.90,
            extra_data={"amplitude": round(amplitude, 4)}
        )
