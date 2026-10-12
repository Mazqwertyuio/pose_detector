"""
Модуль детекції динамічного руху рукою ліворуч (Swipe Left).
Аналізує траєкторію переміщення долоні у часі.
"""
from typing import Dict, Any, List, Tuple
from core.models import FrameContext, DetectionResult

MODULE_NAME = "hand_swipe_left"
REQUIRES = ["hands"]

DEFAULT_CONFIG = {
    "min_distance": 0.18,
    "time_window": 0.4,
    "max_vertical_ratio": 0.6
}


class GestureDetector:
    def __init__(self, config: Dict[str, Any]):
        self.config = {**DEFAULT_CONFIG, **config}
        self.history: List[Tuple[float, float, float]] = []

    def process(self, ctx: FrameContext) -> DetectionResult:
        hands_res = ctx.landmarks.get("hands")
        if not hands_res or not hands_res.hand_landmarks:
            self.history.clear()
            return DetectionResult(triggered=False, value=0.0)

        wrist = hands_res.hand_landmarks[0][0]
        now = ctx.timestamp
        cur_x, cur_y = wrist.x, wrist.y

        self.history.append((now, cur_x, cur_y))

        window = self.config["time_window"]
        self.history = [(t, x, y) for (t, x, y) in self.history if (now - t) <= window]

        if len(self.history) < 4:
            return DetectionResult(triggered=False, value=0.0)

        start_t, start_x, start_y = self.history[0]
        dx = cur_x - start_x  # < 0 якщо рука змістилася ліворуч
        dy = abs(cur_y - start_y)

        min_dist = self.config["min_distance"]
        max_v_ratio = self.config["max_vertical_ratio"]

        # Рух ліворуч:
        is_swipe = (dx <= -min_dist) and (dy <= abs(dx) * max_v_ratio)

        if is_swipe:
            self.history.clear()

        return DetectionResult(
            triggered=bool(is_swipe),
            value=float(round(abs(dx) / min_dist, 2)) if is_swipe else 0.0,
            confidence=0.92,
            extra_data={"dx": round(dx, 3), "dy": round(dy, 3)}
        )
