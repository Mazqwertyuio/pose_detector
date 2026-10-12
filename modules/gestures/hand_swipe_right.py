"""
Модуль детекції динамічного руху рукою праворуч (Swipe Right).
Аналізує траєкторію переміщення долоні у часі.
"""
from typing import Dict, Any, List, Tuple
from core.models import FrameContext, DetectionResult

MODULE_NAME = "hand_swipe_right"
REQUIRES = ["hands"]

DEFAULT_CONFIG = {
    # Мінімальна дистанція зміщення по горизонталі (у частках екрана, наприклад 0.18 = 18% ширини)
    "min_distance": 0.18,
    # Часове вікно аналізу руху (в секундах)
    "time_window": 0.4,
    # Максимальне відхилення по вертикалі (рух має бути переважно горизонтальним)
    "max_vertical_ratio": 0.6
}


class GestureDetector:
    def __init__(self, config: Dict[str, Any]):
        self.config = {**DEFAULT_CONFIG, **config}
        # Історія позицій: [(timestamp, x, y)]
        self.history: List[Tuple[float, float, float]] = []

    def process(self, ctx: FrameContext) -> DetectionResult:
        hands_res = ctx.landmarks.get("hands")
        if not hands_res or not hands_res.hand_landmarks:
            # Якщо рука зникла з кадру — очищаємо історію
            self.history.clear()
            return DetectionResult(triggered=False, value=0.0)

        # Беремо координати зап'ястя (точка 0) першої руки
        wrist = hands_res.hand_landmarks[0][0]
        now = ctx.timestamp
        cur_x, cur_y = wrist.x, wrist.y

        # Додаємо поточну точку в історію
        self.history.append((now, cur_x, cur_y))

        # Залишаємо лише кадри за останнє часове вікно (наприклад, 0.4 с)
        window = self.config["time_window"]
        self.history = [(t, x, y) for (t, x, y) in self.history if (now - t) <= window]

        if len(self.history) < 4:
            return DetectionResult(triggered=False, value=0.0)

        # Початкова точка у вікні та поточна точка
        start_t, start_x, start_y = self.history[0]
        dx = cur_x - start_x  # > 0 якщо рука змістилася праворуч
        dy = abs(cur_y - start_y)

        min_dist = self.config["min_distance"]
        max_v_ratio = self.config["max_vertical_ratio"]

        # Рух праворуч:
        # 1. dx додатне і більше порогу
        # 2. Горизонтальний рух переважає вертикальний (dy / dx < max_v_ratio)
        is_swipe = (dx >= min_dist) and (dy <= dx * max_v_ratio)

        if is_swipe:
            # Очищаємо історію після фіксації свайпу, щоб не дублювати в наступному кадрі
            self.history.clear()

        return DetectionResult(
            triggered=bool(is_swipe),
            value=float(round(dx / min_dist, 2)) if is_swipe else 0.0,
            confidence=0.92,
            extra_data={"dx": round(dx, 3), "dy": round(dy, 3)}
        )
