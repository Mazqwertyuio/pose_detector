"""
Модуль відображення аналогового повзунка (Slider Display).
Відображає відсоткове значення та графічний бар у консолі.
"""
from typing import Dict, Any
from core.models import CommandEvent

MODULE_NAME = "slider_display"
DEFAULT_CONFIG = {
    "bar_length": 25
}


class ActionExecutor:
    def __init__(self, config: Dict[str, Any]):
        self.config = {**DEFAULT_CONFIG, **config}
        self.bar_len = int(self.config.get("bar_length", 25))

    def execute(self, event: CommandEvent) -> bool:
        val = max(0.0, min(1.0, float(event.value)))
        pct = int(val * 100)
        filled = int(val * self.bar_len)
        bar = "█" * filled + "░" * (self.bar_len - filled)
        print(f"\r[SLIDER] {event.gesture_name}: [{bar}] {pct:>3}%", end="", flush=True)
        return True
