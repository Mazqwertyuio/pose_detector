"""
Модуль звукового сигналу (Beep) на Windows.
Корисно для тестування та калібрування жестів зі звуковим відгуком.
"""
import winsound
from typing import Dict, Any
from core.models import CommandEvent

MODULE_NAME = "sound_beep"

DEFAULT_CONFIG = {
    "frequency": 1000,  # Частота звуку в Гц
    "duration": 150     # Тривалість у мілісекундах
}


class ActionExecutor:
    def __init__(self, config: Dict[str, Any]):
        self.config = {**DEFAULT_CONFIG, **config}

    def execute(self, event: CommandEvent) -> bool:
        freq = event.params.get("frequency", self.config["frequency"])
        dur = event.params.get("duration", self.config["duration"])

        try:
            # Відтворення системного звуку Windows
            winsound.Beep(int(freq), int(dur))
            print(f"[ACTION: sound_beep] Звуковий сигнал ({freq}Hz, {dur}ms) від жесту: {event.gesture_name}")
            return True
        except Exception as e:
            print(f"[ACTION: sound_beep] Помилка відтворення звуку: {e}")
            return False
