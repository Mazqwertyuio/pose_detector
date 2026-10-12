"""
Модуль 'Тільки моніторинг' (Monitor Only).
Не виконує жодних фізичних дій із системою, лише логує подію.
Ідеально підходить для калібрування, тестування жестів та наочного спостереження.
"""
from typing import Dict, Any
from core.models import CommandEvent

MODULE_NAME = "monitor_only"
DEFAULT_CONFIG = {}


class ActionExecutor:
    def __init__(self, config: Dict[str, Any]):
        self.config = {**DEFAULT_CONFIG, **config}

    def execute(self, event: CommandEvent) -> bool:
        print(f"[MONITOR] Рух зафіксовано: '{event.gesture_name}' (Значення: {event.value:.2f})")
        return True
