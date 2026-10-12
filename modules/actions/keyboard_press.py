"""
Модуль емуляції натискання клавіш клавіатури через pynput.
"""
from typing import Dict, Any
from pynput.keyboard import Controller, Key
from core.models import CommandEvent

MODULE_NAME = "keyboard_press"

DEFAULT_CONFIG = {
    "key": "space",       # Назва клавіші (space, enter, up, down, w, a, s, d тощо)
    "mode": "tap"         # "tap" (натиснути і відпустити)
}

# Мапа спеціальних клавіш для pynput
SPECIAL_KEYS = {
    "space": Key.space,
    "enter": Key.enter,
    "tab": Key.tab,
    "esc": Key.esc,
    "backspace": Key.backspace,
    "up": Key.up,
    "down": Key.down,
    "left": Key.left,
    "right": Key.right,
    "media_play_pause": Key.media_play_pause,
    "media_volume_up": Key.media_volume_up,
    "media_volume_down": Key.media_volume_down,
    "media_volume_mute": Key.media_volume_mute,
}


class ActionExecutor:
    def __init__(self, config: Dict[str, Any]):
        self.config = {**DEFAULT_CONFIG, **config}
        self.keyboard = Controller()

    def _resolve_key(self, key_name: str):
        key_name = str(key_name).lower().strip()
        if key_name in SPECIAL_KEYS:
            return SPECIAL_KEYS[key_name]
        return key_name

    def execute(self, event: CommandEvent) -> bool:
        key_str = event.params.get("key", self.config["key"])
        resolved_key = self._resolve_key(key_str)

        try:
            self.keyboard.tap(resolved_key)
            print(f"[ACTION: keyboard_press] Натиснуто клавішу: '{key_str}' (Джерело жесту: {event.gesture_name})")
            return True
        except Exception as e:
            print(f"[ACTION: keyboard_press] Помилка натискання '{key_str}': {e}")
            return False
