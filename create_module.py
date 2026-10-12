"""
Утиліта швидкого створення нових модулів (жестів або дій) за єдиним стандартом.
Використання:
    python create_module.py gesture smile_detector
    python create_module.py action open_browser
"""
import sys
import os

GESTURE_TEMPLATE = '''"""
Модуль детекції руху/жесту: {name}
"""
from typing import Dict, Any
from core.models import FrameContext, DetectionResult

MODULE_NAME = "{name}"
REQUIRES = ["face"]  # Варіанти: ["face"], ["hands"], ["pose"] або комбінація

DEFAULT_CONFIG = {{
    "threshold": 0.5,
}}


class GestureDetector:
    def __init__(self, config: Dict[str, Any]):
        self.config = {{**DEFAULT_CONFIG, **config}}

    def process(self, ctx: FrameContext) -> DetectionResult:
        # Доступні дані залежно від REQUIRES:
        # ctx.landmarks.get("face") -> точки обличчя (FaceLandmarker)
        # ctx.landmarks.get("face_blendshapes") -> словник міміки/очей
        # ctx.landmarks.get("hands") -> точки кистей рук
        # ctx.landmarks.get("pose") -> точки скелета тіла

        # Ваша умова детекції:
        is_triggered = False
        val = 0.0

        return DetectionResult(
            triggered=bool(is_triggered),
            value=float(val),
            confidence=0.90
        )
'''

ACTION_TEMPLATE = '''"""
Модуль виконання дії: {name}
"""
from typing import Dict, Any
from core.models import CommandEvent

MODULE_NAME = "{name}"

DEFAULT_CONFIG = {{
    "param1": "default_value"
}}


class ActionExecutor:
    def __init__(self, config: Dict[str, Any]):
        self.config = {{**DEFAULT_CONFIG, **config}}

    def execute(self, event: CommandEvent) -> bool:
        # Отримуємо параметри та викликаємо дію
        print(f"[ACTION: {name}] Отримано команду від: {{event.gesture_name}} (значення: {{event.value}})")
        return True
'''


def main():
    if len(sys.argv) < 3:
        print("Використання:")
        print("  python create_module.py gesture <ім'я_жесту>")
        print("  python create_module.py action <ім'я_дії>")
        return

    m_type = sys.argv[1].lower().strip()
    m_name = sys.argv[2].lower().strip()

    if m_type == "gesture":
        target_dir = "modules/gestures"
        target_path = os.path.join(target_dir, f"{m_name}.py")
        content = GESTURE_TEMPLATE.format(name=m_name)
    elif m_type == "action":
        target_dir = "modules/actions"
        target_path = os.path.join(target_dir, f"{m_name}.py")
        content = ACTION_TEMPLATE.format(name=m_name)
    else:
        print(f"Невідомий тип модуля: '{m_type}'. Дозволені: 'gesture' або 'action'")
        return

    os.makedirs(target_dir, exist_ok=True)
    if os.path.exists(target_path):
        print(f"Помилка: Файл {target_path} вже існує!")
        return

    with open(target_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"Успішно створено новий модуль: {target_path}")
    print("Тепер ви можете описати в ньому вашу логіку та обрати його при запуску python run.py!")


if __name__ == "__main__":
    main()
