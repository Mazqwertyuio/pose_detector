"""
Перевірка (тест) одного жесту наживо.

Запускає камеру з єдиною зв'язкою «жест -> monitor_only» і показує:
  * смугу поточної схожості (0..1),
  * лічильник спрацювань,
  * зелений спалах і напис OK! щойно жест розпізнано.

Використання:
    python test_gesture.py modules/gestures/recorded/my_gesture.json
    python test_gesture.py modules/gestures/hand_swipe_right.py --camera 1
Вихід: ESC або Q у вікні камери.
"""
import sys
import argparse

from core.engine import Engine


def build_config(gesture_file: str, camera_id: int = 0, cooldown: float = 0.4) -> dict:
    return {
        "version": "1.0",
        "settings": {
            "camera_id": camera_id,
            "preview_window": True,
            "show_landmarks": True,
            "test_mode": True,
            "flip_horizontal": True,
        },
        "bindings": [{
            "id": "test_binding",
            "name": "TEST",
            "gesture": {"file": gesture_file, "config": {}},
            "filter": {"cooldown": cooldown, "trigger_on": "rising"},
            "action": {"file": "modules/actions/monitor_only.py", "config": {}},
        }],
    }


def main():
    ap = argparse.ArgumentParser(description="Перевірка жесту наживо")
    ap.add_argument("gesture", help="Шлях до .json зразка або .py модуля жесту")
    ap.add_argument("--camera", type=int, default=0)
    args = ap.parse_args()

    print(f"[Test] Перевірка жесту: {args.gesture}", flush=True)
    try:
        engine = Engine(config_dict=build_config(args.gesture.replace("\\", "/"), args.camera))
        print("[Test] Покажіть жест перед камерою. ESC або Q — вихід.", flush=True)
        engine.run()
    except KeyboardInterrupt:
        pass
    except Exception as ex:
        print(f"[Test] ПОМИЛКА: {ex}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
