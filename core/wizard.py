"""
Інтерактивний майстер вибору джерела, жестів та дій (Pipeline Wizard).
Дозволяє користувачеві самостійно на льоту зібрати потрібний пайплайн.
"""
import os
import glob
import importlib.util
from typing import Dict, Any, List


def _get_module_info(filepath: str) -> Dict[str, Any]:
    """Витягує назву та вимоги модуля без повного імпорту (.py або .json)."""
    name = os.path.splitext(os.path.basename(filepath))[0]
    requires = []
    desc = ""

    if filepath.endswith(".json"):
        import json
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            name = data.get("name", name)
            model_type = data.get("model_type", "hands")
            return {
                "file": filepath,
                "name": f"{name} [Запис]",
                "requires": [model_type],
                "desc": f"Записаний зразок пози ({model_type})"
            }
        except Exception:
            pass

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if (stripped.startswith('"""') or stripped.startswith("'''")) and not desc:
                    desc = stripped.strip(" \"'\n")
                elif stripped.startswith("REQUIRES") and "=" in stripped:
                    req_part = stripped.split("=", 1)[1].strip()
                    try:
                        requires = eval(req_part)
                    except Exception:
                        pass
                elif stripped.startswith("MODULE_NAME") and "=" in stripped:
                    name_part = stripped.split("=", 1)[1].strip()
                    try:
                        name = eval(name_part)
                    except Exception:
                        pass
    except Exception:
        pass

    return {
        "file": filepath,
        "name": name,
        "requires": requires,
        "desc": desc
    }


def run_interactive_wizard() -> Dict[str, Any]:
    """Запускає інтерактивний діалог у терміналі для вибору джерела та рухів."""
    print("\n" + "=" * 65)
    print("   КОНСТРУКТОР ПАЙПЛАЙНУ: ОБЕРІТЬ ДЖЕРЕЛО ТА РУХИ")
    print("=" * 65)

    # -------------------------------------------------------------
    # КРОК 1: Вибір джерела відео (Source)
    # -------------------------------------------------------------
    print("\n[КРОК 1] Оберіть джерело відеопотоку:")
    print("  [1] Стандартна вебкамера (ID 0)")
    print("  [2] Вказати інший номер вебкамери")
    print("  [3] Локальний відеофайл (mp4, avi, mkv)")

    source_choice = input("Ваш вибір (за замовчуванням 1): ").strip()
    source_cfg = {}

    if source_choice == "2":
        cam_id = input("Введіть номер камери (наприклад 1): ").strip() or "0"
        source_cfg = {
            "file": "modules/sources/webcam.py",
            "config": {"camera_id": int(cam_id)}
        }
    elif source_choice == "3":
        video_path = input("Введіть шлях до відеофайлу: ").strip()
        source_cfg = {
            "file": "modules/sources/video_file.py",
            "config": {"filepath": video_path, "loop": True}
        }
    else:
        source_cfg = {
            "file": "modules/sources/webcam.py",
            "config": {"camera_id": 0}
        }

    # -------------------------------------------------------------
    # КРОК 2: Сканування та вибір рухів/жестів
    # -------------------------------------------------------------
    while True:
        py_gestures = [f for f in glob.glob("modules/gestures/*.py") if not os.path.basename(f).startswith("__")]
        json_gestures = glob.glob("modules/gestures/recorded/*.json")
        gesture_files = sorted(py_gestures) + sorted(json_gestures)

        print("\n[КРОК 2] Доступні рухи/жести у системі:")
        gestures_catalog = []
        for idx, g_file in enumerate(gesture_files, 1):
            info = _get_module_info(g_file)
            gestures_catalog.append(info)
            req_str = f" [Потребує: {', '.join(info['requires'])}]" if info['requires'] else ""
            print(f"  [{idx}] {info['name']:<18} ({os.path.basename(g_file)}){req_str}")

        print("  [R] + ЗАПИСАТИ НОВИЙ РУХ З КАМЕРИ ПРЯМО ЗАРАЗ")
        if json_gestures:
            print("  [D] - ВИДАЛИТИ ЗАПИСАНИЙ РУХ (.json)")

        print("\nВведіть номери рухів (наприклад: 1 або 1, 3), 'R' для запису або 'D' для видалення:")
        g_input = input("Вибір: ").strip()

        if g_input.lower() == 'r':
            from record_pose import record_pose_dialog
            record_pose_dialog()
            continue

        if g_input.lower() == 'd':
            from core.delete_manager import delete_gesture_dialog
            delete_gesture_dialog()
            continue

        if not g_input:
            selected_indices = [0]
        else:
            selected_indices = [int(x.strip()) - 1 for x in g_input.split(",") if x.strip().isdigit()]

        selected_gestures = [gestures_catalog[i] for i in selected_indices if 0 <= i < len(gestures_catalog)]
        if not selected_gestures:
            selected_gestures = [gestures_catalog[0]]
        break

    # -------------------------------------------------------------
    # КРОК 3: Вибір дії для кожного обраного руху
    # -------------------------------------------------------------
    action_files = sorted(glob.glob("modules/actions/*.py"))
    action_files = [f for f in action_files if not os.path.basename(f).startswith("__")]
    actions_catalog = [_get_module_info(af) for af in action_files]

    bindings = []

    print("\n[КРОК 3] Доступні дії для призначення:")
    for a_idx, a_info in enumerate(actions_catalog, 1):
        print(f"  [{a_idx}] {a_info['name']:<16} ({os.path.basename(a_info['file'])})")

    for g_info in selected_gestures:
        print(f"\nПризначення дії для руху '{g_info['name']}':")
        a_choice = input(f"Оберіть номер дії (1-{len(actions_catalog)}, за замовчуванням 1): ").strip()
        a_idx = int(a_choice) - 1 if a_choice.isdigit() and 1 <= int(a_choice) <= len(actions_catalog) else 0
        chosen_action = actions_catalog[a_idx]

        action_config = {}
        if "keyboard" in chosen_action["name"]:
            key = input("Яку клавішу натискати? (space, enter, up, down, w тощо, за замовчуванням space): ").strip()
            action_config["key"] = key if key else "space"

        bindings.append({
            "id": f"{g_info['name']}_trigger",
            "name": f"{g_info['name']} -> {chosen_action['name']}",
            "gesture": {
                "file": g_info["file"],
                "config": {}
            },
            "filter": {
                "cooldown": 0.8,
                "trigger_on": "rising"
            },
            "action": {
                "file": chosen_action["file"],
                "config": action_config
            }
        })

    config = {
        "version": "1.0",
        "source": source_cfg,
        "settings": {
            "preview_window": True,
            "show_landmarks": True,
            "fps_limit": 30
        },
        "bindings": bindings
    }

    print("\n" + "=" * 65)
    print(f"Пайплайн успішно сформовано! Підключено зв'язок: {len(bindings)}")
    print("=" * 65 + "\n")
    return config
