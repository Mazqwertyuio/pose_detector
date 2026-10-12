"""
Функціонал видалення записаних жестів (.json).
"""
import os
import glob
import json


def delete_gesture_dialog() -> bool:
    """
    Інтерактивний діалог видалення збереженого зразка жесту.
    Повертає True, якщо файл було видалено.
    """
    recorded_dir = os.path.join("modules", "gestures", "recorded")
    files = sorted(glob.glob(os.path.join(recorded_dir, "*.json")))

    if not files:
        print("\n[Видалення] У папці recorded/ немає жодного записаного жесту для видалення.")
        return False

    print("\n" + "=" * 65)
    print("           ВИДАЛЕННЯ ЗАПИСАНИХ ЖЕСТІВ (.json)             ")
    print("=" * 65)

    catalog = []
    for idx, fpath in enumerate(files, 1):
        fname = os.path.basename(fpath)
        g_name = os.path.splitext(fname)[0]
        m_type = "unknown"
        g_type = "static"

        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
                g_name = data.get("name", g_name)
                m_type = data.get("model_type", "unknown")
                g_type = data.get("gesture_type", "static")
        except Exception:
            pass

        catalog.append((fpath, g_name))
        type_desc = "траєкторія" if g_type == "trajectory" else "поза"
        print(f"  [{idx}] {g_name:<18} ({fname}) — {m_type}, {type_desc}")

    print("  [0] Скасувати / Назад")

    choice = input("\nОберіть номер жесту для видалення (або 0 для скасування): ").strip()
    if not choice.isdigit() or int(choice) == 0:
        print("Скасовано.")
        return False

    sel_idx = int(choice) - 1
    if 0 <= sel_idx < len(catalog):
        target_path, target_name = catalog[sel_idx]
        confirm = input(f"Ви впевнені, що хочете видалити '{target_name}' ({os.path.basename(target_path)})? (т/н): ").strip().lower()

        if confirm in ["т", "y", "yes", "так"]:
            try:
                os.remove(target_path)
                print(f"[Успішно] Жест '{target_name}' видалено!")
                return True
            except Exception as e:
                print(f"[Помилка видалення]: {e}")
        else:
            print("Видалення скасовано.")
    else:
        print("Невірний номер жесту.")

    return False


if __name__ == "__main__":
    delete_gesture_dialog()
