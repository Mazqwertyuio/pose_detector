"""
Консольна утиліта експорту автономного коду керування (Жести, Обличчя, Голос).
Приклади використання:
    python export_code.py --modality face --out run_face_id.py
    python export_code.py --modality voice --mode yamnet --out run_sounds.py
    python export_code.py --modality voice --mode voice_id --out run_voice_id.py
    python export_code.py --modality gesture --gesture modules/gestures/recorded/my_gesture.json --out run_gesture.py
"""
import argparse
import os
import sys

from core.code_exporter import (
    create_export_package,
    create_robot_guard_package,
    export_face_id_standalone,
    export_gesture_standalone,
    export_multimodal_robot_guard,
    export_voice_standalone,
)


def main():
    ap = argparse.ArgumentParser(description="Експорт автономного Python коду та моделей з V-Pose Combine Studio")
    ap.add_argument("--modality", choices=["gesture", "face", "voice", "robot", "package"], required=True,
                    help="Модальність: gesture (жести), face (обличчя), voice (голос), robot (робот 2FA+жести)")
    ap.add_argument("--package", action="store_true",
                    help="Експортувати у вигляді повного ZIP-пакета (з ONNX моделями, профілями та requirements.txt)")
    ap.add_argument("--gesture", type=str, default=None,
                    help="Шлях до .json файлу жесту (для modality=gesture)")
    ap.add_argument("--mode", choices=["yamnet", "voice_id"], default="yamnet",
                    help="Режим для voice: yamnet (звуки оточення) або voice_id (диктор)")
    ap.add_argument("--out", type=str, default=None,
                    help="Куди зберегти згенерований файл або папку")
    args = ap.parse_args()

    modality = "robot" if args.modality == "package" else args.modality
    is_package = args.package or (args.modality == "package")

    if is_package:
        print(f">>> Створення автономного пакета для модальності: «{modality}»...")
        res = create_export_package(
            modality=modality,
            gesture_path=args.gesture,
            voice_mode=args.mode,
            output_dir=args.out,
            zip_package=True
        )
        print("\n[УСПІШНО] Автономний пакет збережено:")
        print(f"  [DIR]    Папка проєкту:  {res['package_dir']}")
        print(f"  [ZIP]    ZIP-архів:      {res['zip_path']}")
        print(f"  [MODELS] Скопійовано моделей: {len(res['models_copied'])}")
        print(f"  [RUN]    Головний скрипт: {res['run_file']}\n")
        return

    code = ""
    if args.modality == "gesture":
        if not args.gesture:
            print("ПОМИЛКА: Для --modality gesture вкажіть шлях до файлу жесту через --gesture <path.json>")
            sys.exit(1)
        if not os.path.exists(args.gesture):
            print(f"ПОМИЛКА: Файл жесту не знайдено: {args.gesture}")
            sys.exit(1)
        code = export_gesture_standalone(args.gesture)
    elif args.modality == "face":
        code = export_face_id_standalone()
    elif args.modality == "voice":
        code = export_voice_standalone(mode=args.mode)
    elif args.modality == "robot":
        code = export_multimodal_robot_guard()

    if args.out:
        out_path = os.path.abspath(args.out)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(code)
        print(f"УСПІШНО! Автономний код збережено у файл: {out_path}")
        print(f"Запуск: python \"{out_path}\"")
    else:
        print(code)


if __name__ == "__main__":
    main()
