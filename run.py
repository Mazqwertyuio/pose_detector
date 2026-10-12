"""
Головна точка входу для запуску комбайна комп'ютерного зору.
Використання:
    python run.py                # Інтерактивне головне меню
    python run.py --wizard       # Прямий запуск конструктора
    python run.py --config pipeline.yaml  # Запуск готового конфігу
"""
import sys
import argparse
from core.engine import Engine
from core.wizard import run_interactive_wizard
from record_pose import record_pose_dialog


def main():
    parser = argparse.ArgumentParser(
        description="Модульний конвеєр жестів та дій (Pose/Gesture Pipeline)"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Шлях до конфігураційного файлу (наприклад, pipeline.yaml)"
    )
    parser.add_argument(
        "--wizard", "-w",
        action="store_true",
        help="Запустити інтерактивний майстер вибору джерела та рухів"
    )

    parser.add_argument(
        "--gui", "-g",
        action="store_true",
        help="Запустити графічний інтерфейс на Flet (GUI Studio)"
    )

    parser.add_argument(
        "--web",
        action="store_true",
        help="Запустити веб-версію студії у браузері (Web Studio)"
    )

    args = parser.parse_args()

    # Якщо користувач явно вказав прапорець --gui
    if args.gui:
        import flet as ft
        from flet_app import PoseStudioApp
        if hasattr(ft, "run"):
            ft.run(lambda page: PoseStudioApp(page))
        else:
            ft.app(target=lambda page: PoseStudioApp(page))
        return

    # Якщо користувач явно вказав прапорець --web
    if args.web:
        import flet as ft
        from flet_app import PoseStudioApp
        print("\n -> Відкриваємо веб-версію V-Pose Studio на http://localhost:8550 ...")
        if hasattr(ft, "run"):
            ft.run(lambda page: PoseStudioApp(page), host="0.0.0.0", port=8550, view=ft.AppView.WEB_BROWSER)
        else:
            ft.app(target=lambda page: PoseStudioApp(page), host="0.0.0.0", port=8550, view=ft.AppView.WEB_BROWSER)
        return

    print("=" * 65)
    print("   МОДУЛЬНИЙ КОНВЕЄР КОМП'ЮТЕРНОГО ЗОРУ (V-POSE COMBINE)   ")
    print("=" * 65)

    config_dict = None
    config_path = args.config

    # Якщо користувач явно вказав прапорець --wizard
    if args.wizard:
        config_dict = run_interactive_wizard()
    elif not args.config:
        while True:
            print("\nГОЛОВНЕ МЕНЮ:")
            print("  [1] Запустити розпізнавання (обрати джерело та рухи)")
            print("  [2] Записати новий рух / позу з камери (Record Pose)")
            print("  [3] Видалити записаний рух (Delete Gesture)")
            print("  [4] Запустити збережений пайплайн (pipeline.yaml)")
            print("  [5] Запустити графічний інтерфейс Flet (GUI Studio - Десктоп)")
            print("  [6] Запустити веб-версію у браузері (Web Studio - Браузер)")
            print("  [0] Вихід")

            choice = input("\nВаш вибір (0-6, за замовчуванням 1): ").strip()

            if choice == "0":
                print("До побачення!")
                return

            elif choice == "5":
                import flet as ft
                from flet_app import PoseStudioApp
                if hasattr(ft, "run"):
                    ft.run(lambda page: PoseStudioApp(page))
                else:
                    ft.app(target=lambda page: PoseStudioApp(page))
                return

            elif choice == "6":
                import flet as ft
                from flet_app import PoseStudioApp
                print("\n -> Відкриваємо веб-версію V-Pose Studio на http://localhost:8550 ...")
                if hasattr(ft, "run"):
                    ft.run(lambda page: PoseStudioApp(page), host="0.0.0.0", port=8550, view=ft.AppView.WEB_BROWSER)
                else:
                    ft.app(target=lambda page: PoseStudioApp(page), host="0.0.0.0", port=8550, view=ft.AppView.WEB_BROWSER)
                return

            elif choice == "2":
                # Запуск запису пози прямо з меню
                new_pose_file = record_pose_dialog()
                if new_pose_file:
                    ask_run = input("\nБажаєте одразу перейти до налаштування та запуску пайплайну? (т/н, за замовчуванням так): ").strip().lower()
                    if ask_run not in ["н", "n", "no"]:
                        config_dict = run_interactive_wizard()
                        break
                continue

            elif choice == "3":
                from core.delete_manager import delete_gesture_dialog
                delete_gesture_dialog()
                continue

            elif choice == "4":
                config_path = "pipeline.yaml"
                break

            else:
                # Вибір 1 (за замовчуванням): інтерактивний майстер
                config_dict = run_interactive_wizard()
                break

    try:
        if config_dict:
            engine = Engine(config_dict=config_dict)
        else:
            engine = Engine(config_path=config_path)

        engine.run()

    except KeyboardInterrupt:
        print("\n[Runner] Роботу зупинено користувачем (Ctrl+C).")
    except Exception as e:
        print(f"\n[Runner] Виникла помилка: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
