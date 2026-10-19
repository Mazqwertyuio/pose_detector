"""
Утиліта запису нових жестів та рухів перед камерою (Record Pose Studio).
Підтримує всі 6 фундаментальних парадигм жестів:
  1. Статична поза (Static Pose)
  2. Траєкторія у просторі (Spatial Trajectory)
  3. Трансформація форми (Pose Transition: Поза А -> Поза Б)
  4. Циклічний рух / помах (Periodic Wave)
  5. Утримання пози (Hold Pose)
  6. Аналоговий регулятор (Continuous Slider 0%..100%)
"""
import os
import sys
import time
import json
import cv2
import numpy as np
from typing import Optional, Dict, Any

from core.model_manager import ModelManager
from utils.pose_math import (
    extract_landmarks_array,
    normalize_landmarks,
    resample_trajectory
)


def _capture_single_pose(cap, model_mgr, model_type, prompt_title: str) -> Optional[np.ndarray]:
    """Знімає 15 кадрів пози перед камерою та усереднює їх."""
    print(f"\n[Recorder] {prompt_title}")
    print("  -> Займіть позу та натисніть [ПРОБІЛ] для зйомки.")
    print("  -> Натисніть [ESC] або [Q] для скасування.")

    recorded = []
    is_recording = False
    start_t = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        now = time.time()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model_mgr.process(frame_rgb)

        detected_pts = None
        if model_type == "hands":
            hands_res = results.get("hands")
            if hands_res and hands_res.hand_landmarks:
                detected_pts = extract_landmarks_array(hands_res.hand_landmarks[0])
                for lm in hands_res.hand_landmarks[0]:
                    cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 4, (0, 255, 0), -1)

        elif model_type == "face":
            face_res = results.get("face")
            if face_res and face_res.face_landmarks:
                detected_pts = extract_landmarks_array(face_res.face_landmarks[0])
                for idx in [1, 33, 263, 61, 291]:
                    lm = face_res.face_landmarks[0][idx]
                    cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 3, (0, 255, 255), -1)

        elif model_type == "pose":
            pose_res = results.get("pose")
            if pose_res and pose_res.pose_landmarks:
                detected_pts = extract_landmarks_array(pose_res.pose_landmarks[0])
                for lm in pose_res.pose_landmarks[0]:
                    cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 4, (255, 100, 0), -1)

        status_text = f"{prompt_title} - Натисніть SPACE"
        status_color = (0, 200, 0) if detected_pts is not None else (0, 0, 255)

        if is_recording:
            if detected_pts is not None:
                norm = normalize_landmarks(detected_pts, model_type)
                recorded.append(norm)
                status_text = f"ФІКСАЦІЯ... Кадр {len(recorded)}/15"
                status_color = (0, 165, 255)
            if len(recorded) >= 15:
                break

        cv2.rectangle(frame, (0, 0), (w, 50), (20, 20, 20), -1)
        cv2.putText(frame, status_text, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2)
        cv2.imshow("Record Pose Studio", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' ') and not is_recording:
            if detected_pts is not None:
                is_recording = True
            else:
                print(f"[Recorder] Не бачу {model_type}!")
        elif key == ord('q') or key == 27:
            return None

    if len(recorded) >= 15:
        return np.mean(recorded, axis=0)
    return None


def _ask(preset: Dict[str, Any], key: str, prompt: str) -> str:
    """Повертає значення з preset (якщо задано з GUI), інакше питає в консолі."""
    if key in preset and preset[key] not in (None, ""):
        print(f"{prompt}{preset[key]}  [задано з інтерфейсу]")
        return str(preset[key])
    return input(prompt)


def record_pose_dialog(camera_id: int = 0, preset: Optional[Dict[str, Any]] = None) -> str:
    """
    Головний діалог запису жестів для всіх 6 типів.
    preset (необов'язково) дозволяє передати відповіді наперед (використовує GUI):
      name, gesture_type, model_type, threshold, trans_time, axis, min_cycles,
      hold_seconds, slider_mode
    """
    preset = preset or {}
    print("\n" + "=" * 65)
    print("      СТУДІЯ ЗАПИСУ ВЛАСНИХ РУХІВ (6 ПАРАДИГМ ЖЕСТІВ)     ")
    print("=" * 65)

    gesture_name = _ask(preset, "name", "Введіть назву жесту (латиницею, наприклад: fist_to_gun, wave, volume_bar): ").strip()
    if not gesture_name:
        gesture_name = f"gesture_{int(time.time())}"

    print("\nОберіть парадигму (характер) жесту:")
    print("  [1] Статична поза       (фіксація форми: кулак, палець вгору)")
    print("  [2] Траєкторія руху     (переміщення у просторі: свайп, коло)")
    print("  [3] Трансформація       (перетворення форми: Поза А -> Поза Б)")
    print("  [4] Циклічний рух       (коливання туди-назад: помахати рукою)")
    print("  [5] Утримання пози      (тримати позу N секунд)")
    print("  [6] Аналоговий повзунок (рука як регулятор від 0% до 100%)")

    g_choice = _ask(preset, "gesture_choice", "Ваш вибір (1-6, за замовчуванням 1): ").strip()
    g_map = {
        "1": "static",
        "2": "trajectory",
        "3": "transition",
        "4": "periodic",
        "5": "hold",
        "6": "slider"
    }
    gesture_type = g_map.get(g_choice, "static")

    print("\nОберіть частину тіла для відстеження:")
    print("  [1] Кисть / пальці руки (hands)")
    print("  [2] Обличчя / голова    (face)")
    print("  [3] Тіло в повний зріст (pose)")

    type_choice = _ask(preset, "model_choice", "Ваш вибір (1, 2 або 3, за замовчуванням 1): ").strip()
    model_type = "face" if type_choice == "2" else ("pose" if type_choice == "3" else "hands")

    out_dir = os.path.join("modules", "gestures", "recorded")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"{gesture_name}.json")

    print(f"\n[Recorder] Ініціалізація моделі '{model_type}' ({gesture_type})...")
    model_mgr = ModelManager({model_type})

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"[Recorder] ПОМИЛКА: Не вдалося відкрити камеру ID={camera_id}.")
        model_mgr.close()
        return ""

    payload: Dict[str, Any] = {
        "name": gesture_name,
        "model_type": model_type,
        "gesture_type": gesture_type,
        "created_at": time.time()
    }

    try:
        # =================================================================
        # 1. СТАТИЧНА ПОЗА
        # =================================================================
        if gesture_type == "static":
            thresh = float(_ask(preset, "threshold", "\nПоріг схожості (за замовчуванням 0.82): ") or "0.82")
            avg_pose = _capture_single_pose(cap, model_mgr, model_type, "СТАТИЧНА ПОЗА: Замріть")
            if avg_pose is None:
                return ""
            payload["threshold"] = thresh
            payload["tolerance"] = 0.32
            payload["template_points"] = avg_pose.tolist()

        # =================================================================
        # 2. ТРАЄКТОРІЯ У ПРОСТОРІ
        # =================================================================
        elif gesture_type == "trajectory":
            thresh = float(_ask(preset, "threshold", "\nПоріг схожості траєкторії (за замовчуванням 0.78): ") or "0.78")
            traj_time = float(_ask(preset, "traj_time", "Час виконання руху, сек (за замовчуванням 1.0): ") or "1.0")
            traj_time = max(0.3, min(traj_time, 4.0))
            print(f"\n[Recorder] ТРАЄКТОРІЯ: Займіть початкову точку, тисніть SPACE і ведіть рукою {traj_time:g} сек.")
            trajectory_pts = []
            is_rec = False
            rec_start = 0.0
            start_t = time.time()

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]
                now = time.time()
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = model_mgr.process(frame_rgb)

                primary_pt = None
                if model_type == "hands":
                    h_res = results.get("hands")
                    if h_res and h_res.hand_landmarks:
                        primary_pt = (h_res.hand_landmarks[0][0].x, h_res.hand_landmarks[0][0].y)
                elif model_type == "face":
                    f_res = results.get("face")
                    if f_res and f_res.face_landmarks:
                        primary_pt = (f_res.face_landmarks[0][1].x, f_res.face_landmarks[0][1].y)
                elif model_type == "pose":
                    p_res = results.get("pose")
                    if p_res and p_res.pose_landmarks:
                        primary_pt = (p_res.pose_landmarks[0][16].x, p_res.pose_landmarks[0][16].y)

                if len(trajectory_pts) > 1:
                    for i in range(1, len(trajectory_pts)):
                        pt1 = (int(trajectory_pts[i-1][0] * w), int(trajectory_pts[i-1][1] * h))
                        pt2 = (int(trajectory_pts[i][0] * w), int(trajectory_pts[i][1] * h))
                        cv2.line(frame, pt1, pt2, (0, 165, 255), 3)

                status = "Натисніть SPACE для старту руху"
                if is_rec:
                    elapsed = now - rec_start
                    if primary_pt:
                        trajectory_pts.append(primary_pt)
                    status = f"ВЕДІТЬ РУКОЮ! Залишилось: {max(0.0, traj_time - elapsed):.1f}с"
                    if elapsed >= traj_time:
                        break

                cv2.rectangle(frame, (0, 0), (w, 50), (20, 20, 20), -1)
                cv2.putText(frame, status, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 0), 2)
                cv2.imshow("Record Pose Studio", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord(' ') and not is_rec and primary_pt:
                    is_rec = True
                    rec_start = now
                elif key == ord('q') or key == 27:
                    return ""

            resampled = resample_trajectory(trajectory_pts, 16)
            if resampled is None:
                print("[Recorder] Рух був занадто коротким. Спробуйте ще раз.")
                return ""
            payload["threshold"] = thresh
            payload["tolerance"] = 0.30
            payload["time_window"] = traj_time
            payload["traj_time"] = traj_time
            payload["template_points"] = resampled.tolist()

        # =================================================================
        # 3. ТРАНСФОРМАЦІЯ ФОРМИ (ПОЗА А -> ПОЗА Б)
        # =================================================================
        elif gesture_type == "transition":
            thresh = float(_ask(preset, "threshold", "\nПоріг схожості кожної пози (за замовчуванням 0.80): ") or "0.80")
            trans_time = float(_ask(preset, "trans_time", "Максимальний час перетворення (сек, за замовчуванням 0.9): ") or "0.9")

            print("\n>>> КРОК 1: Запис ПОЧАТКОВОЇ пози (наприклад, Кулак) <<<")
            pose_a = _capture_single_pose(cap, model_mgr, model_type, "КРОК 1: Початкова поза (Кулак)")
            if pose_a is None:
                return ""

            time.sleep(0.5)

            print("\n>>> КРОК 2: Запис КІНЦЕВОЇ пози (наприклад, Два пальці вправо) <<<")
            pose_b = _capture_single_pose(cap, model_mgr, model_type, "КРОК 2: Кінцева поза (Пальці)")
            if pose_b is None:
                return ""

            payload["threshold"] = thresh
            payload["tolerance"] = 0.32
            payload["transition_max_time"] = trans_time
            payload["pose_a"] = pose_a.tolist()
            payload["pose_b"] = pose_b.tolist()

        # =================================================================
        # 4. ЦИКЛІЧНИЙ РУХ (КОЛИВАННЯ / ПОМАХ)
        # =================================================================
        elif gesture_type == "periodic":
            print("\nОберіть вісь коливань:")
            print("  [1] Горизонтально (ліворуч-праворуч) — наприклад, помахати рукою")
            print("  [2] Вертикально   (вгору-вниз)       — наприклад, кивок")
            ax_choice = _ask(preset, "axis_choice", "Ваш вибір (1 або 2, за замовчуванням 1): ").strip()
            axis = "y" if ax_choice == "2" else "x"
            min_cycles = int(_ask(preset, "min_cycles", "Кількість змін напрямку (за замовчуванням 3): ") or "3")
            period_window = float(_ask(preset, "period_window", "За який час виконати рух, сек (за замовчуванням 1.0): ") or "1.0")
            amplitude = float(_ask(preset, "min_amplitude", "Мін. розмах руху, частка кадру (за замовчуванням 0.09): ") or "0.09")

            payload["axis"] = axis
            payload["min_cycles"] = min_cycles
            payload["min_amplitude"] = amplitude
            payload["time_window"] = max(0.4, min(period_window, 5.0))

        # =================================================================
        # 5. УТРИМАННЯ ПОЗИ (HOLD)
        # =================================================================
        elif gesture_type == "hold":
            hold_sec = float(_ask(preset, "hold_seconds", "\nСкільки секунд утримувати позу? (за замовчуванням 1.5): ") or "1.5")
            thresh = float(_ask(preset, "threshold", "Поріг схожості пози (за замовчуванням 0.82): ") or "0.82")
            avg_pose = _capture_single_pose(cap, model_mgr, model_type, f"ПОЗА ДЛЯ УТРИМАННЯ ({hold_sec}с)")
            if avg_pose is None:
                return ""
            payload["threshold"] = thresh
            payload["tolerance"] = 0.32
            payload["hold_duration"] = hold_sec
            payload["template_points"] = avg_pose.tolist()

        # =================================================================
        # 6. АНАЛОГОВИЙ КОНТРОЛЕР / ПОВЗУНОК
        # =================================================================
        elif gesture_type == "slider":
            print("\nОберіть режим повзунка:")
            print("  [1] Висота руки по вертикалі (Y вгору/вниз -> 0% .. 100%)")
            print("  [2] Позиція руки по горизонталі (X вліво/вправо -> 0% .. 100%)")
            print("  [3] Щипок (дистанція між великим та вказівним пальцями)")
            s_choice = _ask(preset, "slider_choice", "Ваш вибір (1-3, за замовчуванням 1): ").strip()
            s_map = {"1": "height", "2": "horizontal", "3": "pinch"}
            slider_mode = s_map.get(s_choice, "height")

            payload["slider_mode"] = slider_mode

        # ЗБЕРЕЖЕННЯ КОНФІГУРАЦІЇ
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        print("\n" + "=" * 65)
        print("ЖЕСТ УСПІШНО ЗБЕРЕЖЕНО!")
        print(f"Файл: {out_file}")
        print(f"Парадигма: {gesture_type}")
        print("=" * 65)
        return out_file

    finally:
        cap.release()
        cv2.destroyAllWindows()
        model_mgr.close()

    return ""


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Студія запису жестів")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--preset", type=str, default=None,
                    help="JSON з відповідями наперед (використовує GUI)")
    cli = ap.parse_args()
    record_pose_dialog(cli.camera, json.loads(cli.preset) if cli.preset else None)
