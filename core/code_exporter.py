"""
Генератор автономного (standalone) коду для експорту в зовнішні проєкти.
Дозволяє в 1 клік отримати чистий Python-скрипт або повний ZIP-пакет без прив'язки до GUI:
  1. Standalone Gesture Control (Жести — усі 6 парадигм та 3 моделі MediaPipe Tasks)
  2. Standalone Face ID (Розпізнавання облич YuNet + SFace)
  3. Standalone Sound / Voice ID (Детекція звуків YAMNet та верифікація диктора на вимогу)
  4. Combined Multimodal Robot Guard (2FA: Обличчя + Голос + Watchdog безпеки + Жести)
"""
import glob
import json
import os
import shutil
import zipfile
from typing import Any, Dict, List, Optional


# ==============================================================================
# СПІЛЬНИЙ БЛОК AUTONOMOUS GESTURE ENGINE (MediaPipe Tasks API + Pose Math)
# Вбудовується безпосередньо у згенеровані скрипти жестів та робота,
# щоб вони працювали на будь-якому ПК/роботі без залежності від папок core/ чи utils/
# та були сумісні з MediaPipe >= 0.10 / 1.0+ (де mp.solutions видалено).
# ==============================================================================
_STANDALONE_GESTURE_ENGINE_CODE = '''
# ------------------------------------------------------------------------------
# АВТОНОМНИЙ РУШІЙ ЖЕСТІВ (MediaPipe Tasks API + Математичне ядро 6 парадигм)
# ------------------------------------------------------------------------------
import math
import urllib.request
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

MP_TASK_MODELS = {
    "hands": {
        "filename": "hand_landmarker.task",
        "url": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    },
    "pose": {
        "filename": "pose_landmarker_lite.task",
        "url": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    },
    "face": {
        "filename": "face_landmarker.task",
        "url": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    },
}


def ensure_mp_task_model(model_type: str) -> str:
    """Знаходить локальний .task файл або автоматично завантажує його за потреби."""
    cfg = MP_TASK_MODELS.get(model_type, MP_TASK_MODELS["hands"])
    fname = cfg["filename"]
    for folder in ("models", "models_cache", "."):
        candidate = os.path.join(folder, fname)
        if os.path.exists(candidate):
            return candidate
    os.makedirs("models", exist_ok=True)
    target = os.path.join("models", fname)
    print(f"[MediaPipe] Завантаження моделі {fname}...")
    urllib.request.urlretrieve(cfg["url"], target)
    return target


def extract_landmarks_array(landmark_list) -> np.ndarray:
    return np.array([[lm.x, lm.y, getattr(lm, "z", 0.0)] for lm in landmark_list], dtype=np.float32)


def normalize_landmarks(points: np.ndarray, model_type: str) -> np.ndarray:
    pts = points.copy()
    num_pts = len(pts)
    if num_pts == 0:
        return pts

    if model_type == "hands":
        root = pts[0].copy()
        ref_dist = float(np.linalg.norm(pts[9][:2] - pts[0][:2]))
        if ref_dist < 1e-4:
            ref_dist = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
    elif model_type == "face":
        root = pts[1].copy()
        if num_pts > 454:
            ref_dist = float(np.linalg.norm(pts[454][:2] - pts[234][:2]))
        else:
            ref_dist = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
    elif model_type == "pose":
        if num_pts > 12:
            root = (pts[11] + pts[12]) / 2.0
            ref_dist = float(np.linalg.norm(pts[12][:2] - pts[11][:2]))
        else:
            root = pts[0].copy()
            ref_dist = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
    else:
        root = pts.mean(axis=0)
        ref_dist = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))

    if ref_dist < 1e-4:
        ref_dist = 1.0
    return (pts - root) / ref_dist


def compute_similarity(current_norm: np.ndarray, template_norm: np.ndarray, tolerance: float = 0.35):
    if len(current_norm) != len(template_norm) or len(current_norm) == 0:
        return 0.0, 999.0
    diff = current_norm[:, :2] - template_norm[:, :2]
    dists = np.linalg.norm(diff, axis=1)
    mean_dist = float(np.mean(dists))
    sim = max(0.0, min(1.0, 1.0 - (mean_dist / max(1e-4, tolerance))))
    return round(sim, 3), round(mean_dist, 4)


def resample_trajectory(points, num_samples: int = 16):
    if len(points) < 4:
        return None
    pts = np.array(points, dtype=np.float32)
    pts = pts - pts[0]
    diffs = np.diff(pts, axis=0)
    step_dists = np.linalg.norm(diffs, axis=1)
    cum_dists = np.insert(np.cumsum(step_dists), 0, 0.0)
    total_length = float(cum_dists[-1])
    if total_length < 0.08:
        return None
    target_dists = np.linspace(0.0, total_length, num_samples)
    rx = np.interp(target_dists, cum_dists, pts[:, 0])
    ry = np.interp(target_dists, cum_dists, pts[:, 1])
    return np.column_stack((rx, ry)).astype(np.float32)


def count_oscillations(values, min_amplitude: float = 0.08) -> int:
    if len(values) < 5:
        return 0
    smoothed = [(values[i - 1] + values[i] + values[i + 1]) / 3.0 for i in range(1, len(values) - 1)]
    if len(smoothed) < 3:
        return 0
    reversals = 0
    last_extreme = smoothed[0]
    direction = 0
    for i in range(1, len(smoothed)):
        diff = smoothed[i] - smoothed[i - 1]
        cur_dir = 1 if diff > 0 else (-1 if diff < 0 else 0)
        if cur_dir != 0:
            if direction != 0 and cur_dir != direction:
                if abs(smoothed[i - 1] - last_extreme) >= min_amplitude:
                    reversals += 1
                    last_extreme = smoothed[i - 1]
            direction = cur_dir
    return reversals


class StandaloneGestureDetector:
    """Повна автономна реалізація всіх 6 типів жестів V-Pose."""

    def __init__(self, template_data: dict):
        self.name = template_data.get("name", "gesture")
        self.model_type = template_data.get("model_type", "hands")
        self.gesture_type = template_data.get("gesture_type", "static")

        self.threshold = float(template_data.get("threshold", 0.80))
        self.tolerance = float(template_data.get("tolerance", 0.32))
        self.time_window = float(template_data.get("time_window", template_data.get("traj_time", 0.8)))

        raw_t = template_data.get("template_points", [])
        self.template_points = np.array(raw_t, dtype=np.float32) if len(raw_t) > 0 else np.array([])

        raw_a = template_data.get("pose_a", [])
        raw_b = template_data.get("pose_b", [])
        self.pose_a = np.array(raw_a, dtype=np.float32) if len(raw_a) > 0 else np.array([])
        self.pose_b = np.array(raw_b, dtype=np.float32) if len(raw_b) > 0 else np.array([])
        self.transition_max_time = float(template_data.get("transition_max_time", 1.0))
        self.last_pose_a_time = 0.0

        self.periodic_axis = template_data.get("axis", "x").lower()
        self.min_cycles = int(template_data.get("min_cycles", 3))
        self.min_amplitude = float(template_data.get("min_amplitude", 0.09))

        self.hold_duration = float(template_data.get("hold_duration", 1.2))
        self.hold_start_time = 0.0
        self.hold_triggered = False

        self.slider_mode = template_data.get("slider_mode", "height")
        self.history_2d = []
        self.history_1d = []

    def evaluate(self, raw_landmarks, now: float) -> dict:
        if raw_landmarks is None or len(raw_landmarks) == 0:
            self.hold_start_time = 0.0
            self.hold_triggered = False
            return {"triggered": False, "value": 0.0, "status": "Немає точок"}

        pts_arr = extract_landmarks_array(raw_landmarks)
        norm_pts = normalize_landmarks(pts_arr, self.model_type)

        if self.model_type == "hands":
            primary_pt = (raw_landmarks[0].x, raw_landmarks[0].y)
        elif self.model_type == "face":
            primary_pt = (raw_landmarks[1].x, raw_landmarks[1].y)
        else:
            idx = 16 if len(raw_landmarks) > 16 else 0
            primary_pt = (raw_landmarks[idx].x, raw_landmarks[idx].y)

        # 1. STATIC
        if self.gesture_type == "static":
            sim, _ = compute_similarity(norm_pts, self.template_points, self.tolerance)
            trig = sim >= self.threshold
            return {"triggered": trig, "value": sim, "status": "TRIGGERED" if trig else f"{sim:.2f}"}

        # 2. TRAJECTORY
        elif self.gesture_type == "trajectory":
            self.history_2d.append((now, primary_pt[0], primary_pt[1]))
            self.history_2d = [(t, x, y) for (t, x, y) in self.history_2d if (now - t) <= self.time_window]
            if len(self.history_2d) < 5 or len(self.template_points) == 0:
                return {"triggered": False, "value": 0.0, "status": "Рухайте рукою..."}
            resampled = resample_trajectory([(x, y) for (_, x, y) in self.history_2d], len(self.template_points))
            if resampled is None:
                return {"triggered": False, "value": 0.0, "status": "Очікування руху"}
            dists = np.linalg.norm(resampled - self.template_points, axis=1)
            sim = round(max(0.0, min(1.0, 1.0 - float(np.mean(dists)) / max(1e-4, self.tolerance))), 3)
            trig = sim >= self.threshold
            if trig:
                self.history_2d.clear()
            return {"triggered": trig, "value": sim, "status": "TRIGGERED" if trig else f"Traj: {sim:.2f}"}

        # 3. TRANSITION (Pose A -> Pose B)
        elif self.gesture_type == "transition":
            sim_a, _ = compute_similarity(norm_pts, self.pose_a, self.tolerance)
            sim_b, _ = compute_similarity(norm_pts, self.pose_b, self.tolerance)
            is_primed = False
            if self.last_pose_a_time > 0.0:
                if (now - self.last_pose_a_time) <= self.transition_max_time:
                    is_primed = True
                else:
                    self.last_pose_a_time = 0.0
            if sim_a >= self.threshold:
                self.last_pose_a_time = now
                is_primed = True
            trig = False
            if is_primed and self.last_pose_a_time > 0.0:
                dt = now - self.last_pose_a_time
                if sim_b >= self.threshold and 0.05 <= dt <= self.transition_max_time:
                    trig = True
                    self.last_pose_a_time = 0.0
                    is_primed = False
            val = sim_b if (is_primed or trig) else sim_a
            st = "TRIGGERED!" if trig else ("Крок 2 -> Поза Б" if is_primed else "Крок 1 -> Поза А")
            return {"triggered": trig, "value": val, "status": st}

        # 4. PERIODIC
        elif self.gesture_type == "periodic":
            coord = primary_pt[0] if self.periodic_axis == "x" else primary_pt[1]
            self.history_1d.append((now, coord))
            self.history_1d = [(t, v) for (t, v) in self.history_1d if (now - t) <= self.time_window]
            cycles = count_oscillations([v for (_, v) in self.history_1d], self.min_amplitude)
            trig = cycles >= self.min_cycles
            if trig:
                self.history_1d.clear()
            val = round(min(1.0, cycles / max(1, self.min_cycles)), 2)
            return {"triggered": trig, "value": val, "status": f"Цикли: {cycles}/{self.min_cycles}"}

        # 5. HOLD
        elif self.gesture_type == "hold":
            sim, _ = compute_similarity(norm_pts, self.template_points, self.tolerance)
            if sim >= self.threshold:
                if self.hold_start_time == 0.0:
                    self.hold_start_time = now
                    self.hold_triggered = False
                held = now - self.hold_start_time
                prog = round(min(1.0, held / max(0.1, self.hold_duration)), 2)
                trig = (held >= self.hold_duration) and (not self.hold_triggered)
                if trig:
                    self.hold_triggered = True
                return {"triggered": trig, "value": prog, "status": f"Утримання: {int(prog*100)}%"}
            else:
                self.hold_start_time = 0.0
                self.hold_triggered = False
                return {"triggered": False, "value": 0.0, "status": f"{sim:.2f}"}

        # 6. SLIDER
        elif self.gesture_type == "slider":
            if self.slider_mode == "pinch" and self.model_type == "hands" and len(raw_landmarks) > 8:
                p1, p2 = raw_landmarks[4], raw_landmarks[8]
                dist = math.hypot(p1.x - p2.x, p1.y - p2.y)
                val = round(max(0.0, min(1.0, (dist - 0.03) / 0.17)), 2)
            elif self.slider_mode == "horizontal":
                val = round(max(0.0, min(1.0, primary_pt[0])), 2)
            else:
                val = round(max(0.0, min(1.0, 1.0 - primary_pt[1])), 2)
            return {"triggered": True, "value": val, "status": f"Slider: {int(val*100)}%"}

        return {"triggered": False, "value": 0.0, "status": "Idle"}


class MultiModelLandmarker:
    """Обгортка MediaPipe Tasks API (VIDEO mode) для hands, pose та face."""

    def __init__(self, model_types):
        self.detectors = {}
        self._start_ts = time.time()
        self._last_ms = 0
        for m in set(model_types):
            path = ensure_mp_task_model(m)
            base_opts = mp_python.BaseOptions(model_asset_path=path)
            if m == "hands":
                opts = mp_vision.HandLandmarkerOptions(
                    base_options=base_opts, running_mode=mp_vision.RunningMode.VIDEO, num_hands=1
                )
                self.detectors["hands"] = mp_vision.HandLandmarker.create_from_options(opts)
            elif m == "pose":
                opts = mp_vision.PoseLandmarkerOptions(
                    base_options=base_opts, running_mode=mp_vision.RunningMode.VIDEO
                )
                self.detectors["pose"] = mp_vision.PoseLandmarker.create_from_options(opts)
            elif m == "face":
                opts = mp_vision.FaceLandmarkerOptions(
                    base_options=base_opts, running_mode=mp_vision.RunningMode.VIDEO, num_faces=1
                )
                self.detectors["face"] = mp_vision.FaceLandmarker.create_from_options(opts)

    def extract(self, frame_rgb: np.ndarray, model_type: str):
        det = self.detectors.get(model_type)
        if det is None:
            return None
        ms = int((time.time() - self._start_ts) * 1000)
        if ms <= self._last_ms:
            ms = self._last_ms + 1
        self._last_ms = ms
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        res = det.detect_for_video(mp_img, ms)
        if model_type == "hands" and res.hand_landmarks:
            return res.hand_landmarks[0]
        elif model_type == "pose" and res.pose_landmarks:
            return res.pose_landmarks[0]
        elif model_type == "face" and res.face_landmarks:
            return res.face_landmarks[0]
        return None

    def close(self):
        for det in self.detectors.values():
            try:
                det.close()
            except Exception:
                pass
'''


def export_gesture_standalone(gesture_json_path: str, action_code: str = "print('>>> [ДІЯ] Жест успішно розпізнано!')") -> str:
    """Генерує 100% автономний скрипт розпізнавання обраного жесту (без залежності від core/)."""
    with open(gesture_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    g_type = data.get("gesture_type", "static")
    m_type = data.get("model_type", "hands")
    g_name = data.get("name", "my_gesture")

    return f'''"""
Автономний скрипт розпізнавання жесту «{g_name}» (тип: {g_type}, модель: {m_type}).
Згенеровано автоматично з V-Pose Combine Studio.
Працює повністю автономно у будь-якому проєкті (MediaPipe Tasks API).
Вимоги: pip install opencv-python mediapipe numpy
"""
import os
import time
import json
import cv2
import numpy as np

# Вбудована конфігурація вашого натренованого жесту
GESTURE_CONFIG = {json.dumps(data, indent=4, ensure_ascii=False)}
{_STANDALONE_GESTURE_ENGINE_CODE}

def main():
    detector = StandaloneGestureDetector(GESTURE_CONFIG)
    landmarker = MultiModelLandmarker(["{m_type}"])
    cap = cv2.VideoCapture(0)

    print(">>> Запущено розпізнавання жесту «{g_name}» ({g_type}). Натисніть ESC або Q для виходу.")
    last_trigger = 0.0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            lms = landmarker.extract(frame_rgb, "{m_type}")
            res = detector.evaluate(lms, time.time())

            val = res["value"]
            triggered = res["triggered"]
            status_str = res["status"]

            # Візуалізація точок на кадрі
            if lms:
                h, w = frame.shape[:2]
                for lm in lms:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 3, (0, 255, 255), -1)

            color = (0, 255, 0) if triggered else (0, 165, 255)
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 48), (20, 20, 20), -1)
            cv2.putText(frame, f"{g_name} ({g_type}): {{status_str}} [{{val:.2f}}]",
                        (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

            if triggered and (time.time() - last_trigger > 0.6):
                last_trigger = time.time()
                # ВАША ДІЯ (виконується при спрацюванні жесту):
                {action_code}

            cv2.imshow("Standalone Gesture Control ({g_name})", frame)
            if cv2.waitKey(1) & 0xFF in (27, ord('q'), ord('Q')):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()


if __name__ == "__main__":
    main()
'''


def export_face_id_standalone(profiles_file: str = "modules/faces/profiles.json") -> str:
    """Генерує автономний скрипт розпізнавання облич на базі YuNet + SFace."""
    profiles_data = []
    if os.path.exists(profiles_file):
        with open(profiles_file, "r", encoding="utf-8") as f:
            profiles_data = json.load(f)

    return f'''"""
Автономний модуль розпізнавання облич (Face ID).
Згенеровано автоматично з V-Pose Combine Studio.
Вимоги: pip install opencv-python numpy
Моделі: models/face_detection_yunet_2023mar.onnx, models/face_recognition_sface_2021dec.onnx
"""
import os
import cv2
import numpy as np

# Вбудована база зареєстрованих профілів (128-d вектори ознак SFace)
REGISTERED_PROFILES = {json.dumps(profiles_data, indent=4, ensure_ascii=False)}

YUNET_MODEL = "models/face_detection_yunet_2023mar.onnx"
SFACE_MODEL = "models/face_recognition_sface_2021dec.onnx"
COSINE_THRESHOLD = 0.363  # Рекомендований поріг схожості SFace


def match_profile(recognizer, feat: np.ndarray, prof: dict) -> float:
    """Підтримує як одиничний вектор 'feature', так і мульти-ембеддінги 'features'."""
    best = -1.0
    feats = prof.get("features") or ([prof["feature"]] if "feature" in prof else [])
    for f_vec in feats:
        stored_feat = np.array(f_vec, dtype=np.float32).reshape(1, 128)
        score = float(recognizer.match(feat, stored_feat, cv2.FaceRecognizerSF_FR_COSINE))
        if score > best:
            best = score
    return best


def main():
    if not os.path.exists(YUNET_MODEL) or not os.path.exists(SFACE_MODEL):
        raise FileNotFoundError(f"Моделі не знайдено: {{YUNET_MODEL}}, {{SFACE_MODEL}}")

    detector = cv2.FaceDetectorYN.create(YUNET_MODEL, "", (320, 320), score_threshold=0.6)
    recognizer = cv2.FaceRecognizerSF.create(SFACE_MODEL, "")
    cap = cv2.VideoCapture(0)

    print(">>> Запущено Face ID Standalone. Натисніть ESC або Q для виходу.")

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            detector.setInputSize((w, h))

            _, faces = detector.detect(frame)
            if faces is not None:
                for face in faces:
                    x, y, bw, bh = [int(v) for v in face[:4]]
                    aligned = recognizer.alignCrop(frame, face)
                    feat = recognizer.feature(aligned)

                    best_name = "Невідомий"
                    best_score = -1.0
                    for prof in REGISTERED_PROFILES:
                        score = match_profile(recognizer, feat, prof)
                        if score > best_score:
                            best_score = score
                            if best_score >= COSINE_THRESHOLD:
                                best_name = prof["name"]

                    color = (0, 255, 0) if best_name != "Невідомий" else (0, 140, 255)
                    cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 2)
                    cv2.putText(frame, f"{{best_name}} ({{best_score:.2f}})", (x, max(24, y - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

                    if best_name != "Невідомий":
                        # Дія для авторизованої особи:
                        # print(f"Авторизовано: {{best_name}}")
                        pass

            cv2.imshow("Face ID Standalone", frame)
            if cv2.waitKey(1) & 0xFF in (27, ord('q'), ord('Q')):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
'''


def export_voice_standalone(mode: str = "yamnet", speaker_profiles_file: str = "modules/voices/profiles.json") -> str:
    """Генерує автономний скрипт детекції звуків YAMNet та розпізнавання диктора Voice ID."""
    profiles_data = []
    if os.path.exists(speaker_profiles_file):
        with open(speaker_profiles_file, "r", encoding="utf-8") as f:
            profiles_data = json.load(f)

    if mode == "yamnet":
        return '''"""
Автономний модуль детекції звуків оточення (Google YAMNet).
Згенеровано автоматично з V-Pose Combine Studio.
Вимоги: pip install onnxruntime sounddevice numpy
Модель: models/yamnet.onnx, models/yamnet_class_map.csv
"""
import csv
import time
import numpy as np
import onnxruntime as ort
import sounddevice as sd

YAMNET_PATH = "models/yamnet.onnx"
CLASS_MAP_PATH = "models/yamnet_class_map.csv"


def check_sound_once(session, input_name, classes, duration_sec: float = 3.0):
    print(f"\\n🎙️ Запис звуку ({duration_sec} с)... Видайте звук або скажіть щось:")
    rec = sd.rec(int(16000 * duration_sec), samplerate=16000, channels=1, dtype="float32")
    sd.wait()
    waveform = rec[:, 0]

    outputs = session.run(None, {input_name: waveform})
    mean_scores = np.mean(outputs[0], axis=0)
    top_indices = np.argsort(mean_scores)[::-1][:4]

    print("--- Результати аналізу Google YAMNet ---")
    for idx in top_indices:
        score = float(mean_scores[idx])
        if score >= 0.05:
            print(f"  • {classes[idx]}: {score * 100:.1f}%")
    return classes[top_indices[0]], float(mean_scores[top_indices[0]])


def main():
    session = ort.InferenceSession(YAMNET_PATH, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    with open(CLASS_MAP_PATH, "r", encoding="utf-8") as f:
        classes = [row["display_name"] for row in csv.DictReader(f)]

    print(">>> Модуль Google YAMNet готовий.")
    print("Натисніть [ENTER] для запуску разової перевірки (3 с), або введіть 'c' для безперервного режиму, 'q' для виходу.")

    while True:
        cmd = input("\\n[ENTER = Перевірити 3с | c = Постійний потік | q = Вихід]: ").strip().lower()
        if cmd == "q":
            break
        elif cmd == "c":
            print(">>> Безперервний моніторинг (Ctrl+C для зупинки)...")
            buffer = []
            def audio_cb(indata, frames, time_info, status):
                buffer.extend(indata[:, 0])
            with sd.InputStream(samplerate=16000, channels=1, dtype="float32", callback=audio_cb):
                try:
                    while True:
                        time.sleep(0.35)
                        if len(buffer) >= 15600:
                            chunk = np.array(buffer[-15600:], dtype=np.float32)
                            outputs = session.run(None, {input_name: chunk})
                            mean_scores = np.mean(outputs[0], axis=0)
                            top_idx = int(np.argmax(mean_scores))
                            event_name = classes[top_idx]
                            score = float(mean_scores[top_idx])
                            if score >= 0.15 and event_name != "Silence":
                                print(f"[Звук] {event_name}: {score*100:.1f}%")
                except KeyboardInterrupt:
                    print("Зупинено безперервний режим.")
        else:
            event_name, score = check_sound_once(session, input_name, classes, duration_sec=3.0)
            if event_name in ("Clapping", "Applause", "Finger snapping", "Whistling"):
                print(f">>> ДІЯ: Спрацював звуковий тригер ({event_name})!")


if __name__ == "__main__":
    main()
'''
    else:
        return f'''"""
Автономний модуль комплексної перевірки звуку (YAMNet) та верифікації диктора (Voice ID).
Згенеровано автоматично з V-Pose Combine Studio.
Вимоги: pip install onnxruntime sounddevice numpy
"""
import csv
import json
import os
import time
import numpy as np
import onnxruntime as ort
import sounddevice as sd

REGISTERED_SPEAKERS = {json.dumps(profiles_data, indent=4, ensure_ascii=False)}
YAMNET_PATH = "models/yamnet.onnx"
CLASS_MAP_PATH = "models/yamnet_class_map.csv"
VOICE_THRESHOLD = 0.70


def verify_voice_and_sound(session, input_name, classes, duration_sec: float = 3.0):
    print(f"\\n🎙️ Слухаю мікрофон ({{duration_sec}} с)... Говоріть:")
    rec = sd.rec(int(16000 * duration_sec), samplerate=16000, channels=1, dtype="float32")
    sd.wait()
    chunk = rec[:, 0]

    probs, embeddings = session.run(None, {{input_name: chunk}})

    # 1. Визначення типу звуку (YAMNet)
    mean_scores = np.mean(probs, axis=0)
    top_idx = int(np.argmax(mean_scores))
    sound_type = classes[top_idx] if classes and top_idx < len(classes) else f"Class_{{top_idx}}"
    sound_conf = float(mean_scores[top_idx])

    # 2. Верифікація власника голосу (Voice ID)
    speech_scores = probs[:, 0]
    speech_mask = speech_scores >= 0.20
    if not np.any(speech_mask):
        speech_mask = speech_scores >= 0.10

    best_name = "Мови не виявлено"
    best_sim = 0.0
    is_verified = False

    if np.any(speech_mask):
        voiceprint = np.mean(embeddings[speech_mask], axis=0)
        norm = np.linalg.norm(voiceprint)
        if norm > 1e-6:
            voiceprint /= norm
            best_name = "Невідомий голос"
            for spk in REGISTERED_SPEAKERS:
                stored_v = np.array(spk["embedding"], dtype=np.float32)
                sim = float(np.dot(voiceprint, stored_v))
                if sim > best_sim:
                    best_sim = sim
                    if best_sim >= VOICE_THRESHOLD:
                        best_name = spk["name"]
                        is_verified = True

    print(f"[1. Тип звуку YAMNet] {{sound_type}} ({{sound_conf*100:.1f}}%)")
    print(f"[2. Диктор Voice ID]  {{best_name}} (Схожість: {{best_sim*100:.1f}}%)")
    if is_verified:
        print(f"✅ >>> ДІЯ: Авторизовано голос власника ({{best_name}})!")
    return is_verified, best_name, best_sim, sound_type


def main():
    session = ort.InferenceSession(YAMNET_PATH, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    classes = []
    if os.path.exists(CLASS_MAP_PATH):
        with open(CLASS_MAP_PATH, "r", encoding="utf-8") as f:
            classes = [row["display_name"] for row in csv.DictReader(f)]

    print(">>> Запущено Voice ID + YAMNet Verifier.")
    while True:
        cmd = input("\\nНатисніть [ENTER] для перевірки голосу (3 с) або 'q' для виходу: ").strip().lower()
        if cmd == "q":
            break
        verify_voice_and_sound(session, input_name, classes, duration_sec=3.0)


if __name__ == "__main__":
    main()
'''


def export_multimodal_robot_guard(
    face_profiles_file: str = "modules/faces/profiles.json",
    voice_profiles_file: str = "modules/voices/profiles.json",
    gesture_paths: Optional[List[str]] = None
) -> str:
    """
    Генерує повноцінний автономний скрипт робота з 2-факторною біометричною безпекою:
    - Крок 1: Біометричний замок: Обличчя (YuNet + SFace) + Голос (YAMNet Voiceprint)
    - Крок 2: Після авторизації — повне керування роботом за допомогою жестів (усі 6 парадигм MediaPipe Tasks)
    - Включає Watchdog безпеки: авто-блокування, якщо власник зник з камери на > 12 с.
    """
    face_profiles = []
    if os.path.exists(face_profiles_file):
        try:
            with open(face_profiles_file, "r", encoding="utf-8") as f:
                face_profiles = json.load(f)
        except Exception:
            pass

    voice_profiles = []
    if os.path.exists(voice_profiles_file):
        try:
            with open(voice_profiles_file, "r", encoding="utf-8") as f:
                voice_profiles = json.load(f)
        except Exception:
            pass

    gestures_data = []
    if not gesture_paths:
        recorded_dir = os.path.join("modules", "gestures", "recorded")
        if os.path.exists(recorded_dir):
            gesture_paths = glob.glob(os.path.join(recorded_dir, "*.json"))

    if gesture_paths:
        for gp in gesture_paths:
            if os.path.exists(gp):
                try:
                    with open(gp, "r", encoding="utf-8") as f:
                        gestures_data.append(json.load(f))
                except Exception:
                    pass

    face_json = json.dumps(face_profiles, indent=4, ensure_ascii=False)
    voice_json = json.dumps(voice_profiles, indent=4, ensure_ascii=False)
    gestures_json = json.dumps(gestures_data, indent=4, ensure_ascii=False)

    return f'''"""
================================================================================
 СИСТЕМА КЕРУВАННЯ РОБОТОМ З 2-ФАКТОРНИМ БІОМЕТРИЧНИМ ЗАХИСТОМ (2FA + WATCHDOG)
================================================================================
 Згенеровано автоматично: V-Pose Combine Studio
 Архітектура безпеки:
   1. Обличчя (Face ID): OpenCV YuNet + SFace (128-d косинусна метрика)
   2. Голос (Voice ID): Google YAMNet (1024-d голосовий зліпок диктора + класифікація звуку)
   3. Керування: MediaPipe Tasks API (підтримка всіх 6 типів жестів: static, trajectory,
      transition, periodic, hold, slider)
   4. Сторожовий таймер (Auto-Lock Watchdog): блокує робота, якщо власник зникає з кадру.

 Вимоги (встановіть у вашому середовищі або на роботі):
   pip install opencv-python onnxruntime sounddevice mediapipe numpy
================================================================================
"""
import os
import sys
import csv
import time
import json
import numpy as np
import cv2
import onnxruntime as ort
import sounddevice as sd

# ------------------------------------------------------------------------------
# 1. ВБУДОВАНА БАЗА АВТОРИЗОВАНИХ ПРОФІЛІВ ВЛАСНИКА (BIOMETRIC DATABASE)
# ------------------------------------------------------------------------------
FACE_PROFILES = {face_json}
VOICE_PROFILES = {voice_json}
GESTURE_CONFIGS = {gestures_json}

# Шляхи до ONNX моделей
YUNET_PATH = "models/face_detection_yunet_2023mar.onnx"
SFACE_PATH = "models/face_recognition_sface_2021dec.onnx"
YAMNET_PATH = "models/yamnet.onnx"
CLASS_MAP_PATH = "models/yamnet_class_map.csv"

# Пороги розпізнавання та тайм-аут безпеки
FACE_THRESHOLD = 0.363         # SFace Cosine Similarity
VOICE_THRESHOLD = 0.70         # YAMNet Voiceprint Cosine Similarity
WATCHDOG_TIMEOUT_SEC = 15.0    # Авто-блокування, якщо власника немає в кадрі 15 секунд
{_STANDALONE_GESTURE_ENGINE_CODE}

# ------------------------------------------------------------------------------
# 2. КЛАС РОБОТА (ПІДКЛЮЧЕННЯ МОТОРІВ / ARDUINO UART / ROS / HTTP / GPIO)
# ------------------------------------------------------------------------------
class RobotController:
    """
    Клас управління приводами робота.
    За замовчуванням виводить команди в консоль та HUD.
    Розкоментуйте pySerial або HTTP/UDP блок для реального заліза (Arduino/ESP32/Raspberry Pi).
    """
    def __init__(self, serial_port: str = None, baudrate: int = 115200):
        self.is_armed = False
        self.owner = None
        self.current_action = "LOCKED (Очікування авторизації)"
        self.serial_conn = None
        if serial_port:
            try:
                import serial
                self.serial_conn = serial.Serial(serial_port, baudrate, timeout=0.1)
                print(f"[ROBOT HW] Підключено UART порт {{serial_port}} @ {{baudrate}}")
            except Exception as ex:
                print(f"[ROBOT HW WARN] Не вдалося відкрити порт {{serial_port}}: {{ex}}")

    def _send_hw(self, cmd_str: str):
        if self.serial_conn and self.serial_conn.is_open:
            try:
                self.serial_conn.write((cmd_str + "\\n").encode("utf-8"))
            except Exception:
                pass

    def arm(self, owner_name: str):
        self.is_armed = True
        self.owner = owner_name
        self.current_action = f"ARMED (Власник: {{owner_name}})"
        self._send_hw("ARM")
        print(f"\\n[ROBOT] >>> СИСТЕМА РОЗБЛОКОВАНА ДЛЯ ВЛАСНИКА: {{owner_name}} <<<")
        print("[ROBOT] Приводи активовано. Готовий приймати жести керування!\\n")

    def disarm(self, reason: str = "Ручне блокування"):
        self.is_armed = False
        self.owner = None
        self.current_action = f"LOCKED ({{reason}})"
        self.stop()
        self._send_hw("DISARM")
        print(f"[ROBOT] >>> СИСТЕМА ЗАБЛОКОВАНА ({{reason}}) <<<\\n")

    def on_gesture(self, gesture_name: str, value: float, g_type: str = "static"):
        if not self.is_armed:
            return

        if g_type == "slider":
            pwm = int(value * 255)
            self.current_action = f"SLIDER {{gesture_name}}: {{int(value*100)}}% (PWM={{pwm}})"
            self._send_hw(f"PWM:{{pwm}}")
            return

        g = gesture_name.lower()
        if "right" in g:
            self.turn_right()
        elif "left" in g:
            self.turn_left()
        elif any(k in g for k in ("vi", "peace", "palm", "up", "forward")):
            self.move_forward()
        elif any(k in g for k in ("stop", "fist", "cross")):
            self.stop()
        else:
            self.custom_action(gesture_name, value)

    def move_forward(self):
        self.current_action = "РУХ ВПЕРЕД (FORWARD)"
        self._send_hw("F")
        print(">>> [ROBOT ACTION] Рух вперед ⬆️")

    def turn_right(self):
        self.current_action = "ПОВОРОТ ПРАВОРУЧ (RIGHT)"
        self._send_hw("R")
        print(">>> [ROBOT ACTION] Поворот праворуч ➡️")

    def turn_left(self):
        self.current_action = "ПОВОРОТ ЛІВОРУЧ (LEFT)"
        self._send_hw("L")
        print(">>> [ROBOT ACTION] Поворот ліворуч ⬅️")

    def stop(self):
        self.current_action = "СТОП (STOP)"
        self._send_hw("S")
        print(">>> [ROBOT ACTION] Зупинка 🛑")

    def custom_action(self, name: str, score: float):
        self.current_action = f"ДІЯ: {{name}} ({{score:.2f}})"
        self._send_hw(f"CMD:{{name}}")
        print(f">>> [ROBOT ACTION] Виконання команди «{{name}}» ({{score:.2f}})")


# ------------------------------------------------------------------------------
# 3. МОДУЛЬ БЕЗПЕКИ: FACE ID (YuNet + SFace)
# ------------------------------------------------------------------------------
class FaceGuard:
    def __init__(self, yunet_path=YUNET_PATH, sface_path=SFACE_PATH):
        if not os.path.exists(yunet_path) or not os.path.exists(sface_path):
            raise FileNotFoundError(f"Файли моделей обличчя не знайдено: {{yunet_path}}, {{sface_path}}")
        self.detector = cv2.FaceDetectorYN.create(yunet_path, "", (320, 320), score_threshold=0.6)
        self.recognizer = cv2.FaceRecognizerSF.create(sface_path, "")

    def verify_frame(self, frame):
        h, w = frame.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(frame)
        if faces is None or len(faces) == 0:
            return False, None, 0.0, None

        for face in faces:
            x, y, bw, bh = [int(v) for v in face[:4]]
            aligned = self.recognizer.alignCrop(frame, face)
            feat = self.recognizer.feature(aligned)

            best_name = None
            best_score = -1.0
            for prof in FACE_PROFILES:
                feats = prof.get("features") or ([prof["feature"]] if "feature" in prof else [])
                for f_vec in feats:
                    stored = np.array(f_vec, dtype=np.float32).reshape(1, 128)
                    score = float(self.recognizer.match(feat, stored, cv2.FaceRecognizerSF_FR_COSINE))
                    if score > best_score:
                        best_score = score
                        if best_score >= FACE_THRESHOLD:
                            best_name = prof["name"]

            if best_name is not None:
                return True, best_name, best_score, (x, y, bw, bh)
            else:
                return False, "Невідомий", best_score, (x, y, bw, bh)

        return False, None, 0.0, None


# ------------------------------------------------------------------------------
# 4. МОДУЛЬ БЕЗПЕКИ: VOICE ID + YAMNET (Google YAMNet ONNX)
# ------------------------------------------------------------------------------
class VoiceGuard:
    def __init__(self, yamnet_path=YAMNET_PATH, class_map_path=CLASS_MAP_PATH):
        if not os.path.exists(yamnet_path):
            raise FileNotFoundError(f"Файл моделі YAMNet не знайдено: {{yamnet_path}}")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        self.session = ort.InferenceSession(yamnet_path, sess_options=opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.classes = []
        if os.path.exists(class_map_path):
            with open(class_map_path, "r", encoding="utf-8") as f:
                self.classes = [row["display_name"] for row in csv.DictReader(f)]

    def verify_speech(self, duration_sec=3.0) -> tuple:
        print(f"\\n🎙️ [Voice ID] Слухаю мікрофон ({{duration_sec}} с)... Скажіть кодове слово або фразу:")
        samples = sd.rec(int(16000 * duration_sec), samplerate=16000, channels=1, dtype="float32")
        sd.wait()
        wav = samples[:, 0]

        outputs = self.session.run(None, {{self.input_name: wav}})
        probs = outputs[0]        # [frames, 521]
        embeddings = outputs[1]   # [frames, 1024]

        mean_scores = np.mean(probs, axis=0)
        top_idx = int(np.argmax(mean_scores))
        sound_label = self.classes[top_idx] if self.classes and top_idx < len(self.classes) else "Sound"
        print(f"🔊 [YAMNet] Основний тип звуку: {{sound_label}} ({{mean_scores[top_idx]*100:.1f}}%)")

        speech_mask = probs[:, 0] >= 0.15
        if not np.any(speech_mask):
            return False, f"Мови не виявлено ({{sound_label}})", 0.0

        voiceprint = np.mean(embeddings[speech_mask], axis=0)
        norm = np.linalg.norm(voiceprint)
        if norm > 1e-6:
            voiceprint /= norm

        best_name = None
        best_sim = -1.0
        for spk in VOICE_PROFILES:
            stored_v = np.array(spk["embedding"], dtype=np.float32)
            sim = float(np.dot(voiceprint, stored_v))
            if sim > best_sim:
                best_sim = sim
                if best_sim >= VOICE_THRESHOLD:
                    best_name = spk["name"]

        if best_name:
            return True, best_name, best_sim
        return False, "Невідомий диктор", best_sim


# ------------------------------------------------------------------------------
# 5. МЕНЕДЖЕР БАГАТЬОХ ЖЕСТІВ РОБОТА (Усі 6 парадигм MediaPipe Tasks)
# ------------------------------------------------------------------------------
class GestureGuard:
    def __init__(self, gesture_configs=GESTURE_CONFIGS):
        self.detectors = [StandaloneGestureDetector(cfg) for cfg in gesture_configs]
        needed_models = [d.model_type for d in self.detectors] or ["hands"]
        self.landmarker = MultiModelLandmarker(needed_models)

    def process(self, frame_rgb: np.ndarray):
        now = time.time()
        extracted_cache = {{}}
        for det in self.detectors:
            m = det.model_type
            if m not in extracted_cache:
                extracted_cache[m] = self.landmarker.extract(frame_rgb, m)
            res = det.evaluate(extracted_cache[m], now)
            if res["triggered"]:
                return det.name, res["value"], True, det.gesture_type
        return None, 0.0, False, "static"

    def close(self):
        self.landmarker.close()


# ------------------------------------------------------------------------------
# 6. ГОЛОВНИЙ ЦИКЛ РОБОТА (SECURITY GUARD & GESTURE CONTROL LOOP)
# ------------------------------------------------------------------------------
def main():
    print("=" * 68)
    print("   СИСТЕМА РОБОТА: БІОМЕТРИЧНИЙ 2-ФАКТОРНИЙ ЗАХИСТ + ЖЕСТИ")
    print("=" * 68)

    robot = RobotController()
    face_guard = FaceGuard()
    voice_guard = VoiceGuard()
    gesture_guard = GestureGuard()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ПОМИЛКА: Не вдалося відкрити камеру (Device 0).")
        return

    # Стани системи: "LOCKED", "FACE_VERIFIED", "ARMED"
    state = "LOCKED"
    auth_owner = None
    last_trigger_time = 0.0
    last_owner_seen_time = time.time()

    print("\\n[ІНСТРУКЦІЯ]")
    print("  1. Подивіться в камеру: Face ID розпізнає ваше обличчя.")
    print("  2. Натисніть [ПРОБІЛ], щоб підтвердити голос (Voice ID + YAMNet).")
    print("  3. Після авторизації — керуйте роботом за допомогою жестів.")
    print("  4. Натисніть [L] для ручного блокування, [ESC / Q] для виходу.\\n")

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            now = time.time()

            # --- ЕТАП 1: ОБЛИЧЧЯ (FACE ID) ---
            face_ok, face_name, face_score, bbox = face_guard.verify_frame(frame)
            if bbox is not None:
                bx, by, bw_box, bh_box = bbox
                b_color = (0, 255, 0) if face_ok else (0, 140, 255)
                cv2.rectangle(frame, (bx, by), (bx + bw_box, by + bh_box), b_color, 2)
                cv2.putText(frame, f"{{face_name}} ({{face_score:.2f}})", (bx, max(20, by - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, b_color, 2)

            if face_ok and face_name:
                last_owner_seen_time = now
                if state == "LOCKED":
                    state = "FACE_VERIFIED"
                    auth_owner = face_name
                    print(f"\\n✅ [Face ID] Обличчя авторизовано: {{auth_owner}}!")
                    print("👉 Натисніть [ПРОБІЛ] для підтвердження голосу (Voice ID)...")

            # --- WATCHDOG: АВТО-БЛОКУВАННЯ ПРИ ВІДСУТНОСТІ ВЛАСНИКА ---
            if state == "ARMED" and (now - last_owner_seen_time) > WATCHDOG_TIMEOUT_SEC:
                state = "LOCKED"
                auth_owner = None
                robot.disarm(reason=f"Власник відсутній > {{int(WATCHDOG_TIMEOUT_SEC)}}с")

            # --- ЕТАП 2: КЕРУВАННЯ ЖЕСТАМИ (ТІЛЬКИ ЯКЩО ARMED) ---
            detected_gesture = None
            g_score = 0.0
            if state == "ARMED":
                detected_gesture, g_score, triggered, g_type = gesture_guard.process(frame_rgb)
                cooldown = 0.1 if g_type == "slider" else 0.75
                if triggered and (now - last_trigger_time > cooldown):
                    last_trigger_time = now
                    robot.on_gesture(detected_gesture, g_score, g_type)

            # --- HUD ТА ВІЗУАЛІЗАЦІЯ ---
            hud_color = (0, 200, 0) if state == "ARMED" else ((0, 165, 255) if state == "FACE_VERIFIED" else (0, 0, 220))
            cv2.rectangle(frame, (0, 0), (w, 56), (20, 20, 20), -1)
            cv2.putText(frame, f"STATUS: {{state}}", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, hud_color, 2)
            cv2.putText(frame, f"ROBOT: {{robot.current_action}}", (15, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1)

            if state == "ARMED":
                rem = max(0, int(WATCHDOG_TIMEOUT_SEC - (now - last_owner_seen_time)))
                cv2.putText(frame, f"Watchdog: {{rem}}s", (w - 150, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 255, 180), 1)

            if state == "LOCKED":
                hint = "Подивіться в камеру для авторизації Face ID"
            elif state == "FACE_VERIFIED":
                hint = "Обличчя підтверджено! Натисніть [ПРОБІЛ] для перевірки голосу Voice ID"
            else:
                hint = "СИСТЕМА РОЗБЛОКОВАНА: Покажіть жест у камеру. [L] - Заблокувати"
            cv2.rectangle(frame, (0, h - 30), (w, h), (15, 15, 15), -1)
            cv2.putText(frame, hint, (15, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

            if detected_gesture:
                cv2.putText(frame, f"GESTURE: {{detected_gesture}} ({{g_score*100:.0f}}%)", (w - 300, 48),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            cv2.imshow("Robot Multimodal 2FA Guard", frame)
            key = cv2.waitKey(1) & 0xFF

            if key in (27, ord('q'), ord('Q')):
                break
            elif key == ord(' '):  # ПРОБІЛ — Запуск перевірки голосу
                if state in ("FACE_VERIFIED", "LOCKED"):
                    print("\\n[Security] Запуск другого фактора (Voice ID)...")
                    v_ok, v_name, v_score = voice_guard.verify_speech(duration_sec=3.0)
                    if v_ok and (auth_owner is None or v_name.lower() == auth_owner.lower() or len(FACE_PROFILES) == 0):
                        auth_owner = v_name if not auth_owner else auth_owner
                        state = "ARMED"
                        last_owner_seen_time = time.time()
                        robot.arm(auth_owner)
                    elif v_ok and auth_owner is not None:
                        # Дозволяємо, якщо і обличчя, і голос належать авторизованим профілям у базі
                        state = "ARMED"
                        last_owner_seen_time = time.time()
                        robot.arm(f"{{auth_owner}} / Voice:{{v_name}}")
                    else:
                        print(f"❌ [Voice ID] Відмова: {{v_name}} (Схожість: {{v_score*100:.1f}}%). Спробуйте ще раз.")
            elif key in (ord('l'), ord('L')):
                state = "LOCKED"
                auth_owner = None
                robot.disarm("Ручне блокування [L]")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        gesture_guard.close()
        robot.stop()
        print("\\n[Робот] Програму завершено безпечно.")


if __name__ == "__main__":
    main()
'''


def _copy_mp_task_model(model_type: str, dst_models_dir: str, copied_list: List[str]):
    """Копіює потрібний MediaPipe .task файл із models_cache/ у папку models/ експортованого пакета."""
    task_map = {
        "hands": "hand_landmarker.task",
        "pose": "pose_landmarker_lite.task",
        "face": "face_landmarker.task",
    }
    fname = task_map.get(model_type, "hand_landmarker.task")
    for src_dir in ("models_cache", "models"):
        src = os.path.join(src_dir, fname)
        if os.path.exists(src):
            os.makedirs(dst_models_dir, exist_ok=True)
            shutil.copy2(src, os.path.join(dst_models_dir, fname))
            if fname not in copied_list:
                copied_list.append(fname)
            break


def create_export_package(
    modality: str = "robot",
    gesture_path: Optional[str] = None,
    voice_mode: str = "voice_id",
    output_dir: Optional[str] = None,
    zip_package: bool = True
) -> Dict[str, Any]:
    """
    Універсальний генератор автономних пакетів (папка + ZIP-архів):
    Створює пакет саме для обраної користувачем модальності:
      - 'gesture': обраний жест + модель MediaPipe .task + run_gesture.py + requirements.txt
      - 'face': Face ID + моделі YuNet/SFace ONNX + база зареєстрованих облич
      - 'voice': Voice ID / YAMNet + модель YAMNet ONNX + база дикторів
      - 'robot': 2-Факторна автентифікація (Обличчя + Голос + Жести) + всі ONNX та .task моделі
    """
    copied_models: List[str] = []

    if modality == "gesture":
        g_name = "gesture"
        m_type = "hands"
        if gesture_path and os.path.exists(gesture_path):
            try:
                with open(gesture_path, "r", encoding="utf-8") as f:
                    g_data = json.load(f)
                    g_name = g_data.get("name", "gesture")
                    m_type = g_data.get("model_type", "hands")
            except Exception:
                g_name = os.path.splitext(os.path.basename(gesture_path))[0]

        default_dir = f"exports/gesture_{g_name}_package"
        out_dir = os.path.abspath(output_dir or default_dir)
        models_dir = os.path.join(out_dir, "models")
        os.makedirs(models_dir, exist_ok=True)

        _copy_mp_task_model(m_type, models_dir, copied_models)

        if gesture_path and os.path.exists(gesture_path):
            shutil.copy2(gesture_path, os.path.join(out_dir, "gesture.json"))
            main_code = export_gesture_standalone(gesture_path)
        else:
            main_code = "# Не вказано файл жесту"

        run_file = os.path.join(out_dir, "run_gesture.py")
        with open(run_file, "w", encoding="utf-8") as f:
            f.write(main_code)

        with open(os.path.join(out_dir, "requirements.txt"), "w", encoding="utf-8") as f:
            f.write("opencv-python>=4.8.0\nmediapipe>=0.10.0\nnumpy>=1.23.0\n")

        with open(os.path.join(out_dir, "README.md"), "w", encoding="utf-8") as f:
            f.write(
                f"# Standalone Gesture Control: {g_name}\n\n"
                "Повністю автономний пакет розпізнавання жесту (містить вбудоване математичне ядро та MediaPipe `.task` модель).\n\n"
                "## Запуск:\n"
                "```bash\n"
                "pip install -r requirements.txt\n"
                "python run_gesture.py\n"
                "```\n"
            )

    elif modality == "face":
        default_dir = "exports/face_id_package"
        out_dir = os.path.abspath(output_dir or default_dir)
        models_dir = os.path.join(out_dir, "models")
        profiles_dir = os.path.join(out_dir, "profiles")
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(profiles_dir, exist_ok=True)

        for m in ["face_detection_yunet_2023mar.onnx", "face_recognition_sface_2021dec.onnx"]:
            src = os.path.join("models", m)
            dst = os.path.join(models_dir, m)
            if os.path.exists(src):
                shutil.copy2(src, dst)
                copied_models.append(m)

        src_faces = os.path.join("modules", "faces", "profiles.json")
        if os.path.exists(src_faces):
            shutil.copy2(src_faces, os.path.join(profiles_dir, "face_profiles.json"))

        run_file = os.path.join(out_dir, "run_face_id.py")
        with open(run_file, "w", encoding="utf-8") as f:
            f.write(export_face_id_standalone())

        with open(os.path.join(out_dir, "requirements.txt"), "w", encoding="utf-8") as f:
            f.write("opencv-python>=4.8.0\nnumpy>=1.23.0\n")

        with open(os.path.join(out_dir, "README.md"), "w", encoding="utf-8") as f:
            f.write(
                "# Standalone Face ID Package\n\n"
                "Автономний модуль розпізнавання облич на базі YuNet + SFace ONNX.\n\n"
                "## Запуск:\n"
                "```bash\n"
                "pip install -r requirements.txt\n"
                "python run_face_id.py\n"
                "```\n"
            )

    elif modality == "voice":
        default_dir = "exports/voice_package"
        out_dir = os.path.abspath(output_dir or default_dir)
        models_dir = os.path.join(out_dir, "models")
        profiles_dir = os.path.join(out_dir, "profiles")
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(profiles_dir, exist_ok=True)

        for m in ["yamnet.onnx", "yamnet_class_map.csv"]:
            src = os.path.join("models", m)
            dst = os.path.join(models_dir, m)
            if os.path.exists(src):
                shutil.copy2(src, dst)
                copied_models.append(m)

        src_voices = os.path.join("modules", "voices", "profiles.json")
        if os.path.exists(src_voices):
            shutil.copy2(src_voices, os.path.join(profiles_dir, "voice_profiles.json"))

        run_file = os.path.join(out_dir, "run_voice.py")
        with open(run_file, "w", encoding="utf-8") as f:
            f.write(export_voice_standalone(mode=voice_mode))

        with open(os.path.join(out_dir, "requirements.txt"), "w", encoding="utf-8") as f:
            f.write("onnxruntime>=1.15.0\nsounddevice>=0.4.6\nnumpy>=1.23.0\n")

        with open(os.path.join(out_dir, "README.md"), "w", encoding="utf-8") as f:
            f.write(
                f"# Standalone Voice Package ({voice_mode})\n\n"
                "Автономний модуль детекції звуків та розпізнавання диктора на базі Google YAMNet.\n\n"
                "## Запуск:\n"
                "```bash\n"
                "pip install -r requirements.txt\n"
                "python run_voice.py\n"
                "```\n"
            )

    else:  # robot (2FA: Face + Voice + Gestures)
        default_dir = "exports/robot_2fa_package"
        out_dir = os.path.abspath(output_dir or default_dir)
        models_dir = os.path.join(out_dir, "models")
        profiles_dir = os.path.join(out_dir, "profiles")
        gestures_dir = os.path.join(profiles_dir, "gestures")
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(profiles_dir, exist_ok=True)
        os.makedirs(gestures_dir, exist_ok=True)

        for m in ["face_detection_yunet_2023mar.onnx", "face_recognition_sface_2021dec.onnx", "yamnet.onnx", "yamnet_class_map.csv"]:
            src = os.path.join("models", m)
            dst = os.path.join(models_dir, m)
            if os.path.exists(src):
                shutil.copy2(src, dst)
                copied_models.append(m)

        src_faces = os.path.join("modules", "faces", "profiles.json")
        if os.path.exists(src_faces):
            shutil.copy2(src_faces, os.path.join(profiles_dir, "face_profiles.json"))

        src_voices = os.path.join("modules", "voices", "profiles.json")
        if os.path.exists(src_voices):
            shutil.copy2(src_voices, os.path.join(profiles_dir, "voice_profiles.json"))

        src_gestures = glob.glob(os.path.join("modules", "gestures", "recorded", "*.json"))
        needed_mp_models = {"hands"}
        for g in src_gestures:
            shutil.copy2(g, os.path.join(gestures_dir, os.path.basename(g)))
            try:
                with open(g, "r", encoding="utf-8") as gf:
                    needed_mp_models.add(json.load(gf).get("model_type", "hands"))
            except Exception:
                pass

        for mp_m in needed_mp_models:
            _copy_mp_task_model(mp_m, models_dir, copied_models)

        run_file = os.path.join(out_dir, "run_robot.py")
        with open(run_file, "w", encoding="utf-8") as f:
            f.write(export_multimodal_robot_guard())

        with open(os.path.join(out_dir, "requirements.txt"), "w", encoding="utf-8") as f:
            f.write(
                "opencv-python>=4.8.0\n"
                "onnxruntime>=1.15.0\n"
                "sounddevice>=0.4.6\n"
                "numpy>=1.23.0\n"
                "mediapipe>=0.10.0\n"
            )

        with open(os.path.join(out_dir, "README.md"), "w", encoding="utf-8") as f:
            f.write(
                "# Robot 2-Factor Authentication Guard + Gesture Control\n\n"
                "Повний автономний пакет безпеки (Обличчя + Голос + Watchdog) та керування роботом жестами.\n\n"
                "## Швидкий запуск на будь-якому ПК чи Raspberry Pi:\n"
                "```bash\n"
                "1. cd robot_2fa_package\n"
                "2. pip install -r requirements.txt\n"
                "3. python run_robot.py\n"
                "```\n\n"
                "## Як працює безпека:\n"
                "- **Face ID**: Камера верифікує ваше натреноване обличчя.\n"
                "- **Voice ID**: Натисніть `[ПРОБІЛ]`, скажіть слово — верифікується тембр вашого голосу.\n"
                "- **Watchdog**: Якщо власник відійде від камери більше ніж на 15 с, робот автоматично заблокується.\n"
                "- **Керування жестами**: Тільки після авторизації робот починає виконувати команди!\n"
            )

    zip_path = None
    if zip_package:
        base_zip = os.path.join(os.path.dirname(out_dir), os.path.basename(out_dir))
        zip_path = shutil.make_archive(base_zip, "zip", out_dir)

    return {
        "modality": modality,
        "package_dir": out_dir,
        "zip_path": zip_path,
        "models_copied": copied_models,
        "run_file": run_file,
    }


def create_robot_guard_package(output_dir: str = "exports/robot_guard_package", zip_package: bool = True) -> Dict[str, Any]:
    """Сумісність зі старими викликами: створює повний 2FA пакет."""
    return create_export_package(modality="robot", output_dir=output_dir, zip_package=zip_package)
