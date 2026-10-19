import os
import json
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

YUNET_PATH = os.path.join("models", "face_detection_yunet_2023mar.onnx")
SFACE_PATH = os.path.join("models", "face_recognition_sface_2021dec.onnx")
FACES_DIR = os.path.join("modules", "faces")
CROPS_DIR = os.path.join("modules", "faces", "crops")
PROFILES_FILE = os.path.join(FACES_DIR, "profiles.json")

# Поріг косинусної схожості для SFace (офіційна рекомендація OpenCV ~0.363)
DEFAULT_COSINE_THRESHOLD = 0.363


class FaceEngine:
    """Модуль детекції, реєстрації та розпізнавання облич на базі YuNet + SFace."""

    def __init__(self, score_thresh: float = 0.65, cosine_thresh: float = DEFAULT_COSINE_THRESHOLD):
        self.score_thresh = score_thresh
        self.cosine_thresh = cosine_thresh
        self.detector = None
        self.recognizer = None
        self.profiles: List[Dict[str, Any]] = []

        os.makedirs(FACES_DIR, exist_ok=True)
        os.makedirs(CROPS_DIR, exist_ok=True)

        self._liveness_history: List[float] = []
        self.liveness_required: bool = True
        self._init_models()
        self.load_profiles()

    def reset_liveness(self):
        """Скидає історію Liveness для нової перевірки."""
        self._liveness_history.clear()

    def _init_models(self):
        if not os.path.exists(YUNET_PATH) or not os.path.exists(SFACE_PATH):
            raise FileNotFoundError(
                f"Моделі облич відсутні за шляхами {YUNET_PATH} або {SFACE_PATH}."
            )
        self.detector = cv2.FaceDetectorYN.create(
            model=YUNET_PATH,
            config="",
            input_size=(320, 320),
            score_threshold=self.score_thresh,
            nms_threshold=0.3,
            top_k=5000,
        )
        self.recognizer = cv2.FaceRecognizerSF.create(
            model=SFACE_PATH,
            config="",
        )

    def load_profiles(self) -> List[Dict[str, Any]]:
        if os.path.exists(PROFILES_FILE):
            try:
                with open(PROFILES_FILE, "r", encoding="utf-8") as f:
                    self.profiles = json.load(f)
            except Exception:
                self.profiles = []
        else:
            self.profiles = []
        return self.profiles

    def save_profiles(self):
        with open(PROFILES_FILE, "w", encoding="utf-8") as f:
            json.dump(self.profiles, f, indent=2, ensure_ascii=False)

    def delete_profile(self, profile_id: str) -> bool:
        initial_len = len(self.profiles)
        self.profiles = [p for p in self.profiles if p.get("id") != profile_id]
        if len(self.profiles) != initial_len:
            self.save_profiles()
            # Видаляємо файл кропу, якщо він існує
            crop_path = os.path.join(CROPS_DIR, f"{profile_id}.jpg")
            if os.path.exists(crop_path):
                try:
                    os.remove(crop_path)
                except Exception:
                    pass
            return True
        return False

    def detect_raw(self, frame: np.ndarray) -> Optional[np.ndarray]:
        h, w = frame.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(frame)
        return faces

    def _extract_largest_face_feature(self, frame: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Знаходить найбільше обличчя на кадрі, повертає (aligned_crop, 128d_feature)."""
        faces = self.detect_raw(frame)
        if faces is None or len(faces) == 0:
            return None

        best_face = None
        max_area = 0
        for f in faces:
            w, h = f[2], f[3]
            area = w * h
            if area > max_area:
                max_area = area
                best_face = f

        aligned = self.recognizer.alignCrop(frame, best_face)
        feature = self.recognizer.feature(aligned)
        return aligned, feature

    def register_face(self, name: str, frame: np.ndarray) -> Dict[str, Any]:
        """
        Захоплює обличчя з одного кадру або делегує у мульти-кадрову реєстрацію.
        """
        return self.register_face_multi(name, [frame])

    def register_face_multi(self, name: str, frames: List[np.ndarray]) -> Dict[str, Any]:
        """
        Мульти-кадрова реєстрація (Multi-Shot Enrollment):
        Вилучає 128-d ембеддінги з серії кадрів, відкидає кадри без обличчя,
        обчислює нормалізований центроїд та зберігає індивідуальні зліпки для високої точності.
        """
        features_list: List[List[float]] = []
        best_crop: Optional[np.ndarray] = None

        for frm in frames:
            if frm is None:
                continue
            res = self._extract_largest_face_feature(frm)
            if res is not None:
                crop, feat = res
                if best_crop is None:
                    best_crop = crop
                features_list.append(feat[0].tolist())

        if not features_list or best_crop is None:
            raise ValueError("Обличчя не виявлено в кадрі. Подивіться прямо в камеру при достатньому освітленні.")

        # Розрахунок усередненого та L2-нормалізованого вектора (центроїда)
        mat = np.array(features_list, dtype=np.float32)
        mean_feat = np.mean(mat, axis=0)
        norm = float(np.linalg.norm(mean_feat))
        if norm > 1e-6:
            mean_feat = mean_feat / norm
        centroid_list = mean_feat.tolist()

        prof_id = str(uuid.uuid4())[:8]
        crop_path = os.path.join(CROPS_DIR, f"{prof_id}.jpg")
        cv2.imwrite(crop_path, best_crop)

        profile = {
            "id": prof_id,
            "name": name.strip(),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "crop_path": crop_path.replace("\\", "/"),
            "samples_count": len(features_list),
            "feature": centroid_list,
            "features": [centroid_list] + features_list[:4],
        }
        self.profiles.append(profile)
        self.save_profiles()
        return profile

    def assess_frame_quality(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Оцінює якість кадру перед реєстрацією або верифікацією:
        - Освітленість (яскравість 0..255)
        - Наявність обличчя та його відносний розмір (відстань до камери)
        - Центрування обличчя в кадрі
        """
        if frame is None or frame.size == 0:
            return {
                "ok": False,
                "light_ok": False,
                "distance_ok": False,
                "brightness": 0,
                "distance_status": "Немає кадру",
                "light_status": "Немає кадру",
                "face_ratio": 0.0,
                "hint": "Камера не передає зображення",
                "color": "#ef4444"
            }

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = int(np.mean(gray))
        h, w = frame.shape[:2]
        frame_area = max(1, w * h)

        if brightness < 55:
            light_status = f"Занадто темно ({brightness}/255)"
            light_ok = False
        elif brightness > 225:
            light_status = f"Засвічено ({brightness}/255)"
            light_ok = False
        else:
            light_status = f"Освітлення в нормі ({brightness}/255)"
            light_ok = True

        faces = self.detect_raw(frame)
        if faces is None or len(faces) == 0:
            return {
                "ok": False,
                "light_ok": light_ok,
                "distance_ok": False,
                "brightness": brightness,
                "light_status": light_status,
                "distance_status": "Обличчя не в кадрі",
                "face_ratio": 0.0,
                "hint": "Подивіться прямо в об'єктив камери",
                "color": "#f59e0b",
            }

        # Беремо найбільше обличчя
        best_f = max(faces, key=lambda f: f[2] * f[3])
        fx, fy, fw, fh = [float(v) for v in best_f[:4]]
        ratio = (fw * fh) / frame_area
        cx, cy = (fx + fw / 2.0) / w, (fy + fh / 2.0) / h

        if ratio < 0.035:
            dist_status = "Занадто далеко — наблизьтесь"
            dist_ok = False
        elif ratio > 0.45:
            dist_status = "Занадто близько — відсуньтесь"
            dist_ok = False
        else:
            dist_status = "Відстань ідеальна (40–80 см)"
            dist_ok = True

        centered = (0.20 <= cx <= 0.80) and (0.15 <= cy <= 0.85)
        if not centered:
            dist_status = "Зміщено з центру кадру"
            dist_ok = False

        all_ok = light_ok and dist_ok
        if all_ok:
            hint = "✅ Ідеальна позиція та освітлення — готово до зйомки!"
            color = "#22c55e"
        elif not light_ok:
            hint = f"⚠️ {light_status} — додайте світла спереду"
            color = "#f59e0b"
        else:
            hint = f"⚠️ {dist_status}"
            color = "#f59e0b"

        return {
            "ok": all_ok,
            "light_ok": light_ok,
            "distance_ok": dist_ok,
            "brightness": brightness,
            "light_status": light_status,
            "distance_status": dist_status,
            "face_ratio": round(ratio, 3),
            "hint": hint,
            "color": color,
        }

    def compute_liveness(self, landmarks: List[List[int]]) -> Tuple[bool, float, str]:
        """
        Перевірка «живості» (Liveness / Anti-Spoofing проти фото з телефона):
        Аналізує зміну відносного положення кінчика носа між правим і лівим оком (3D паралакс голови)
        за останні кадри. Якщо користувач злегка повернув або нахилив голову — liveness = True.
        """
        if len(landmarks) < 3:
            return False, 0.0, "Очікування точок"
        reye, leye, nose = landmarks[0], landmarks[1], landmarks[2]
        eye_dist = max(1.0, float(np.hypot(leye[0] - reye[0], leye[1] - reye[1])))
        eye_mid_x = (reye[0] + leye[0]) / 2.0
        yaw_ratio = (nose[0] - eye_mid_x) / eye_dist

        self._liveness_history.append(yaw_ratio)
        if len(self._liveness_history) > 25:
            self._liveness_history.pop(0)

        if len(self._liveness_history) < 4:
            return False, 0.2, "Дивіться в камеру та злегка поверніть голову"

        span = float(max(self._liveness_history) - min(self._liveness_history))
        # Якщо паралакс змінився хоча б на 0.075 (легкий рух живої голови в 3D)
        is_live = span >= 0.075
        progress = min(1.0, span / 0.075)
        status = "✅ Жива людина (3D рух підтверджено)" if is_live else f"Злегка поверніть голову ({int(progress*100)}%)"
        return is_live, round(progress, 2), status

    def recognize(self, frame: np.ndarray, check_liveness: bool = False) -> List[Dict[str, Any]]:
        """
        Знаходить усі обличчя на кадрі та зіставляє їх із збереженими профілями.
        Підтримує як одиничний вектор 'feature', так і мульти-ембеддінги 'features',
        а також Liveness / Anti-Spoofing перевірку.
        """
        results = []
        faces = self.detect_raw(frame)
        if faces is None:
            self._liveness_history.clear()
            return results

        for face in faces:
            bbox = [int(v) for v in face[:4]]
            landmarks = face[4:14].reshape((5, 2)).astype(int).tolist()

            aligned = self.recognizer.alignCrop(frame, face)
            current_feat = self.recognizer.feature(aligned)

            best_name = "Невідомий"
            best_id = None
            best_score = -1.0
            is_known = False

            for prof in self.profiles:
                candidate_vectors = prof.get("features") or ([prof["feature"]] if "feature" in prof else [])
                for vec in candidate_vectors:
                    stored_feat = np.array(vec, dtype=np.float32).reshape(1, 128)
                    score = float(self.recognizer.match(current_feat, stored_feat, cv2.FaceRecognizerSF_FR_COSINE))
                    if score > best_score:
                        best_score = score
                        if best_score >= self.cosine_thresh:
                            best_name = prof["name"]
                            best_id = prof["id"]
                            is_known = True

            pct = 0
            if best_score > 0:
                pct = int(min(100, max(0, (best_score / 0.75) * 100)))

            is_live, live_prog, live_msg = self.compute_liveness(landmarks)
            is_authorized = is_known and (not check_liveness or is_live)

            results.append({
                "bbox": bbox,
                "landmarks": landmarks,
                "name": best_name,
                "id": best_id,
                "is_known": is_known,
                "is_authorized": is_authorized,
                "score": best_score,
                "confidence_pct": pct,
                "is_live": is_live,
                "liveness_ok": is_live,
                "liveness_progress": live_prog,
                "liveness_status": live_msg,
                "liveness_hint": live_msg,
            })

        return results

    def draw_results(self, frame: np.ndarray, results: List[Dict[str, Any]]) -> np.ndarray:
        """
        Малює рамки, імена, відсотки схожості та маркери орієнтирів на кадрі.
        """
        if not results:
            return frame

        out = frame.copy()
        for r in results:
            bbox = r["bbox"]
            x, y, bw, bh = bbox
            name = r["name"]
            is_known = r.get("is_known", False)
            is_auth = r.get("is_authorized", is_known)
            pct = r.get("confidence_pct", 0)

            # Зелений — розпізнано та авторизовано, синій — очікує руху (liveness), помаранчевий — чужий
            if is_auth:
                color = (0, 220, 100)
            elif is_known:
                color = (255, 180, 50)
            else:
                color = (50, 120, 245)

            label = f"{name} ({pct}%)"
            if r.get("liveness_status") and is_known and not is_auth:
                label += " [Check 3D]"

            # Рамка навколо обличчя
            cv2.rectangle(out, (x, y), (x + bw, y + bh), color, 2)

            # Фон підпису
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)[0]
            cv2.rectangle(out, (x, max(0, y - 24)), (x + label_size[0] + 10, max(24, y)), color, -1)
            cv2.putText(out, label, (x + 5, max(18, y - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

            # Точки обличчя (очі, ніс, кути рота)
            for pt in r.get("landmarks", []):
                cv2.circle(out, (int(pt[0]), int(pt[1])), 3, (0, 255, 255), -1)

        return out
