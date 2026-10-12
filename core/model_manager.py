"""
Менеджер завантаження та виконання AI-моделей (MediaPipe Tasks).
Завантажує в пам'ять лише затребувані моделі (On-Demand / Lazy Loading).
"""
import os
import time
import urllib.request
from typing import Set, Dict, Any, Optional
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Базові конфігурації підтримуваних моделей
MODEL_CONFIGS = {
    "face": {
        "url": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        "filename": "face_landmarker.task",
        "description": "MediaPipe Face Landmarker (Сітка обличчя + Blendshapes очей/міміки)"
    },
    "hands": {
        "url": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        "filename": "hand_landmarker.task",
        "description": "MediaPipe Hand Landmarker (Точки кистей та пальців)"
    },
    "pose": {
        "url": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
        "filename": "pose_landmarker_lite.task",
        "description": "MediaPipe Pose Landmarker (Скелет тіла)"
    }
}


class ModelManager:
    def __init__(self, required_models: Set[str], cache_dir: str = "models_cache"):
        self.cache_dir = os.path.abspath(cache_dir)
        os.makedirs(self.cache_dir, exist_ok=True)
        self.required_models = set(required_models)
        self.models: Dict[str, Any] = {}
        self._last_timestamp_ms = -1
        self._session_start_time = time.time()

        self._initialize_required_models()

    def _ensure_model_file(self, model_key: str) -> str:
        """Перевіряє наявність файлу моделі, завантажує за потреби."""
        cfg = MODEL_CONFIGS[model_key]
        filepath = os.path.join(self.cache_dir, cfg["filename"])
        if not os.path.exists(filepath):
            print(f"[ModelManager] Завантаження моделі '{model_key}' ({cfg['description']})...")
            urllib.request.urlretrieve(cfg["url"], filepath)
            print(f"[ModelManager] Модель '{model_key}' успішно збережена в {filepath}")
        return filepath

    def _initialize_required_models(self):
        """Ініціалізує лише ті моделі, які потрібні для активних модулів."""
        for model_key in self.required_models:
            if model_key not in MODEL_CONFIGS:
                print(f"[ModelManager] УВАГА: Невідома модель '{model_key}', пропускаємо.")
                continue

            model_path = self._ensure_model_file(model_key)
            print(f"[ModelManager] Ініціалізація моделі '{model_key}'...")

            base_options = python.BaseOptions(model_asset_path=model_path)

            if model_key == "face":
                options = vision.FaceLandmarkerOptions(
                    base_options=base_options,
                    running_mode=vision.RunningMode.VIDEO,
                    output_face_blendshapes=True,
                    num_faces=1
                )
                self.models["face"] = vision.FaceLandmarker.create_from_options(options)

            elif model_key == "hands":
                options = vision.HandLandmarkerOptions(
                    base_options=base_options,
                    running_mode=vision.RunningMode.VIDEO,
                    num_hands=2
                )
                self.models["hands"] = vision.HandLandmarker.create_from_options(options)

            elif model_key == "pose":
                options = vision.PoseLandmarkerOptions(
                    base_options=base_options,
                    running_mode=vision.RunningMode.VIDEO
                )
                self.models["pose"] = vision.PoseLandmarker.create_from_options(options)

        print(f"[ModelManager] Активні моделі: {list(self.models.keys())}")

    def process(self, frame_rgb, timestamp_ms: Optional[int] = None) -> Dict[str, Any]:
        """
        Проганяє кадр через активні моделі.
        Гарантує строго монотонне зростання таймстемпів для MediaPipe.
        """
        if not self.models:
            return {}

        if timestamp_ms is None:
            ts = int((time.time() - self._session_start_time) * 1000)
        else:
            ts = int(timestamp_ms)

        # Захист від ValueError: Input timestamp must be monotonically increasing.
        if ts <= self._last_timestamp_ms:
            ts = self._last_timestamp_ms + 1
        self._last_timestamp_ms = ts

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        results = {}

        if "face" in self.models:
            face_res = self.models["face"].detect_for_video(mp_image, ts)
            results["face"] = face_res
            # Зручний словник Blendshapes (міміка, кліпання) для швидкого доступу
            blendshapes_dict = {}
            if face_res.face_blendshapes and len(face_res.face_blendshapes) > 0:
                for category in face_res.face_blendshapes[0]:
                    blendshapes_dict[category.category_name] = category.score
            results["face_blendshapes"] = blendshapes_dict

        if "hands" in self.models:
            results["hands"] = self.models["hands"].detect_for_video(mp_image, ts)

        if "pose" in self.models:
            results["pose"] = self.models["pose"].detect_for_video(mp_image, ts)

        return results

    def close(self):
        """Звільняє ресурси моделей."""
        for name, model in self.models.items():
            try:
                model.close()
            except Exception:
                pass
        self.models.clear()
