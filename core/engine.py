"""
Головний конвеєр (Pipeline Engine).
Координує камеру, завантаження моделей, виклик жестів, фільтрацію та виконання дій.
"""
import time
import os
import cv2
import yaml
from typing import Dict, Any, List, Optional

from core.models import FrameContext, CommandEvent, DetectionResult
from core.model_manager import ModelManager
from core.module_loader import load_gesture_module, load_action_module, load_source_module
from core.visualizer import Visualizer
from modules.sources.webcam import VideoSource as DefaultWebcamSource


class VirtualBiometricDetector:
    """Віртуальний детектор для прив'язки подій Face ID, Voice ID або звуків YAMNet до дій у pipeline.yaml."""
    def __init__(self, spec: str):
        self.spec = spec  # наприклад: "face:Max", "yamnet:Clapping", "voice:max"
        parts = spec.split(":", 1)
        self.kind = parts[0].lower()
        self.target = parts[1].strip() if len(parts) > 1 else ""
        self.requires: List[str] = []

    def evaluate_from_engines(self, frame, face_engine, audio_mgr) -> DetectionResult:
        if self.kind == "face" and face_engine is not None:
            faces = face_engine.recognize(frame)
            for f in faces:
                if f.get("is_known") and (self.target == "*" or f.get("name", "").lower() == self.target.lower()):
                    return DetectionResult(triggered=True, value=float(f.get("score", 1.0)), confidence=1.0)
            return DetectionResult(triggered=False, value=0.0)

        if self.kind == "yamnet" and audio_mgr is not None and audio_mgr.running:
            snap = audio_mgr.get_snapshot()
            for ev in snap.get("top_events", []):
                if self.target.lower() in (ev.get("name", "").lower(), ev.get("label", "").lower()):
                    if ev.get("score", 0.0) >= 0.15:
                        return DetectionResult(triggered=True, value=float(ev.get("score", 1.0)), confidence=1.0)
            return DetectionResult(triggered=False, value=0.0)

        if self.kind == "voice" and audio_mgr is not None and audio_mgr.running:
            spk = audio_mgr.get_snapshot().get("speaker", {})
            if spk.get("is_verified") and (self.target == "*" or spk.get("name", "").lower() == self.target.lower()):
                return DetectionResult(triggered=True, value=float(spk.get("similarity", 1.0)), confidence=1.0)
            return DetectionResult(triggered=False, value=0.0)

        return DetectionResult(triggered=False, value=0.0)


class Binding:
    """Зв'язка між конкретним тригером (жест, обличчя, звук, голос) та дією з фільтрами."""
    def __init__(self, binding_cfg: Dict[str, Any]):
        self.id = binding_cfg.get("id", "unnamed_binding")
        self.name = binding_cfg.get("name", self.id)

        # 1. Завантаження тригера (жест або біометрична/звукова подія)
        gesture_cfg = binding_cfg["gesture"]
        g_file = str(gesture_cfg.get("file", ""))
        self.is_virtual = g_file.startswith(("face:", "yamnet:", "voice:"))
        if self.is_virtual:
            self.detector = VirtualBiometricDetector(g_file)
            self.requires = []
            self.gesture_name = g_file
        else:
            self.detector, self.requires, self.gesture_name = load_gesture_module(
                g_file,
                gesture_cfg.get("config", {})
            )

        # 2. Фільтри
        filter_cfg = binding_cfg.get("filter", {})
        self.cooldown = filter_cfg.get("cooldown", 0.5)
        self.trigger_on = filter_cfg.get("trigger_on", "rising") # "rising" або "continuous"
        self.min_hold = filter_cfg.get("min_hold", 0.0)

        # Стан фільтра
        self.last_trigger_time = 0.0
        self.prev_triggered = False
        self.hold_start_time = 0.0

        # 3. Завантаження дії
        action_cfg = binding_cfg["action"]
        self.action_params = action_cfg.get("config", {})
        self.action_executor, self.action_name = load_action_module(
            action_cfg["file"],
            self.action_params
        )


class Engine:
    def __init__(self, config_path: str = "pipeline.yaml", config_dict: Optional[Dict[str, Any]] = None):
        self.config_path = config_path
        if config_dict is not None:
            self.config = config_dict
        else:
            self.config = self._load_config(config_path)

        self.settings = self.config.get("settings", {})
        self.camera_id = self.settings.get("camera_id", 0)
        self.show_preview = self.settings.get("preview_window", True)
        self.show_landmarks = self.settings.get("show_landmarks", True)
        self.fps_limit = self.settings.get("fps_limit", 30)
        self.test_mode = bool(self.settings.get("test_mode", False))
        self.fire_count = 0
        self.last_fire_time = 0.0

        # Ініціалізація джерела відеопотоку (Source)
        source_cfg = self.config.get("source")
        if source_cfg and "file" in source_cfg:
            self.video_source, self.source_name = load_source_module(
                source_cfg["file"],
                source_cfg.get("config", {})
            )
        else:
            flip_h = bool(self.settings.get("flip_horizontal", True))
            self.video_source = DefaultWebcamSource({
                "camera_id": self.camera_id,
                "flip_horizontal": flip_h
            })
            self.source_name = "webcam_default"

        print(f"[Engine] Джерело відеопотоку: {self.source_name}")

        # Ініціалізація зв'язок
        self.bindings: List[Binding] = []
        required_models = set()
        needs_face_engine = False
        needs_audio_engine = False

        raw_bindings = self.config.get("bindings", [])
        if not raw_bindings:
            print("[Engine] УВАГА: У pipeline.yaml немає активних зв'язок (bindings)!")

        for b_cfg in raw_bindings:
            binding = Binding(b_cfg)
            self.bindings.append(binding)
            for req in binding.requires:
                required_models.add(req)
            if binding.is_virtual:
                if binding.detector.kind == "face":
                    needs_face_engine = True
                elif binding.detector.kind in ("yamnet", "voice"):
                    needs_audio_engine = True

        self.face_engine = None
        if needs_face_engine:
            try:
                from core.face_engine import FaceEngine
                self.face_engine = FaceEngine()
                print("[Engine] Підключено модуль Face ID для мультимодальних зв'язок.")
            except Exception as ex:
                print(f"[Engine] Не вдалося завантажити FaceEngine: {ex}")

        self.audio_mgr = None
        if needs_audio_engine:
            try:
                from core.audio_engine import AudioStreamManager
                self.audio_mgr = AudioStreamManager()
                self.audio_mgr.start()
                print("[Engine] Підключено мікрофонний потік YAMNet/Voice ID для мультимодальних зв'язок.")
            except Exception as ex:
                print(f"[Engine] Не вдалося запустити AudioStreamManager: {ex}")

        print(f"[Engine] Підключено зв'язок: {len(self.bindings)}")
        print(f"[Engine] Необхідні моделі для завантаження: {required_models}")

        # Ініціалізація AI Model Manager (лише з потрібними моделями!)
        self.model_manager = ModelManager(required_models)
        self.visualizer = Visualizer(window_name="Pose/Gesture Pipeline - Studio")
        self.recent_actions: List[str] = []

    def _load_config(self, path: str) -> Dict[str, Any]:
        """Завантажує YAML конфігурацію."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Файл конфігурації не знайдено: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def run(self):
        """Головний цикл обробки відеопотоку."""
        print(f"[Engine] Запуск обробки потоку ({self.source_name})...")

        frame_id = 0
        start_time = time.time()

        print("[Engine] Конвеєр успішно запущено! Натисніть 'q' або ESC для виходу.")

        try:
            while True:
                ret, frame = self.video_source.read()
                if not ret or frame is None:
                    print("[Engine] Потік завершено або кадр не отримано.")
                    break

                # Орієнтація кадру вже коректно налаштована джерелом video_source (webcam за замовчуванням дзеркальна)
                h, w = frame.shape[:2]
                now = time.time()
                timestamp_ms = int((now - start_time) * 1000)

                # Перетворення кольору в RGB для нейромережі
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # 1. Єдиний інференс моделей для всього кадру
                landmarks = self.model_manager.process(frame_rgb, timestamp_ms)

                # 2. Створення єдиного контексту кадру
                ctx = FrameContext(
                    frame=frame,
                    frame_rgb=frame_rgb,
                    timestamp=now,
                    frame_id=frame_id,
                    width=w,
                    height=h,
                    landmarks=landmarks
                )

                active_gestures_hud = []

                # 3. Виклик кожного підключеного детектора та обробка дій
                for binding in self.bindings:
                    if binding.is_virtual:
                        det_result: DetectionResult = binding.detector.evaluate_from_engines(
                            frame, self.face_engine, self.audio_mgr
                        )
                    else:
                        det_result: DetectionResult = binding.detector.process(ctx)

                    # Запис для віджетів HUD
                    active_gestures_hud.append({
                        "name": binding.gesture_name,
                        "triggered": det_result.triggered,
                        "value": det_result.value,
                        "extra": det_result.extra_data
                    })

                    # 4. Фільтрація спрацьовувань (Cooldown, Edge, Hold)
                    should_fire = False

                    if det_result.triggered:
                        # Перевірка Hold часу
                        if binding.min_hold > 0:
                            if binding.hold_start_time == 0.0:
                                binding.hold_start_time = now
                            is_held_enough = (now - binding.hold_start_time) >= binding.min_hold
                        else:
                            is_held_enough = True

                        # Перевірка Rising Edge або Continuous
                        edge_condition = True
                        if binding.trigger_on == "rising":
                            edge_condition = not binding.prev_triggered

                        # Перевірка Cooldown
                        cooldown_ok = (now - binding.last_trigger_time) >= binding.cooldown

                        if is_held_enough and edge_condition and cooldown_ok:
                            should_fire = True
                            binding.last_trigger_time = now
                    else:
                        binding.hold_start_time = 0.0

                    binding.prev_triggered = det_result.triggered

                    # 5. Виконання дії при спрацюванні тригера
                    if should_fire:
                        event = CommandEvent(
                            event_id=binding.id,
                            gesture_name=binding.gesture_name,
                            timestamp=now,
                            value=det_result.value,
                            params=binding.action_params
                        )
                        binding.action_executor.execute(event)

                        action_info = f"[{time.strftime('%H:%M:%S')}] {binding.gesture_name} -> {binding.action_name}"
                        self.recent_actions.append(action_info)
                        self.fire_count += 1
                        self.last_fire_time = now
                        if self.test_mode:
                            print(f"[Test] Жест спрацював! #{self.fire_count}  (значення {det_result.value:.2f})", flush=True)
                        if len(self.recent_actions) > 5:
                            self.recent_actions.pop(0)

                # 6. Відображення прев'ю вікна (якщо увімкнено)
                if self.show_preview:
                    annotated_frame = self.visualizer.draw(
                        frame=frame,
                        landmarks=landmarks,
                        active_gestures=active_gestures_hud,
                        recent_actions=self.recent_actions,
                        show_landmarks=self.show_landmarks,
                        test_info=(dict(fires=self.fire_count, flash=(now - self.last_fire_time) < 0.7)
                                   if self.test_mode else None)
                    )
                    cv2.imshow("Pose/Gesture Pipeline - Studio", annotated_frame)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == 27:  # 'q' або ESC
                        print("[Engine] Зупинка за запитом користувача.")
                        break

                frame_id += 1

        finally:
            if getattr(self, "audio_mgr", None) is not None:
                try:
                    self.audio_mgr.stop()
                except Exception:
                    pass
            self.video_source.release()
            cv2.destroyAllWindows()
            self.model_manager.close()
            print("[Engine] Роботу конвеєра завершено.")
