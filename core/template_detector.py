"""
Універсальний детектор записаних зразків (.json).
Повна підтримка 6 парадигм жестів:
  1. static     — Статична поза
  2. trajectory — Траєкторія руху у просторі
  3. transition — Трансформація форми (Поза А -> Поза Б)
  4. periodic   — Циклічні коливання (помахати рукою)
  5. hold       — Утримання пози певний час
  6. slider     — Безперервний аналоговий контролер (0.0 .. 1.0)
"""
import math
from typing import Dict, Any, List, Tuple, Optional
import numpy as np

from core.models import FrameContext, DetectionResult
from utils.pose_math import (
    extract_landmarks_array,
    normalize_landmarks,
    compute_similarity,
    resample_trajectory,
    compute_trajectory_similarity,
    count_oscillations
)


class TemplateGestureDetector:
    def __init__(self, template_data: Dict[str, Any], user_config: Dict[str, Any]):
        self.name = template_data.get("name", "unnamed_template")
        self.model_type = template_data.get("model_type", "hands")
        self.gesture_type = template_data.get("gesture_type", "static")

        # Базові конфігурації
        self.threshold = float(user_config.get("threshold", template_data.get("threshold", 0.80)))
        self.tolerance = float(user_config.get("tolerance", template_data.get("tolerance", 0.32)))
        self.time_window = float(user_config.get("time_window", template_data.get("time_window", template_data.get("traj_time", 0.8))))

        # 1 & 2: Точки для static та trajectory
        raw_template = template_data.get("template_points", [])
        self.template_points = np.array(raw_template, dtype=np.float32)

        # 3: Трансформація (Поза А -> Поза Б)
        raw_a = template_data.get("pose_a", [])
        raw_b = template_data.get("pose_b", [])
        self.pose_a = np.array(raw_a, dtype=np.float32) if len(raw_a) > 0 else np.array([])
        self.pose_b = np.array(raw_b, dtype=np.float32) if len(raw_b) > 0 else np.array([])
        self.transition_max_time = float(user_config.get("transition_max_time", template_data.get("transition_max_time", 1.0)))
        self.last_pose_a_time = 0.0

        # 4: Циклічний рух (Periodic)
        self.periodic_axis = template_data.get("axis", "x").lower()
        self.min_cycles = int(user_config.get("min_cycles", template_data.get("min_cycles", 3)))  # зміни напрямку
        self.min_amplitude = float(user_config.get("min_amplitude", template_data.get("min_amplitude", 0.09)))

        # 5: Утримання (Hold)
        self.hold_duration = float(template_data.get("hold_duration", 1.2))
        self.hold_start_time = 0.0
        self.hold_triggered = False

        # 6: Повзунок (Slider)
        self.slider_mode = template_data.get("slider_mode", "height") # "height", "horizontal", "pinch"

        # Буфери історії
        self.history_2d: List[Tuple[float, float, float]] = []  # [(t, x, y)]
        self.history_1d: List[Tuple[float, float]] = []         # [(t, val)]

    def _get_active_landmarks(self, ctx: FrameContext) -> Optional[np.ndarray]:
        """Отримує нормалізовані координати точок за типом моделі."""
        if self.model_type == "hands":
            hands_res = ctx.landmarks.get("hands")
            if hands_res and hands_res.hand_landmarks:
                pts = extract_landmarks_array(hands_res.hand_landmarks[0])
                return normalize_landmarks(pts, "hands")
        elif self.model_type == "face":
            face_res = ctx.landmarks.get("face")
            if face_res and face_res.face_landmarks:
                pts = extract_landmarks_array(face_res.face_landmarks[0])
                return normalize_landmarks(pts, "face")
        elif self.model_type == "pose":
            pose_res = ctx.landmarks.get("pose")
            if pose_res and pose_res.pose_landmarks:
                pts = extract_landmarks_array(pose_res.pose_landmarks[0])
                return normalize_landmarks(pts, "pose")
        return None

    def _get_primary_point(self, ctx: FrameContext) -> Optional[Tuple[float, float]]:
        """Отримує опорну точку (зап'ястя/ніс) для розрахунку траєкторії."""
        if self.model_type == "hands":
            hands_res = ctx.landmarks.get("hands")
            if hands_res and hands_res.hand_landmarks:
                lm = hands_res.hand_landmarks[0][0]
                return (lm.x, lm.y)
        elif self.model_type == "face":
            face_res = ctx.landmarks.get("face")
            if face_res and face_res.face_landmarks:
                lm = face_res.face_landmarks[0][1]
                return (lm.x, lm.y)
        elif self.model_type == "pose":
            pose_res = ctx.landmarks.get("pose")
            if pose_res and pose_res.pose_landmarks:
                lm = pose_res.pose_landmarks[0][16]
                return (lm.x, lm.y)
        return None

    def process(self, ctx: FrameContext) -> DetectionResult:
        now = ctx.timestamp

        # -----------------------------------------------------------------
        # 1. СТАТИЧНА ПОЗА (STATIC)
        # -----------------------------------------------------------------
        if self.gesture_type == "static":
            norm_pts = self._get_active_landmarks(ctx)
            if norm_pts is None or len(self.template_points) == 0:
                return DetectionResult(triggered=False, value=0.0)

            sim, dist = compute_similarity(norm_pts, self.template_points, self.tolerance)
            is_active = sim >= self.threshold
            return DetectionResult(
                triggered=is_active,
                value=sim,
                confidence=sim,
                extra_data={"similarity": sim, "dist": dist}
            )

        # -----------------------------------------------------------------
        # 2. ТРАЄКТОРІЯ У ПРОСТОРІ (TRAJECTORY)
        # -----------------------------------------------------------------
        elif self.gesture_type == "trajectory":
            pt = self._get_primary_point(ctx)
            if pt is None:
                self.history_2d.clear()
                return DetectionResult(triggered=False, value=0.0)

            self.history_2d.append((now, pt[0], pt[1]))
            self.history_2d = [(t, x, y) for (t, x, y) in self.history_2d if (now - t) <= self.time_window]

            if len(self.history_2d) < 5 or len(self.template_points) == 0:
                return DetectionResult(triggered=False, value=0.0)

            raw_path = [(x, y) for (_, x, y) in self.history_2d]
            resampled = resample_trajectory(raw_path, num_samples=len(self.template_points))
            if resampled is None:
                return DetectionResult(triggered=False, value=0.0)

            sim, dist = compute_trajectory_similarity(resampled, self.template_points, self.tolerance)
            is_triggered = sim >= self.threshold
            if is_triggered:
                self.history_2d.clear()

            return DetectionResult(
                triggered=is_triggered,
                value=sim,
                confidence=sim,
                extra_data={"similarity": sim, "dist": dist}
            )

        # -----------------------------------------------------------------
        # 3. ТРАНСФОРМАЦІЯ ФОРМИ (TRANSITION: ПОЗА А -> ПОЗА Б)
        # -----------------------------------------------------------------
        elif self.gesture_type == "transition":
            norm_pts = self._get_active_landmarks(ctx)
            if norm_pts is None or len(self.pose_a) == 0 or len(self.pose_b) == 0:
                return DetectionResult(triggered=False, value=0.0)

            sim_a, _ = compute_similarity(norm_pts, self.pose_a, self.tolerance)
            sim_b, _ = compute_similarity(norm_pts, self.pose_b, self.tolerance)

            # Перевіряємо, чи діє часове вікно перетворення (Поза А була зафіксована)
            is_primed = False
            time_left = 0.0
            if self.last_pose_a_time > 0.0:
                dt = now - self.last_pose_a_time
                if dt <= self.transition_max_time:
                    is_primed = True
                    time_left = max(0.0, self.transition_max_time - dt)
                else:
                    self.last_pose_a_time = 0.0

            # Фіксуємо або оновлюємо позу А (поки користувач тримає початкову позу)
            if sim_a >= self.threshold:
                self.last_pose_a_time = now
                is_primed = True
                time_left = self.transition_max_time

            is_transition = False
            was_primed = is_primed
            if is_primed and self.last_pose_a_time > 0.0:
                dt = now - self.last_pose_a_time
                # Перетворення: бачимо позу Б після пози А
                if sim_b >= self.threshold and 0.05 <= dt <= self.transition_max_time:
                    is_transition = True
                    self.last_pose_a_time = 0.0
                    is_primed = False

            # Значення для шкали прогресу:
            # Якщо Крок 1 (Поза А) пройдено або жест спрацював -> показуємо sim_b (прогрес до Пози Б)
            # Якщо Крок 1 ще не пройдено -> показуємо sim_a (прогрес до Пози А)!
            display_val = sim_b if (is_primed or is_transition) else sim_a
            stage_str = "TRIGGERED!" if is_transition else (f"STEP 2: B ({time_left:.1f}s)" if is_primed else "STEP 1: A (Fist)")

            return DetectionResult(
                triggered=is_transition,
                value=display_val,
                confidence=display_val,
                extra_data={
                    "gesture_type": "transition",
                    "is_primed": is_primed,
                    "stage": stage_str,
                    "time_left": time_left,
                    "sim_a": sim_a,
                    "sim_b": sim_b
                }
            )

        # -----------------------------------------------------------------
        # 4. ЦИКЛІЧНИЙ РУХ / ПОМАХ (PERIODIC)
        # -----------------------------------------------------------------
        elif self.gesture_type == "periodic":
            pt = self._get_primary_point(ctx)
            if pt is None:
                self.history_1d.clear()
                return DetectionResult(triggered=False, value=0.0)

            val = pt[0] if self.periodic_axis == "x" else pt[1]
            self.history_1d.append((now, val))
            self.history_1d = [(t, v) for (t, v) in self.history_1d if (now - t) <= self.time_window]

            if len(self.history_1d) < 8:
                return DetectionResult(triggered=False, value=0.0)

            series = [v for (_, v) in self.history_1d]
            cycles = count_oscillations(series, min_amplitude=self.min_amplitude)
            is_active = cycles >= self.min_cycles

            if is_active:
                self.history_1d.clear()

            return DetectionResult(
                triggered=is_active,
                value=float(min(1.0, cycles / self.min_cycles)),
                confidence=0.90,
                extra_data={"cycles": cycles}
            )

        # -----------------------------------------------------------------
        # 5. УТРИМАННЯ ПОЗИ (HOLD)
        # -----------------------------------------------------------------
        elif self.gesture_type == "hold":
            norm_pts = self._get_active_landmarks(ctx)
            if norm_pts is None or len(self.template_points) == 0:
                self.hold_start_time = 0.0
                self.hold_triggered = False
                return DetectionResult(triggered=False, value=0.0)

            sim, _ = compute_similarity(norm_pts, self.template_points, self.tolerance)

            if sim >= self.threshold:
                if self.hold_start_time == 0.0:
                    self.hold_start_time = now
                    self.hold_triggered = False

                held_time = now - self.hold_start_time
                progress = min(1.0, held_time / self.hold_duration)

                is_fired = (held_time >= self.hold_duration) and (not self.hold_triggered)
                if is_fired:
                    self.hold_triggered = True

                return DetectionResult(
                    triggered=is_fired,
                    value=round(progress, 2),
                    confidence=sim,
                    extra_data={"held_time": round(held_time, 2), "progress": round(progress, 2)}
                )
            else:
                self.hold_start_time = 0.0
                self.hold_triggered = False
                return DetectionResult(triggered=False, value=0.0)

        # -----------------------------------------------------------------
        # 6. АНАЛОГОВИЙ КОНТРОЛЕР / ПОВЗУНОК (SLIDER: 0.0 .. 1.0)
        # -----------------------------------------------------------------
        elif self.gesture_type == "slider":
            norm_val = 0.0
            found = False

            if self.slider_mode == "pinch" and self.model_type == "hands":
                # Дистанція між великим (4) і вказівним (8) пальцями
                hands_res = ctx.landmarks.get("hands")
                if hands_res and hands_res.hand_landmarks:
                    h = hands_res.hand_landmarks[0]
                    p1, p2 = h[4], h[8]
                    dist = math.hypot(p1.x - p2.x, p1.y - p2.y)
                    # 0.03 (стиснуто) до 0.20 (розведено)
                    norm_val = max(0.0, min(1.0, (dist - 0.03) / 0.17))
                    found = True

            else:
                pt = self._get_primary_point(ctx)
                if pt is not None:
                    found = True
                    if self.slider_mode == "horizontal":
                        # X координата (зліва направо 0.0 .. 1.0)
                        norm_val = max(0.0, min(1.0, pt[0]))
                    else:
                        # Y координата: висота руки (вище = ближче до 1.0)
                        # Екранні Y інвертовані (0 зверху, 1 знизу)
                        norm_val = max(0.0, min(1.0, 1.0 - pt[1]))

            return DetectionResult(
                triggered=found,
                value=round(norm_val, 2),
                confidence=1.0 if found else 0.0,
                extra_data={"slider_val": round(norm_val, 2)}
            )

        return DetectionResult(triggered=False, value=0.0)
