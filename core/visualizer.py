"""
Візуалізатор для налагодження та зворотного зв'язку (OpenCV HUD).
Малює точки, індикатори активних жестів та статус дій.
"""
import time
import cv2
import numpy as np
from typing import List, Dict, Any, Optional


class Visualizer:
    def __init__(self, window_name: str = "Pose/Gesture Pipeline"):
        self.window_name = window_name
        self.prev_time = time.time()
        self.fps = 0.0

    def draw(
        self,
        frame: np.ndarray,
        landmarks: Dict[str, Any],
        active_gestures: List[Dict[str, Any]],
        recent_actions: List[str],
        show_landmarks: bool = True,
        test_info: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """
        Малює інформаційний шар на кадрі.
        """
        display_frame = frame.copy()
        h, w = display_frame.shape[:2]

        # Розрахунок FPS
        cur_time = time.time()
        dt = cur_time - self.prev_time
        if dt > 0:
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt)
        self.prev_time = cur_time

        # Малюємо точки обличчя, якщо вони є і увімкнено
        if show_landmarks and "face" in landmarks:
            face_result = landmarks["face"]
            if face_result and face_result.face_landmarks:
                for face_lms in face_result.face_landmarks:
                    # Малюємо ключові контури (очі, ніс, овал)
                    # Щоб не перевантажувати екран 478 точками, малюємо точки очей
                    eye_indices = [
                        # Ліве око
                        33, 160, 158, 133, 153, 144, 362, 385, 387, 263, 373, 380,
                        # Зіниці
                        468, 473
                    ]
                    for idx in eye_indices:
                        if idx < len(face_lms):
                            lm = face_lms[idx]
                            cx, cy = int(lm.x * w), int(lm.y * h)
                            cv2.circle(display_frame, (cx, cy), 2, (0, 255, 255), -1)

        # Малюємо точки рук
        if show_landmarks and "hands" in landmarks:
            hands_result = landmarks["hands"]
            if hands_result and hands_result.hand_landmarks:
                for hand_lms in hands_result.hand_landmarks:
                    for lm in hand_lms:
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        cv2.circle(display_frame, (cx, cy), 3, (0, 200, 0), -1)

        # Верхній інформаційний бар (напівпрозорий фон)
        overlay = display_frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 50), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.7, display_frame, 0.3, 0, display_frame)

        # FPS
        cv2.putText(
            display_frame,
            f"FPS: {int(self.fps)}",
            (15, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2
        )

        # Індикатори активних жестів
        offset_x = 120
        for g in active_gestures:
            name = g.get("name", "gesture")
            triggered = g.get("triggered", False)
            val = g.get("value", 0.0)
            extra = g.get("extra", {})

            bg_color = (0, 180, 0) if triggered else ((0, 140, 220) if extra.get("is_primed") else (60, 60, 60))
            text_color = (255, 255, 255)
            stage = extra.get("stage")
            if stage:
                badge_text = f"{name} [{stage}]: {'ACTIVE' if triggered else f'{val:.2f}'}"
            else:
                badge_text = f"{name}: {'ACTIVE' if triggered else f'{val:.2f}'}"

            # Значок бейджика
            text_size = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            badge_w = text_size[0] + 16

            cv2.rectangle(
                display_frame,
                (offset_x, 10),
                (offset_x + badge_w, 40),
                bg_color,
                -1
            )
            cv2.putText(
                display_frame,
                badge_text,
                (offset_x + 8, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                text_color,
                1
            )
            offset_x += badge_w + 10

        # Нижній бар для останніх виконаних дій
        if recent_actions:
            last_action_text = f"Last Action: {recent_actions[-1]}"
            cv2.rectangle(display_frame, (0, h - 35), (w, h), (10, 10, 10), -1)
            cv2.putText(
                display_frame,
                last_action_text,
                (15, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                1
            )

        if test_info:
            self._draw_test_overlay(display_frame, active_gestures, test_info)

        return display_frame

    def _draw_test_overlay(self, img: np.ndarray, gestures: List[Dict[str, Any]], info: Dict[str, Any]):
        """Режим перевірки жесту: смуга схожості, лічильник і спалах при спрацюванні."""
        h, w = img.shape[:2]

        # Смуга поточної схожості / значення (0..1) під верхнім баром
        for g in gestures[:1]:
            val = max(0.0, min(1.0, float(g.get("value", 0.0))))
            triggered = g.get("triggered", False)
            extra = g.get("extra", {})
            is_primed = extra.get("is_primed", False)
            stage = extra.get("stage")

            x0, y0, x1, y1 = 15, 62, w - 15, 86
            cv2.rectangle(img, (x0, y0), (x1, y1), (40, 40, 40), -1)
            fill_x = x0 + int((x1 - x0) * val)

            # Зелений при спрацюванні, жовто-золотий при очікуванні кроку 2 (Пози Б), помаранчевий на кроці 1
            color = (0, 215, 0) if triggered else ((0, 215, 255) if is_primed else (0, 160, 255))
            cv2.rectangle(img, (x0, y0), (fill_x, y1), color, -1)
            cv2.rectangle(img, (x0, y0), (x1, y1), (220, 220, 220), 1)

            label = f"{val:.2f}"
            if stage:
                label = f"{stage}  |  {val:.2f}"
            cv2.putText(img, label, (x0 + 10, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        # Лічильник спрацювань
        cv2.putText(img, f"TEST  triggers: {info.get('fires', 0)}", (15, 108),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Спалах рамки + великий напис одразу після спрацювання
        if info.get("flash"):
            cv2.rectangle(img, (0, 0), (w - 1, h - 1), (0, 220, 0), 10)
            text = f"OK! #{info.get('fires', 0)}"
            size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 2.0, 5)[0]
            cv2.putText(img, text, ((w - size[0]) // 2, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 9)
            cv2.putText(img, text, ((w - size[0]) // 2, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 5)
