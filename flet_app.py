"""
V-Pose Combine Studio — графічний інтерфейс на Flet (Material 3).

Можливості:
  * темна / світла / системна тема + 6 акцентних палітр (миттєве перефарбування)
  * анімований градієнтний фон з «дрейфуючими» світлими плямами
  * скляні (glassmorphism) картки з blur, hover-підйомом та тінями
  * каскадна поява елементів, плавні переходи між вкладками (AnimatedSwitcher)
  * пульсуюча кнопка запуску з живою консоллю конвеєра
  * студія запису 6 парадигм жестів з параметрами прямо в інтерфейсі
  * бібліотека з пошуком/фільтром та видаленням із підтвердженням
  * конструктор зв'язок «жест → дія» з візуальним превʼю
  * гарячі клавіші: Ctrl+1..5 — вкладки, Ctrl+T — зміна теми

Запуск:
    python flet_app.py
    або: python run.py --gui
"""
import os
import re
import sys
import glob
import json
import time
import yaml
import atexit
import random
import colorsys
import threading
import subprocess
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import flet as ft

from core.face_engine import FaceEngine, CROPS_DIR
from core.audio_engine import AudioStreamManager, YamnetDetector, VoiceIdEngine
from core.code_exporter import (
    export_gesture_standalone,
    export_face_id_standalone,
    export_voice_standalone,
    export_multimodal_robot_guard,
    create_robot_guard_package,
    create_export_package,
)

PIPELINE_FILE = "pipeline.yaml"
SETTINGS_FILE = "gui_settings.json"
GESTURES_DIR = os.path.join("modules", "gestures")
RECORDED_DIR = os.path.join("modules", "gestures", "recorded")
ACTIONS_DIR = os.path.join("modules", "actions")

# Валідний мінімальний JPEG data-URI для стабільного рендерингу у Flet 1.0+
BLANK_FRAME_JPEG = (
    "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAIBAQEBAQIBAQECAgICAgQDAgICAgUEBAMEBgUGBgYFBgYGBwkIBgcJBwYGCAsICQoKCgoKBggLDAsKDAkKCgr/2wBDAQICAgICAgUDAwUKBwYHCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgr/wAARCAAQABADASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwD+f+iiigD/2Q=="
)


def frame_to_data_uri(frame: np.ndarray, quality: int = 80) -> Optional[str]:
    """Кодує кадр OpenCV у base64 Data URI для стабільного відображення у Flet."""
    if frame is None or frame.size == 0:
        return None
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return None
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")

PALETTES = [
    ("Бірюза", "#14b8a6"),
    ("Фіалка", "#8b5cf6"),
    ("Троянда", "#f43f5e"),
    ("Бурштин", "#f59e0b"),
    ("Лазур", "#3b82f6"),
    ("Лайм", "#84cc16"),
]

DEFAULT_SETTINGS = {
    "mode": "dark",
    "accent": "#14b8a6",
    "animations": True,
    "orbs": True,
    "collapsed": False,
}

NAV = [
    (ft.Icons.ACCESSIBILITY_NEW_OUTLINED, ft.Icons.ACCESSIBILITY_NEW, "Поза (V-Pose)", "Жести рук і тіла, запис і конвеєр"),
    (ft.Icons.FACE_OUTLINED, ft.Icons.FACE, "Обличчя (Face ID)", "Розпізнавання облич, база користувачів"),
    (ft.Icons.MIC_OUTLINED, ft.Icons.MIC, "Голос (Audio & Voice)", "Детекція звуків YAMNet та Voice ID"),
    (ft.Icons.SMART_TOY_OUTLINED, ft.Icons.SMART_TOY, "Полігон 2FA Робота", "Симулятор 2FA (Обличчя + Голос + Жести)"),
    (ft.Icons.SETTINGS_OUTLINED, ft.Icons.SETTINGS, "Вигляд", "Тема, кольори та ефекти"),
]

PARADIGMS = [
    dict(type="static", choice="1", icon=ft.Icons.BACK_HAND, color="#38bdf8",
         title="Статична поза", desc="Фіксація форми: кулак, палець вгору, «V»."),
    dict(type="trajectory", choice="2", icon=ft.Icons.GESTURE, color="#f97316",
         title="Траєкторія", desc="Рух у просторі: свайп, коло, зигзаг."),
    dict(type="transition", choice="3", icon=ft.Icons.TRANSFORM, color="#a855f7",
         title="Трансформація", desc="Поза А → Поза Б: кулак → розкриті пальці."),
    dict(type="periodic", choice="4", icon=ft.Icons.WAVES, color="#22c55e",
         title="Циклічний рух", desc="Коливання туди-назад: помахати рукою."),
    dict(type="hold", choice="5", icon=ft.Icons.TIMER, color="#eab308",
         title="Утримання", desc="Тримати позу N секунд без руху."),
    dict(type="slider", choice="6", icon=ft.Icons.TUNE, color="#ec4899",
         title="Повзунок 0–100%", desc="Рука як аналоговий регулятор гучності."),
]
PARADIGM_BY_TYPE = {p["type"]: p for p in PARADIGMS}

MODELS = [
    ("hands", "1", "Кисть", ft.Icons.PAN_TOOL_ALT),
    ("face", "2", "Обличчя", ft.Icons.FACE),
    ("pose", "3", "Тіло", ft.Icons.ACCESSIBILITY_NEW),
]

ACTION_LABELS = {
    "keyboard_press": ("Клавіатура", ft.Icons.KEYBOARD),
    "sound_beep": ("Звуковий сигнал", ft.Icons.VOLUME_UP),
    "slider_display": ("Аналоговий індикатор", ft.Icons.TUNE),
    "monitor_only": ("Лише моніторинг", ft.Icons.VISIBILITY),
}


# ----------------------------------------------------------------------------
# Допоміжні функції
# ----------------------------------------------------------------------------
def op(opacity: float, color) -> str:
    return ft.Colors.with_opacity(opacity, color)


def shift_hue(hex_color: str, delta: float) -> str:
    """Повертає колір з повернутим відтінком (для другого кольору градієнта)."""
    try:
        hex_color = str(hex_color).lstrip("#")
        if len(hex_color) != 6:
            return "#64748b"
        r, g, b = (int(hex_color[i:i + 2], 16) / 255 for i in (0, 2, 4))
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        r, g, b = colorsys.hsv_to_rgb((h + delta) % 1.0, s, v)
        return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))
    except Exception:
        return "#64748b"


def read_cfg() -> Dict[str, Any]:
    if not os.path.exists(PIPELINE_FILE):
        return {"version": "1.0", "settings": {"camera_id": 0, "preview_window": True,
                                               "show_landmarks": True, "flip_horizontal": True, "fps_limit": 30}, "bindings": []}
    with open(PIPELINE_FILE, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def write_cfg(cfg: Dict[str, Any]):
    with open(PIPELINE_FILE, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True)


def scan_gestures() -> List[Dict[str, Any]]:
    """Збирає .py модулі та записані .json зразки."""
    items: List[Dict[str, Any]] = []
    for fpath in sorted(glob.glob(os.path.join(GESTURES_DIR, "*.py"))):
        if os.path.basename(fpath).startswith("__"):
            continue
        items.append(dict(kind="py", path=fpath.replace("\\", "/"), file=os.path.basename(fpath),
                          name=os.path.splitext(os.path.basename(fpath))[0]))
    for fpath in sorted(glob.glob(os.path.join(RECORDED_DIR, "*.json"))):
        info = dict(kind="json", path=fpath.replace("\\", "/"), file=os.path.basename(fpath),
                    name=os.path.splitext(os.path.basename(fpath))[0],
                    gesture_type="static", model_type="?", threshold=None, created=None)
        try:
            with open(fpath, "r", encoding="utf-8") as jf:
                data = json.load(jf)
            info.update(name=data.get("name", info["name"]),
                        gesture_type=data.get("gesture_type", "static"),
                        model_type=data.get("model_type", "?"),
                        threshold=data.get("threshold"),
                        created=data.get("created_at"))
        except Exception:
            pass
        items.append(info)
    return items


def scan_actions() -> List[Dict[str, Any]]:
    items = []
    for fpath in sorted(glob.glob(os.path.join(ACTIONS_DIR, "*.py"))):
        base = os.path.splitext(os.path.basename(fpath))[0]
        if base.startswith("__"):
            continue
        label, icon = ACTION_LABELS.get(base, (base, ft.Icons.BOLT))
        items.append(dict(path=fpath.replace("\\", "/"), name=base, label=label, icon=icon))
    return items


def fmt_uptime(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 3600:02d}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"


# ----------------------------------------------------------------------------
# Головний застосунок
# ----------------------------------------------------------------------------
class PoseStudioApp:
    def __init__(self, page: ft.Page):
        self.page = page
        self.alive = True
        self.settings = self._load_settings()
        self.animations: bool = bool(self.settings["animations"])

        self.tab = 0
        self.nav_items: List[ft.Container] = []
        self.orbs: List[ft.Container] = []

        # Процеси та журнали
        self.proc: Optional[subprocess.Popen] = None
        self.proc_started = 0.0
        self.log_lines: deque = deque(maxlen=400)
        self.log_view: Optional[ft.ListView] = None
        self.rec_proc: Optional[subprocess.Popen] = None

        # Стан живої перевірки жесту
        self.test_proc: Optional[subprocess.Popen] = None
        self.test_target: Optional[str] = None
        self.test_name = ""
        self.test_hits = 0
        self.test_last = ""
        self.test_banner: Optional[ft.Container] = None
        self.test_banner_title: Optional[ft.Text] = None
        self.test_banner_sub: Optional[ft.Text] = None

        # Стан вкладки запису
        self.rec_type = "transition"
        self.rec_model = "hands"
        self.rec_params: Dict[str, Any] = dict(
            threshold=0.80, trans_time=0.9, hold_seconds=1.5, min_cycles=3,
            axis_choice="1", slider_choice="1", traj_time=1.0, period_window=1.0, min_amplitude=0.09,
        )
        self.rec_name = ""

        # Стан бібліотеки та зв'язок
        self.lib_query = ""
        self.lib_filter = "all"
        self.bld: Dict[str, Any] = dict(gesture=None, action=None, key="space", cooldown=0.8)

        # Посилання на динамічні віджети
        self.power_btn: Optional[ft.Container] = None
        self.power_ring: Optional[ft.ProgressRing] = None
        self.run_title: Optional[ft.Text] = None
        self.run_sub: Optional[ft.Text] = None
        self.uptime_text: Optional[ft.Text] = None
        self.status_dot: Optional[ft.Container] = None
        self.status_text: Optional[ft.Text] = None
        self.status_pill: Optional[ft.Container] = None
        self.theme_icon_switcher: Optional[ft.AnimatedSwitcher] = None
        self.title_text: Optional[ft.Text] = None
        self.subtitle_text: Optional[ft.Text] = None
        self.rec_btn: Optional[ft.Container] = None
        self.rec_status: Optional[ft.Text] = None
        self.rec_ring: Optional[ft.ProgressRing] = None
        self.rec_cards: Dict[str, ft.Container] = {}
        self.rec_params_switcher: Optional[ft.AnimatedSwitcher] = None
        self.model_seg: Optional[ft.SegmentedButton] = None
        self.model_lock_info: Optional[ft.Container] = None
        self.rec_camera_hint: Optional[ft.Text] = None
        self.lib_list: Optional[ft.Column] = None
        self.lib_count: Optional[ft.Text] = None
        self.bind_list: Optional[ft.Column] = None
        self.flow_row: Optional[ft.Row] = None
        self.key_field: Optional[ft.TextField] = None

        # Під-вкладка V-Pose (0: Конвеєр, 1: Запис, 2: Бібліотека, 3: Зв'язки)
        self.vpose_subtab = 0

        # Модуль Face ID + Вбудоване прев'ю камери + Якість кадру + Anti-Spoofing (Liveness)
        self.face_engine: Optional[FaceEngine] = None
        try:
            self.face_engine = FaceEngine()
        except Exception as ex:
            print(f"[FaceEngine] Попередження: {ex}")
        self.face_proc: Optional[subprocess.Popen] = None
        self.face_reg_name = ""
        self.face_status_text: Optional[ft.Text] = None
        self.face_profiles_list: Optional[ft.Column] = None
        self.face_live_badge: Optional[ft.Container] = None
        self.face_live_title: Optional[ft.Text] = None
        self.face_live_sub: Optional[ft.Text] = None
        self.face_scan_btn: Optional[ft.Container] = None
        self.face_reg_ring: Optional[ft.ProgressRing] = None
        self.face_reg_preview_img: Optional[ft.Image] = None
        self.face_reg_preview_box: Optional[ft.Container] = None
        self.face_embedded_running: bool = False
        self.face_preview_img: Optional[ft.Image] = None
        self.face_quality_badge: Optional[ft.Container] = None
        self.face_quality_text: Optional[ft.Text] = None
        self.face_light_pill: Optional[ft.Container] = None
        self.face_dist_pill: Optional[ft.Container] = None
        self.face_liveness_required: bool = True
        self.face_liveness_bar: Optional[ft.ProgressBar] = None
        self.face_liveness_text: Optional[ft.Text] = None

        # Модуль Audio & Voice + 16-смугова жива осцилограма
        self.audio_mgr: Optional[AudioStreamManager] = None
        try:
            self.audio_mgr = AudioStreamManager()
        except Exception as ex:
            print(f"[AudioStreamManager] Попередження: {ex}")
        self.voice_reg_name = ""
        self.audio_vu_bar: Optional[ft.ProgressBar] = None
        self.audio_db_text: Optional[ft.Text] = None
        self.audio_events_col: Optional[ft.Column] = None
        self.voice_speaker_badge: Optional[ft.Container] = None
        self.voice_speaker_title: Optional[ft.Text] = None
        self.voice_speaker_sub: Optional[ft.Text] = None
        self.voice_profiles_list: Optional[ft.Column] = None
        self.audio_toggle_btn: Optional[ft.Container] = None
        self.voice_rec_btn: Optional[ft.Container] = None
        self.voice_rec_ring: Optional[ft.ProgressRing] = None

        # Експрес-перевірка звуку та голосу (на вимогу) + 16 смуг спектра
        self.voice_spot_duration: int = 3
        self.voice_spot_btn: Optional[ft.Container] = None
        self.voice_spot_ring: Optional[ft.ProgressRing] = None
        self.voice_spot_progress_bar: Optional[ft.ProgressBar] = None
        self.voice_spot_vu_bar: Optional[ft.ProgressBar] = None
        self.voice_spot_status_text: Optional[ft.Text] = None
        self.voice_spot_result_card: Optional[ft.Container] = None
        self.voice_waveform_bars: List[ft.Container] = []
        self.voice_waveform_row: Optional[ft.Row] = None
        self.voice_is_checking: bool = False
        self.last_spot_result: Optional[Dict[str, Any]] = None

        # Тестовий полігон 2FA Робота (Симулятор у додатку)
        self.sim_running: bool = False
        self.sim_state: str = "LOCKED"  # LOCKED -> FACE_OK -> ARMED
        self.sim_owner: Optional[str] = None
        self.sim_face_ok: bool = False
        self.sim_liveness_ok: bool = False
        self.sim_voice_ok: bool = False
        self.sim_voice_name: Optional[str] = None
        self.sim_last_gesture: str = "Немає"
        self.sim_robot_x: float = 0.50   # 0.0 .. 1.0
        self.sim_robot_y: float = 0.50   # 0.0 .. 1.0
        self.sim_robot_cmd: str = "ОЧІКУВАННЯ АВТОРИЗАЦІЇ (LOCKED)"
        self.sim_preview_img: Optional[ft.Image] = None
        self.sim_cam_btn: Optional[ft.Container] = None
        self.sim_voice_btn: Optional[ft.Container] = None
        self.sim_status_banner: Optional[ft.Container] = None
        self.sim_step1_card: Optional[ft.Container] = None
        self.sim_step2_card: Optional[ft.Container] = None
        self.sim_step3_card: Optional[ft.Container] = None
        self.sim_arena_bot: Optional[ft.Container] = None
        self.sim_cmd_text: Optional[ft.Text] = None

        self._pulse_on = False
        self._compute_colors()
        self._setup_page()
        self._build_ui()

        atexit.register(self._cleanup)
        threading.Thread(target=self._ticker, daemon=True).start()
        threading.Thread(target=self._audio_ui_ticker, daemon=True).start()

    # ------------------------------------------------------------------
    # Налаштування, тема, сторінка
    # ------------------------------------------------------------------
    def _load_settings(self) -> Dict[str, Any]:
        data = dict(DEFAULT_SETTINGS)
        try:
            if os.path.exists(SETTINGS_FILE):
                with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                    data.update(json.load(f))
        except Exception:
            pass
        return data

    def _save_settings(self):
        try:
            with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
                json.dump(self.settings, f, indent=2)
        except Exception:
            pass

    def _compute_colors(self):
        self.c1: str = self.settings["accent"]
        self.c2: str = shift_hue(self.c1, 0.12)

    def _is_dark(self) -> bool:
        mode = self.settings["mode"]
        if mode == "system":
            try:
                return self.page.platform_brightness == ft.Brightness.DARK
            except Exception:
                return True
        return mode == "dark"

    def _theme_mode(self) -> ft.ThemeMode:
        return {"dark": ft.ThemeMode.DARK, "light": ft.ThemeMode.LIGHT,
                "system": ft.ThemeMode.SYSTEM}[self.settings["mode"]]

    def _setup_page(self):
        p = self.page
        p.title = "V-Pose Combine Studio"
        p.padding = 0
        p.spacing = 0
        p.theme_mode = self._theme_mode()
        p.theme = ft.Theme(color_scheme_seed=self.c1, use_material3=True)
        p.dark_theme = ft.Theme(color_scheme_seed=self.c1, use_material3=True)
        p.on_keyboard_event = self._on_key
        try:
            p.window.width = 1320
            p.window.height = 860
            p.window.min_width = 980
            p.window.min_height = 640
        except Exception:
            pass

    def _bg_colors(self) -> List[str]:
        if self._is_dark():
            return ["#090d16", "#0c111d", "#070a12"]
        return ["#f8fafc", "#f1f5f9", "#e2e8f0"]

    def _on_key(self, e: ft.KeyboardEvent):
        if not e.ctrl:
            return
        key = str(e.key).upper()
        if key in ("1", "2", "3", "4", "5"):
            self._goto(int(key) - 1)
        elif key == "T":
            self._set_mode("light" if self._is_dark() else "dark")
        elif key == "E":
            self._open_export_modal()
        elif key == "H":
            self._open_help_modal()

    # ------------------------------------------------------------------
    # Безпечні оновлення (потоки / незмонтовані контроли)
    # ------------------------------------------------------------------
    @staticmethod
    def _safe(ctrl: Optional[ft.Control]):
        if ctrl is None:
            return
        try:
            ctrl.update()
        except Exception:
            pass

    def _page_update(self):
        try:
            self.page.update()
        except Exception:
            pass

    def _toast(self, message: str, kind: str = "info"):
        icon = {"ok": ft.Icons.CHECK_CIRCLE, "err": ft.Icons.ERROR_OUTLINE}.get(kind, ft.Icons.INFO_OUTLINE)
        bg = {"ok": "#15803d", "err": "#b91c1c"}.get(kind, "#334155")
        try:
            self.page.show_dialog(ft.SnackBar(
                content=ft.Row(tight=True, spacing=10, controls=[
                    ft.Icon(icon, color="white", size=20),
                    ft.Text(message, color="white", weight=ft.FontWeight.W_500)]),
                bgcolor=bg, behavior=ft.SnackBarBehavior.FLOATING,
                shape=ft.RoundedRectangleBorder(radius=14), duration=3500))
        except Exception:
            print(f"[UI] {message}")

    def _confirm(self, title: str, text: str, yes_label: str, on_yes):
        def close(_):
            self.page.pop_dialog()

        def yes(_):
            self.page.pop_dialog()
            on_yes()

        self.page.show_dialog(ft.AlertDialog(
            modal=True,
            shape=ft.RoundedRectangleBorder(radius=22),
            title=ft.Row(spacing=10, controls=[
                ft.Icon(ft.Icons.WARNING_AMBER_ROUNDED, color="#f59e0b"), ft.Text(title)]),
            content=ft.Text(text),
            actions=[ft.TextButton("Скасувати", on_click=close),
                     ft.FilledButton(yes_label, icon=ft.Icons.DELETE_OUTLINE,
                                     style=ft.ButtonStyle(bgcolor="#dc2626", color="white"), on_click=yes)],
            actions_alignment=ft.MainAxisAlignment.END))

    # ------------------------------------------------------------------
    # Візуальні примітиви
    # ------------------------------------------------------------------
    def _anim(self, ms: int = 250, curve=ft.AnimationCurve.EASE_OUT) -> ft.Animation:
        return ft.Animation(ms, curve)

    def _shadow(self, lifted: bool = False, color: Optional[str] = None):
        return ft.BoxShadow(
            blur_radius=14 if lifted else 6, spread_radius=0,
            color=op(0.18 if lifted else 0.06, color or self.c1),
            offset=ft.Offset(0, 4 if lifted else 1))

    def _glass_border(self, lifted: bool = False, color: Optional[str] = None):
        if lifted:
            return ft.Border.all(1.2, op(0.55, color or self.c1))
        return ft.Border.all(1, op(0.10, ft.Colors.ON_SURFACE))

    def _glass(self, content: ft.Control, padding=16, radius=10, hover=True, blur=True,
               on_click=None, tint: Optional[str] = None, **kwargs) -> ft.Container:
        def on_hover(e):
            lifted = str(e.data).lower() == "true"
            e.control.offset = ft.Offset(0, -0.010) if lifted else ft.Offset(0, 0)
            e.control.shadow = self._shadow(lifted, tint)
            e.control.border = self._glass_border(lifted, tint)
            self._safe(e.control)

        return ft.Container(
            content=content, padding=padding, border_radius=radius,
            bgcolor=op(0.60 if self._is_dark() else 0.85, ft.Colors.SURFACE_CONTAINER_LOW),
            blur=ft.Blur(10, 10) if blur else None,
            clip_behavior=ft.ClipBehavior.ANTI_ALIAS,
            border=self._glass_border(False), shadow=self._shadow(False, tint),
            animate=self._anim(200), animate_offset=self._anim(180, ft.AnimationCurve.EASE_OUT_CUBIC),
            on_hover=on_hover if (hover and self.animations) else None,
            on_click=on_click, **kwargs)

    def _gradient_text(self, value: str, size=24, weight=ft.FontWeight.W_800) -> ft.Control:
        try:
            return ft.ShaderMask(
                shader=ft.LinearGradient(begin=ft.Alignment.CENTER_LEFT, end=ft.Alignment.CENTER_RIGHT,
                                         colors=[self.c1, self.c2]),
                blend_mode=ft.BlendMode.SRC_IN,
                content=ft.Text(value, size=size, weight=weight, color="white"))
        except Exception:
            return ft.Text(value, size=size, weight=weight, color=self.c1)

    def _icon_badge(self, icon, color: str, size=40, icon_size=20) -> ft.Container:
        return ft.Container(
            width=size, height=size, border_radius=8, alignment=ft.Alignment.CENTER,
            clip_behavior=ft.ClipBehavior.ANTI_ALIAS,
            gradient=ft.LinearGradient(begin=ft.Alignment.TOP_LEFT, end=ft.Alignment.BOTTOM_RIGHT,
                                       colors=[color, shift_hue(color, 0.08)]),
            shadow=ft.BoxShadow(blur_radius=8, color=op(0.30, color), offset=ft.Offset(0, 2)),
            content=ft.Icon(icon, color="white", size=icon_size))

    def _pill(self, text: str, color: str, icon=None) -> ft.Container:
        row: List[ft.Control] = []
        if icon:
            row.append(ft.Icon(icon, size=12, color=color))
        row.append(ft.Text(text, size=11, weight=ft.FontWeight.W_600, color=color, font_family="Consolas"))
        return ft.Container(
            content=ft.Row(tight=True, spacing=4, controls=row),
            padding=ft.Padding.symmetric(horizontal=8, vertical=3), border_radius=6,
            bgcolor=op(0.12, color), border=ft.Border.all(1, op(0.28, color)))

    def _gradient_button(self, label: str, icon, on_click, colors: Optional[List[str]] = None,
                         height=46) -> ft.Container:
        colors = colors or [self.c1, self.c2]

        def on_hover(e):
            hot = str(e.data).lower() == "true"
            e.control.scale = 1.02 if hot else 1
            self._safe(e.control)

        return ft.Container(
            height=height, padding=ft.Padding.symmetric(horizontal=20), border_radius=8,
            alignment=ft.Alignment.CENTER, clip_behavior=ft.ClipBehavior.ANTI_ALIAS,
            gradient=ft.LinearGradient(begin=ft.Alignment.CENTER_LEFT, end=ft.Alignment.CENTER_RIGHT, colors=colors),
            shadow=ft.BoxShadow(blur_radius=12, color=op(0.30, colors[0]), offset=ft.Offset(0, 4)),
            animate=self._anim(200), animate_scale=self._anim(180, ft.AnimationCurve.EASE_OUT),
            animate_opacity=self._anim(200),
            on_hover=on_hover if self.animations else None, on_click=on_click,
            content=ft.Row(tight=True, alignment=ft.MainAxisAlignment.CENTER, spacing=8, controls=[
                ft.Icon(icon, color="white", size=18),
                ft.Text(label, color="white", size=14, weight=ft.FontWeight.W_700)]))

    def _section_title(self, text: str, icon=None, subtitle: Optional[str] = None) -> ft.Control:
        left: List[ft.Control] = []
        if icon:
            left.append(ft.Icon(icon, color=self.c1, size=22))
        col = [ft.Text(text, size=16, weight=ft.FontWeight.W_700)]
        if subtitle:
            col.append(ft.Text(subtitle, size=12, color=ft.Colors.ON_SURFACE_VARIANT))
        left.append(ft.Column(spacing=1, tight=True, controls=col))
        return ft.Row(spacing=10, controls=left)

    def _guide_banner(self, steps: List[Tuple[str, str, str]]) -> ft.Control:
        """Візуальний покроковий банер-підказка (1 -> 2 -> 3) у верхній частині вкладки."""
        cards: List[ft.Control] = []
        for idx, (num, title, desc) in enumerate(steps):
            badge = ft.Container(
                width=26, height=26, border_radius=13, alignment=ft.Alignment.CENTER,
                gradient=ft.LinearGradient(begin=ft.Alignment.TOP_LEFT, end=ft.Alignment.BOTTOM_RIGHT,
                                           colors=[self.c1, self.c2]),
                content=ft.Text(num, size=12, weight=ft.FontWeight.W_800, color="white"))
            step_item = ft.Container(
                col={"xs": 12, "md": 4},
                padding=ft.Padding.symmetric(horizontal=12, vertical=8),
                border_radius=8,
                bgcolor=op(0.06, self.c1),
                border=ft.Border.all(1, op(0.18, self.c1)),
                content=ft.Row(spacing=10, vertical_alignment=ft.CrossAxisAlignment.START, controls=[
                    badge,
                    ft.Column(spacing=2, expand=True, tight=True, controls=[
                        ft.Text(title, size=12, weight=ft.FontWeight.W_700),
                        ft.Text(desc, size=11, color=ft.Colors.ON_SURFACE_VARIANT)])]))
            cards.append(step_item)
        return ft.ResponsiveRow(spacing=10, run_spacing=8, controls=cards)

    # ------------------------------------------------------------------
    # Каскадна поява
    # ------------------------------------------------------------------
    def _prepare_enter(self, controls: List[ft.Control]):
        if not self.animations:
            return
        for c in controls:
            c.opacity = 0
            c.offset = ft.Offset(0, 0.035)
            c.animate_opacity = ft.Animation(420, ft.AnimationCurve.EASE_OUT)
            c.animate_offset = ft.Animation(520, ft.AnimationCurve.EASE_OUT_CUBIC)

    def _play_enter(self, controls: List[ft.Control], delay: float = 0.07):
        if not self.animations:
            return

        def run():
            time.sleep(0.12)
            for c in controls:
                if not self.alive:
                    return
                c.opacity = 1
                c.offset = ft.Offset(0, 0)
                self._safe(c)
                time.sleep(delay)

        threading.Thread(target=run, daemon=True).start()

    # ------------------------------------------------------------------
    # Каркас інтерфейсу
    # ------------------------------------------------------------------
    def _build_ui(self):
        w = self.page.width or 1300
        h = self.page.height or 800
        orb_specs = [(self.c1, 400, 0.05, 0.05), (self.c2, 340, 0.60, 0.40), ("#38bdf8", 280, 0.30, 0.65)]
        self.orbs = []
        for color, size, fx, fy in orb_specs:
            self.orbs.append(ft.Container(
                width=size, height=size, border_radius=size / 2, left=w * fx, top=h * fy,
                visible=bool(self.settings["orbs"]),
                animate_position=ft.Animation(14000, ft.AnimationCurve.EASE_IN_OUT),
                gradient=ft.RadialGradient(colors=[op(0.12 if self._is_dark() else 0.08, color), op(0, color)])))

        self.bg = ft.Container(
            left=0, top=0, right=0, bottom=0, animate=ft.Animation(800, ft.AnimationCurve.EASE_IN_OUT),
            gradient=ft.LinearGradient(begin=ft.Alignment.TOP_LEFT, end=ft.Alignment.BOTTOM_RIGHT,
                                       colors=self._bg_colors()))

        self.sidebar = self._build_sidebar()
        header = self._build_header()
        self.switcher = ft.AnimatedSwitcher(
            content=ft.Container(), transition=ft.AnimatedSwitcherTransition.FADE,
            duration=380, reverse_duration=160,
            switch_in_curve=ft.AnimationCurve.EASE_OUT_CUBIC, switch_out_curve=ft.AnimationCurve.EASE_IN,
            expand=True)

        body = ft.Container(
            left=0, top=0, right=0, bottom=0,
            content=ft.Row(expand=True, spacing=0, controls=[
                self.sidebar,
                ft.Column(expand=True, spacing=0, controls=[
                    header,
                    ft.Container(content=self.switcher, expand=True,
                                 clip_behavior=ft.ClipBehavior.ANTI_ALIAS,
                                 padding=ft.Padding(24, 6, 24, 20))])]))

        def on_resize(e):
            pw = self.page.width or 1200
            ph = self.page.height or 800
            for orb, (_, _, fx, fy) in zip(self.orbs, orb_specs):
                orb.left = pw * fx
                orb.top = ph * fy
                self._safe(orb)

        self.page.on_resized = on_resize
        self.page.controls.clear()
        self.page.add(ft.Stack(expand=True, controls=[self.bg, *self.orbs, body]))
        self._show_tab(self.tab)
        self._refresh_status()

    # --- бічна панель ------------------------------------------------------
    def _build_sidebar(self) -> ft.Container:
        collapsed = bool(self.settings["collapsed"])
        self.nav_items = [self._nav_item(i) for i in range(len(NAV))]
        self.brand_label = ft.Column(
            spacing=0, tight=True, visible=not collapsed,
            controls=[self._gradient_text("V-Pose", 22), ft.Text("Combine Studio", size=11,
                                                                 color=ft.Colors.ON_SURFACE_VARIANT)])
        brand = ft.Row(spacing=12, controls=[
            self._icon_badge(ft.Icons.VIDEOCAM, self.c1, size=44, icon_size=24), self.brand_label])

        self.collapse_icon = ft.Icon(ft.Icons.KEYBOARD_DOUBLE_ARROW_LEFT, size=20,
                                     color=ft.Colors.ON_SURFACE_VARIANT)
        self.collapse_btn = ft.Container(
            content=ft.Row(spacing=12, controls=[self.collapse_icon]), padding=10, border_radius=8, ink=True,
            tooltip="Згорнути / розгорнути панель", on_click=self._toggle_sidebar)
        self._apply_collapse_icon()

        return ft.Container(
            width=76 if collapsed else 238, animate=ft.Animation(380, ft.AnimationCurve.EASE_OUT_CUBIC),
            padding=ft.Padding(12, 18, 12, 14), bgcolor=op(0.5, ft.Colors.SURFACE_CONTAINER_LOW),
            blur=ft.Blur(18, 18), border=ft.Border(right=ft.BorderSide(1, op(0.10, ft.Colors.ON_SURFACE))),
            content=ft.Column(spacing=4, controls=[
                ft.Container(content=brand, padding=ft.Padding(2, 0, 0, 14)),
                *self.nav_items, ft.Container(expand=True), self.collapse_btn]))

    def _apply_collapse_icon(self):
        self.collapse_icon.icon = (ft.Icons.KEYBOARD_DOUBLE_ARROW_RIGHT if self.settings["collapsed"]
                                   else ft.Icons.KEYBOARD_DOUBLE_ARROW_LEFT)

    def _nav_item(self, index: int) -> ft.Container:
        icon, sel_icon, label, _ = NAV[index]
        selected = index == self.tab
        collapsed = bool(self.settings["collapsed"])
        icon_ctrl = ft.Icon(sel_icon if selected else icon, size=20,
                            color=self.c1 if selected else ft.Colors.ON_SURFACE_VARIANT,
                            animate_size=self._anim(200))
        text_ctrl = ft.Text(label, size=13, weight=ft.FontWeight.W_700 if selected else ft.FontWeight.W_500,
                            color=ft.Colors.ON_SURFACE if selected else ft.Colors.ON_SURFACE_VARIANT,
                            visible=not collapsed)

        def on_hover(e):
            if index == self.tab:
                return
            hot = str(e.data).lower() == "true"
            e.control.bgcolor = op(0.08, ft.Colors.ON_SURFACE) if hot else None
            e.control.scale = 1.02 if hot else 1
            self._safe(e.control)

        return ft.Container(
            data=dict(icon=icon_ctrl, text=text_ctrl), tooltip=f"{label}  (Ctrl+{index + 1})",
            padding=ft.Padding.symmetric(horizontal=12, vertical=10), border_radius=8,
            bgcolor=op(0.14, self.c1) if selected else None,
            border=ft.Border(left=ft.BorderSide(3, self.c1 if selected else "transparent")),
            animate=self._anim(200), animate_scale=self._anim(180),
            on_hover=on_hover, on_click=lambda e, i=index: self._goto(i),
            content=ft.Row(spacing=12, controls=[icon_ctrl, text_ctrl]))

    def _toggle_sidebar(self, e):
        self.settings["collapsed"] = not self.settings["collapsed"]
        collapsed = self.settings["collapsed"]
        self.sidebar.width = 76 if collapsed else 238
        self.brand_label.visible = not collapsed
        for item in self.nav_items:
            item.data["text"].visible = not collapsed
        self._apply_collapse_icon()
        self._save_settings()
        self._page_update()

    # --- шапка --------------------------------------------------------------
    def _build_header(self) -> ft.Container:
        self.title_text = ft.Text(NAV[self.tab][2], size=22, weight=ft.FontWeight.W_800)
        self.subtitle_text = ft.Text(NAV[self.tab][3], size=12, color=ft.Colors.ON_SURFACE_VARIANT)

        self.status_dot = ft.Container(width=8, height=8, border_radius=4, bgcolor="#64748b",
                                       animate=self._anim(500), animate_opacity=ft.Animation(900, ft.AnimationCurve.EASE_IN_OUT))
        self.status_text = ft.Text("Зупинено", size=11, weight=ft.FontWeight.W_700, font_family="Consolas")
        self.status_pill = ft.Container(
            padding=ft.Padding.symmetric(horizontal=12, vertical=6), border_radius=6,
            bgcolor=op(0.5, ft.Colors.SURFACE_CONTAINER), border=ft.Border.all(1, op(0.12, ft.Colors.ON_SURFACE)),
            animate=self._anim(300),
            content=ft.Row(tight=True, spacing=6, controls=[self.status_dot, self.status_text]))

        self.theme_icon_switcher = ft.AnimatedSwitcher(
            content=self._theme_icon(), transition=ft.AnimatedSwitcherTransition.ROTATION, duration=350)
        theme_btn = ft.Container(
            content=self.theme_icon_switcher, padding=9, border_radius=8, ink=True,
            bgcolor=op(0.5, ft.Colors.SURFACE_CONTAINER), border=ft.Border.all(1, op(0.12, ft.Colors.ON_SURFACE)),
            tooltip="Світла / темна тема (Ctrl+T)", on_click=lambda e: self._set_mode("light" if self._is_dark() else "dark"))

        help_btn = ft.Container(
            content=ft.Row(spacing=6, tight=True, controls=[
                ft.Icon(ft.Icons.HELP_OUTLINE, size=18, color=ft.Colors.ON_SURFACE),
                ft.Text("Підказки", size=12, weight=ft.FontWeight.W_600)]),
            padding=ft.Padding.symmetric(horizontal=11, vertical=7), border_radius=8, ink=True,
            bgcolor=op(0.5, ft.Colors.SURFACE_CONTAINER), border=ft.Border.all(1, op(0.14, ft.Colors.ON_SURFACE)),
            tooltip="Інтерактивний довідник та інструкції (Ctrl+H)", on_click=self._open_help_modal)

        export_btn = ft.Container(
            content=ft.Row(spacing=6, tight=True, controls=[
                ft.Icon(ft.Icons.CODE, size=18, color=self.c1),
                ft.Text("Експорт коду", size=12, weight=ft.FontWeight.W_700, color=self.c1)]),
            padding=ft.Padding.symmetric(horizontal=12, vertical=7), border_radius=8, ink=True,
            bgcolor=op(0.12, self.c1), border=ft.Border.all(1, op(0.3, self.c1)),
            tooltip="Експортувати код керування та моделі (Ctrl+E)", on_click=self._open_export_modal)

        return ft.Container(
            padding=ft.Padding(30, 22, 30, 12),
            content=ft.Row(alignment=ft.MainAxisAlignment.SPACE_BETWEEN, controls=[
                ft.Column(spacing=2, tight=True, controls=[self.title_text, self.subtitle_text]),
                ft.Row(spacing=10, controls=[help_btn, export_btn, self.status_pill, theme_btn])]))

    def _theme_icon(self) -> ft.Icon:
        dark = self._is_dark()
        return ft.Icon(ft.Icons.DARK_MODE if dark else ft.Icons.LIGHT_MODE, size=22,
                       color="#a5b4fc" if dark else "#f59e0b", key="dark" if dark else "light")

    def _set_mode(self, mode: str):
        self.settings["mode"] = mode
        self.page.theme_mode = self._theme_mode()
        self.bg.gradient = ft.LinearGradient(begin=ft.Alignment.TOP_LEFT, end=ft.Alignment.BOTTOM_RIGHT,
                                             colors=self._bg_colors())
        strength = 0.12 if self._is_dark() else 0.08
        for orb, color in zip(self.orbs, [self.c1, self.c2, "#38bdf8"]):
            orb.gradient = ft.RadialGradient(colors=[op(strength, color), op(0, color)])
        if self.theme_icon_switcher:
            self.theme_icon_switcher.content = self._theme_icon()
        self._save_settings()
        self._page_update()
        if self.tab in (3, 4):
            self._show_tab(self.tab)

    def _set_accent(self, color: str):
        self.settings["accent"] = color
        self._compute_colors()
        self.page.theme = ft.Theme(color_scheme_seed=self.c1, use_material3=True)
        self.page.dark_theme = ft.Theme(color_scheme_seed=self.c1, use_material3=True)
        self._save_settings()
        self._build_ui()
        self._toast("Акцентний колір змінено", "ok")

    # ------------------------------------------------------------------
    # Інтерактивний довідник (Швидкий старт та Підказки)
    # ------------------------------------------------------------------
    def _open_help_modal(self, e=None):
        def help_card(icon, title: str, color: str, bullets: List[str]) -> ft.Control:
            return self._glass(
                hover=False, blur=False, radius=10, padding=14,
                content=ft.Column(spacing=6, controls=[
                    ft.Row(spacing=10, controls=[
                        self._icon_badge(icon, color, 34, 18),
                        ft.Text(title, size=14, weight=ft.FontWeight.W_800)]),
                    *[ft.Row(spacing=8, vertical_alignment=ft.CrossAxisAlignment.START, controls=[
                        ft.Icon(ft.Icons.CHECK_CIRCLE_OUTLINE, size=14, color=color),
                        ft.Text(b, size=12, color=ft.Colors.ON_SURFACE_VARIANT, expand=True)
                    ]) for b in bullets]]))

        dlg = ft.AlertDialog(
            title=ft.Row(spacing=10, controls=[
                ft.Icon(ft.Icons.LIGHTBULB_OUTLINE, color=self.c1, size=24),
                ft.Text("Довідник та підказки V-Pose Combine Studio", size=18, weight=ft.FontWeight.W_800)]),
            content=ft.Container(
                width=740, height=500,
                content=ft.Column(scroll=ft.ScrollMode.AUTO, spacing=12, controls=[
                    help_card(
                        ft.Icons.ACCESSIBILITY_NEW, "1. Модальність «Поза» (V-Pose Жести)", self.c1,
                        [
                            "У під-вкладці «Запис» оберіть тип жесту (Статична поза, Траєкторія, Перехід А->Б, Помах, Утримання або Повзунок). Для Повзунка доступна лише модель «Кисть».",
                            "У під-вкладці «Бібліотека» натисніть «Тест у камері», щоб перевірити точність жесту наживо.",
                            "У під-вкладці «Зв'язки» прив'яжіть жест до дії (клавіша, клік миші, гучність) та натисніть велику кнопку живлення в «Конвеєр»."
                        ]),
                    help_card(
                        ft.Icons.FACE_RETOUCHING_NATURAL, "2. Модальність «Обличчя» (Face ID — YuNet + SFace)", "#38bdf8",
                        [
                            "Введіть своє ім'я та натисніть «Захопити обличчя з камери». Система автоматично зробить серію з 5 кадрів та створить стійкий 128-d біометричний профіль.",
                            "За допомогою повзунка «Поріг суворості Face ID» налаштуйте чутливість (рекомендовано 0.36).",
                            "Натисніть «Запустити живий сканер», щоб перевірити розпізнавання в реальному часі (зелена рамка — власник, помаранчева — чужий)."
                        ]),
                    help_card(
                        ft.Icons.RECORD_VOICE_OVER, "3. Модальність «Голос» (Google YAMNet + Voice ID)", "#22c55e",
                        [
                            "Спочатку зареєструйте свій голос у блоці «База дикторів» (говоріть звичайним тембром 3.5 секунди).",
                            "Для перевірки натисніть «Запустити перевірку (3 с)» у верхньому блоці — мікрофон увімкнеться лише на час запису і видасть 2 результати: Тип звуку (YAMNet) та Кому належить голос (Voice ID).",
                            "Повзунок «Поріг схожості голосу» дозволяє зробити перевірку суворішою або м'якшою."
                        ]),
                    help_card(
                        ft.Icons.SMART_TOY, "4. Експорт у власні проєкти та 2FA-Робот (Ctrl+E)", "#f59e0b",
                        [
                            "Натисніть «Експорт коду» у верхній шапці. Ви можете окремо завантажити ZIP-архів або чистий .py код для Жестів, Face ID, Голосу або комплексного 2FA-Робота.",
                            "Пакет «Робот (2FA+Жести)» містить двофакторний захист (Обличчя + Голос), сторожовий таймер авто-блокування (Watchdog 15 с) та готові команди для Arduino/ESP32/Raspberry Pi."
                        ])])),
            actions=[ft.FilledButton("Зрозуміло", on_click=lambda _: self.page.pop_dialog())],
            shape=ft.RoundedRectangleBorder(radius=14))
        self.page.show_dialog(dlg)

    # ------------------------------------------------------------------
    # Діалог експорту коду (Python SDK)
    # ------------------------------------------------------------------
    def _open_export_modal(self, e=None, default_modality="gesture", gesture_path=None):
        gestures = [g for g in scan_gestures() if g["kind"] == "json"]
        curr_gpath = gesture_path or (gestures[0]["path"] if gestures else None)
        curr_modality = default_modality
        curr_vmode = "yamnet"

        code_view = ft.TextField(
            multiline=True, min_lines=14, max_lines=22, read_only=True,
            text_size=11, text_style=ft.TextStyle(font_family="Consolas"),
            border=ft.OutlineInputBorder(border_radius=8), expand=True)

        status_lbl = ft.Text("", size=11, color=ft.Colors.ON_SURFACE_VARIANT)

        def generate_code() -> str:
            if curr_modality == "gesture":
                if curr_gpath and os.path.exists(curr_gpath):
                    return export_gesture_standalone(curr_gpath)
                return "# Записаних жестів (.json) поки що немає. Створіть жест у вкладці «Поза» -> «Запис»."
            elif curr_modality == "face":
                return export_face_id_standalone()
            elif curr_modality == "voice":
                return export_voice_standalone(mode=curr_vmode)
            elif curr_modality == "robot":
                return export_multimodal_robot_guard()
            return ""

        def refresh_code():
            code_view.value = generate_code()
            status_lbl.value = ""
            self._safe(code_view)
            self._safe(status_lbl)

        def on_g_select(e):
            nonlocal curr_gpath
            curr_gpath = e.control.value
            update_action_buttons()
            refresh_code()

        g_opts = [ft.dropdown.Option(g["path"], g["name"]) for g in gestures]
        g_dd = ft.Dropdown(
            label="Оберіть жест для експорту", options=g_opts, value=curr_gpath,
            border=ft.OutlineInputBorder(border_radius=8), visible=(curr_modality == "gesture"),
            on_select=on_g_select)

        def on_v_change(e):
            nonlocal curr_vmode
            curr_vmode = list(e.control.selected)[0]
            update_action_buttons()
            refresh_code()

        v_seg = ft.SegmentedButton(
            selected=[curr_vmode], allow_empty_selection=False, visible=(curr_modality == "voice"),
            segments=[
                ft.Segment(value="yamnet", label=ft.Text("Звуки YAMNet")),
                ft.Segment(value="voice_id", label=ft.Text("Диктор Voice ID")),
            ],
            on_change=on_v_change)

        def get_modality_title() -> str:
            if curr_modality == "gesture":
                return "Жести"
            elif curr_modality == "face":
                return "Face ID"
            elif curr_modality == "voice":
                return f"Голос ({'YAMNet' if curr_vmode == 'yamnet' else 'Voice ID'})"
            return "2FA Робот"

        package_btn = ft.FilledButton(
            f"Завантажити ZIP-пакет ({get_modality_title()})",
            icon=ft.Icons.FOLDER_ZIP)

        file_btn = ft.OutlinedButton(
            f"Зберегти код (.py)",
            icon=ft.Icons.CODE)

        def update_action_buttons():
            title = get_modality_title()
            package_btn.text = f"Завантажити ZIP-пакет ({title})"
            file_btn.text = f"Зберегти код ({title}.py)"
            self._safe(package_btn)
            self._safe(file_btn)

        def save_file(e):
            os.makedirs("exports", exist_ok=True)
            fname = f"run_{curr_modality}.py"
            if curr_modality == "gesture" and curr_gpath:
                base = os.path.splitext(os.path.basename(curr_gpath))[0]
                fname = f"run_gesture_{base}.py"
            elif curr_modality == "voice":
                fname = f"run_voice_{curr_vmode}.py"
            elif curr_modality == "robot":
                fname = "run_robot_guard.py"
            out_path = os.path.join("exports", fname)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(code_view.value)
            status_lbl.value = f"Файл коду збережено: {out_path}"
            self._safe(status_lbl)
            self._toast(f"Код збережено у {out_path}", "ok")

        def save_package(e):
            title = get_modality_title()
            status_lbl.value = f"Створення ZIP-пакета для «{title}»..."
            self._safe(status_lbl)
            try:
                res = create_export_package(
                    modality=curr_modality,
                    gesture_path=curr_gpath,
                    voice_mode=curr_vmode,
                    zip_package=True
                )
                zname = os.path.basename(res['zip_path'])
                pdir = os.path.basename(res['package_dir'])
                status_lbl.value = f"Успішно! Створено архів: {zname} (Папка: {pdir})"
                self._safe(status_lbl)
                self._toast(f"ZIP-пакет «{title}» успішно експортовано!", "ok")
            except Exception as ex:
                status_lbl.value = f"Помилка створення пакета: {ex}"
                self._safe(status_lbl)
                self._toast(f"Помилка експорту: {ex}", "err")

        package_btn.on_click = save_package
        file_btn.on_click = save_file

        def on_mod_change(e):
            nonlocal curr_modality
            curr_modality = list(e.control.selected)[0]
            g_dd.visible = (curr_modality == "gesture")
            v_seg.visible = (curr_modality == "voice")
            self._safe(g_dd)
            self._safe(v_seg)
            update_action_buttons()
            refresh_code()

        mod_seg = ft.SegmentedButton(
            selected=[curr_modality], allow_empty_selection=False,
            segments=[
                ft.Segment(value="gesture", label=ft.Text("Жести"), icon=ft.Icon(ft.Icons.ACCESSIBILITY_NEW, size=16)),
                ft.Segment(value="face", label=ft.Text("Face ID"), icon=ft.Icon(ft.Icons.FACE, size=16)),
                ft.Segment(value="voice", label=ft.Text("Голос"), icon=ft.Icon(ft.Icons.MIC, size=16)),
                ft.Segment(value="robot", label=ft.Text("Робот (2FA+Жести)"), icon=ft.Icon(ft.Icons.SMART_TOY, size=16)),
            ],
            on_change=on_mod_change)

        def copy_code(e):
            try:
                self.page.set_clipboard(code_view.value or "")
            except Exception:
                pass
            status_lbl.value = "✅ Код скопійовано в буфер обміну!"
            self._safe(status_lbl)
            self._toast("Код скопійовано в буфер обміну", "ok")

        def open_exports_folder(e):
            os.makedirs("exports", exist_ok=True)
            abs_exp = os.path.abspath("exports")
            try:
                os.startfile(abs_exp)
            except Exception:
                pass
            status_lbl.value = f"📂 Відкрито папку: {abs_exp}"
            self._safe(status_lbl)

        copy_btn = ft.OutlinedButton("Копіювати код", icon=ft.Icons.CONTENT_COPY, on_click=copy_code)
        folder_btn = ft.TextButton("Відкрити папку exports/", icon=ft.Icons.FOLDER_OPEN, on_click=open_exports_folder)

        def close_dialog(e):
            self.page.pop_dialog()

        code_view.value = generate_code()

        dlg = ft.AlertDialog(
            title=ft.Row(spacing=10, controls=[
                ft.Icon(ft.Icons.CODE, color=self.c1, size=24),
                ft.Text("Експорт автономного Python коду та моделей", size=18, weight=ft.FontWeight.W_700)]),
            content=ft.Container(
                width=780, height=530,
                content=ft.Column(spacing=12, controls=[
                    ft.Text("Оберіть, що саме ви хочете експортувати: окремо жести, окремо обличчя (Face ID), окремо голос або комплексний 2FA захист. Для кожного варіанта доступне завантаження як повноцінного ZIP-пакета (з ONNX/.task моделями та профілями), так і чистого автономного Python-коду.", size=12, color=ft.Colors.ON_SURFACE_VARIANT),
                    mod_seg,
                    g_dd,
                    v_seg,
                    code_view,
                    status_lbl])),
            actions=[
                package_btn,
                file_btn,
                copy_btn,
                folder_btn,
                ft.TextButton("Закрити", on_click=close_dialog)],
            shape=ft.RoundedRectangleBorder(radius=12))

        self.page.show_dialog(dlg)

    def _export_single_gesture(self, gesture_path: str):
        self._open_export_modal(default_modality="gesture", gesture_path=gesture_path)

    # ------------------------------------------------------------------
    # Навігація
    # ------------------------------------------------------------------
    def _goto(self, index: int):
        if index == self.tab and self.switcher.content is not None and getattr(self.switcher.content, "key", None) == f"tab{index}":
            return
        if self.tab == 1 and index != 1:
            self.face_embedded_running = False
        if self.tab == 3 and index != 3:
            self.sim_running = False
        self.tab = index
        for i, item in enumerate(self.nav_items):
            selected = i == index
            icon, sel_icon, _, _ = NAV[i]
            item.bgcolor = op(0.16, self.c1) if selected else None
            item.scale = 1
            item.border = ft.Border(left=ft.BorderSide(3, self.c1 if selected else "transparent"))
            item.data["icon"].icon = sel_icon if selected else icon
            item.data["icon"].color = self.c1 if selected else ft.Colors.ON_SURFACE_VARIANT
            item.data["text"].color = ft.Colors.ON_SURFACE if selected else ft.Colors.ON_SURFACE_VARIANT
            item.data["text"].weight = ft.FontWeight.W_700 if selected else ft.FontWeight.W_500
        self.title_text.value = NAV[index][2]
        self.subtitle_text.value = NAV[index][3]
        self._show_tab(index)

    def _show_tab(self, index: int):
        self.tab = index
        builders = [self._view_vpose, self._view_face, self._view_voice, self._view_robot_sim, self._view_appearance]
        view = builders[index]()
        view.key = f"tab{index}"
        self.switcher.content = view
        self._page_update()
        self._play_enter(view.controls)

    def _view(self, controls: List[ft.Control]) -> ft.Column:
        self._prepare_enter(controls)
        return ft.Column(expand=True, scroll=ft.ScrollMode.AUTO, spacing=18,
                         horizontal_alignment=ft.CrossAxisAlignment.STRETCH, controls=controls)

    # ==================================================================
    # МОДАЛЬНІСТЬ 1 — ПОЗА (V-POSE)
    # ==================================================================
    def _view_vpose(self) -> ft.Column:
        subtabs = [
            ("Конвеєр", ft.Icons.DASHBOARD_OUTLINED),
            ("Запис", ft.Icons.FIBER_MANUAL_RECORD_OUTLINED),
            ("Бібліотека", ft.Icons.FOLDER_COPY_OUTLINED),
            ("Зв'язки", ft.Icons.ACCOUNT_TREE_OUTLINED),
        ]
        seg = ft.SegmentedButton(
            selected=[str(self.vpose_subtab)],
            allow_empty_selection=False,
            segments=[
                ft.Segment(value=str(i), label=ft.Text(title), icon=ft.Icon(icon, size=16))
                for i, (title, icon) in enumerate(subtabs)
            ],
            on_change=self._on_vpose_subtab_change,
        )
        builders = [self._view_pipeline, self._view_record, self._view_library, self._view_bindings]
        content_view = builders[self.vpose_subtab]()

        top_bar = self._glass(
            hover=False, blur=False, padding=ft.Padding.symmetric(horizontal=14, vertical=10), radius=10,
            content=ft.Row(alignment=ft.MainAxisAlignment.SPACE_BETWEEN, wrap=True, spacing=10, controls=[
                ft.Row(spacing=10, controls=[
                    self._icon_badge(ft.Icons.ACCESSIBILITY_NEW, self.c1, 38, 20),
                    ft.Column(spacing=1, tight=True, controls=[
                        ft.Text("V-Pose Studio (Жести)", size=14, weight=ft.FontWeight.W_700),
                        ft.Text("Керування жестами рук, тіла та зв'язками дій", size=11, color=ft.Colors.ON_SURFACE_VARIANT)])]),
                seg]))

        guide = self._guide_banner([
            ("1", "Запишіть жест", "Перейдіть у «Запис», оберіть тип жесту та модель (Кисть / Обличчя / Тіло)."),
            ("2", "Протестуйте та прив'яжіть", "Перевірте жест у «Бібліотеці» та призначте дію у «Зв'язках»."),
            ("3", "Запустіть або експортуйте", "Увімкніть живлення у «Конвеєр» або натисніть «Експорт коду» для свого проєкту."),
        ])

        return self._view([top_bar, guide, *content_view.controls])

    def _on_vpose_subtab_change(self, e):
        sel = list(e.control.selected)
        if sel:
            self.vpose_subtab = int(sel[0])
            self._show_tab(0)

    # --- Під-модулі V-Pose ---
    def _is_running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def _view_pipeline(self) -> ft.Column:
        cfg = read_cfg()
        settings = cfg.get("settings", {})
        running = self._is_running()

        self.power_ring = ft.ProgressRing(width=186, height=186, stroke_width=3, color="#22c55e", visible=running)
        self.power_btn = ft.Container(
            width=150, height=150, border_radius=75, alignment=ft.Alignment.CENTER, on_click=self._toggle_pipeline,
            animate=ft.Animation(1000, ft.AnimationCurve.EASE_IN_OUT),
            animate_scale=ft.Animation(1000, ft.AnimationCurve.EASE_IN_OUT),
            content=ft.Icon(ft.Icons.POWER_SETTINGS_NEW, size=64, color="white"), tooltip="Запустити / зупинити конвеєр")
        self._style_power(running)

        power_stack = ft.Stack(width=196, height=196, controls=[
            ft.Container(content=self.power_ring, alignment=ft.Alignment.CENTER, width=196, height=196),
            ft.Container(content=self.power_btn, alignment=ft.Alignment.CENTER, width=196, height=196)])

        self.run_title = ft.Text("Конвеєр працює" if running else "Конвеєр зупинено", size=24, weight=ft.FontWeight.W_800)
        self.run_sub = ft.Text("Розпізнавання активне — дивіться вікно прев'ю." if running
                               else "Натисніть кнопку живлення, щоб запустити розпізнавання жестів.",
                               size=13, color=ft.Colors.ON_SURFACE_VARIANT)
        self.uptime_text = ft.Text(fmt_uptime(time.time() - self.proc_started) if running else "00:00:00", size=34,
                                   weight=ft.FontWeight.W_300, font_family="Consolas", color=self.c1)

        cam = str(settings.get("camera_id", 0))
        camera_seg = ft.SegmentedButton(
            selected=[cam if cam in ("0", "1", "2", "3") else "0"], allow_empty_selection=False,
            segments=[ft.Segment(value=str(i), label=ft.Text(f"Камера {i}"), icon=ft.Icon(ft.Icons.VIDEOCAM, size=16))
                      for i in range(4)],
            on_change=self._on_camera)
        preview_sw = ft.Switch(label="Вікно прев'ю (OpenCV)", value=bool(settings.get("preview_window", True)),
                               on_change=lambda e: self._save_setting("preview_window", bool(e.control.value)))
        marks_sw = ft.Switch(label="Точки на руках / обличчі", value=bool(settings.get("show_landmarks", True)),
                             on_change=lambda e: self._save_setting("show_landmarks", bool(e.control.value)))
        mirror_sw = ft.Switch(label="Дзеркальний режим (Mirror)", value=bool(settings.get("flip_horizontal", True)),
                              on_change=lambda e: self._save_setting("flip_horizontal", bool(e.control.value)))

        hero = self._glass(
            padding=20, radius=10, tint=self.c1,
            content=ft.ResponsiveRow(
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
                spacing=20, run_spacing=16,
                controls=[
                    ft.Container(col={"xs": 12, "md": 4, "lg": 3}, content=power_stack, alignment=ft.Alignment.CENTER),
                    ft.Container(
                        col={"xs": 12, "md": 8, "lg": 9},
                        content=ft.Column(spacing=8, controls=[
                            self.run_title, self.run_sub, self.uptime_text,
                            ft.Container(height=4), camera_seg,
                            ft.Row(wrap=True, spacing=16, run_spacing=8, controls=[preview_sw, marks_sw, mirror_sw])
                        ])
                    )
                ]
            )
        )

        gestures = scan_gestures()
        tiles = ft.ResponsiveRow(spacing=14, run_spacing=14, controls=[
            self._stat_tile(ft.Icons.ACCOUNT_TREE, "Активні зв'язки", len(cfg.get("bindings", [])), self.c1),
            self._stat_tile(ft.Icons.CODE, "Модулі жестів (.py)", sum(g["kind"] == "py" for g in gestures), "#38bdf8"),
            self._stat_tile(ft.Icons.FIBER_SMART_RECORD, "Записані зразки", sum(g["kind"] == "json" for g in gestures), "#f59e0b"),
            self._stat_tile(ft.Icons.BOLT, "Доступні дії", len(scan_actions()), "#ec4899"),
        ])

        self.log_view = ft.ListView(expand=True, spacing=2, auto_scroll=True,
                                    controls=[self._log_line(t) for t in self.log_lines])
        if not self.log_lines:
            self.log_view.controls.append(ft.Text("Консоль порожня. Запустіть конвеєр, щоб побачити журнал...",
                                                  size=12, font_family="Consolas", color=ft.Colors.ON_SURFACE_VARIANT))
        console = self._glass(
            hover=False, radius=10, content=ft.Column(spacing=10, controls=[
                ft.Row(alignment=ft.MainAxisAlignment.SPACE_BETWEEN, controls=[
                    self._section_title("Консоль конвеєра", ft.Icons.TERMINAL, "Вивід процесу run.py у реальному часі"),
                    ft.IconButton(ft.Icons.DELETE_SWEEP, tooltip="Очистити консоль", on_click=self._clear_log)]),
                ft.Container(height=240, padding=12, border_radius=8, bgcolor=op(0.42, "#000000"),
                             border=ft.Border.all(1, op(0.1, ft.Colors.ON_SURFACE)), content=self.log_view)]))

        return self._view([hero, tiles, console])

    def _stat_tile(self, icon, label: str, value: int, color: str) -> ft.Control:
        number = ft.Text("0", size=26, weight=ft.FontWeight.W_800, font_family="Consolas")
        tile = self._glass(
            tint=color, padding=14, radius=10, col={"xs": 6, "md": 3},
            content=ft.Row(spacing=14, controls=[
                self._icon_badge(icon, color), ft.Column(spacing=0, tight=True, controls=[
                    number, ft.Text(label, size=12, color=ft.Colors.ON_SURFACE_VARIANT)])]))
        self._count_up(number, value)
        return tile

    def _count_up(self, text: ft.Text, target: int):
        if not self.animations or target <= 0:
            text.value = str(target)
            return

        def run():
            time.sleep(0.25)
            steps = min(22, max(target, 6))
            for i in range(1, steps + 1):
                if not self.alive:
                    return
                text.value = str(round(target * i / steps))
                self._safe(text)
                time.sleep(0.03)

        threading.Thread(target=run, daemon=True).start()

    def _style_power(self, running: bool):
        colors = ["#22c55e", "#15803d"] if running else [self.c1, self.c2]
        self.power_btn.gradient = ft.LinearGradient(begin=ft.Alignment.TOP_LEFT, end=ft.Alignment.BOTTOM_RIGHT, colors=colors)
        self.power_btn.shadow = ft.BoxShadow(blur_radius=34, spread_radius=2, color=op(0.55, colors[0]), offset=ft.Offset(0, 10))

    def _log_line(self, text: str) -> ft.Text:
        low = text.lower()
        color = ft.Colors.ON_SURFACE
        if "traceback" in low or "error" in low or "помилка" in low or "exception" in low:
            color = "#f87171"
        elif text.startswith("[Engine]") or "успішно" in low:
            color = "#4ade80"
        elif text.startswith("[") and "]" in text:
            color = "#7dd3fc"
        return ft.Text(text, size=12, font_family="Consolas", color=color, selectable=True)

    def _push_log(self, text: str):
        self.log_lines.append(text)
        if self.tab == 0 and self.log_view is not None:
            if len(self.log_view.controls) >= 400:
                self.log_view.controls.pop(0)
            if self.log_view.controls and getattr(self.log_view.controls[0], "value", "").startswith("Консоль порожня"):
                self.log_view.controls.clear()
            self.log_view.controls.append(self._log_line(text))
            self._safe(self.log_view)

    def _clear_log(self, e):
        self.log_lines.clear()
        if self.log_view is not None:
            self.log_view.controls.clear()
            self._safe(self.log_view)

    def _on_camera(self, e):
        sel = list(e.control.selected)
        if sel:
            self._save_setting("camera_id", int(sel[0]))

    def _save_setting(self, key: str, value: Any):
        cfg = read_cfg()
        cfg.setdefault("settings", {})[key] = value
        write_cfg(cfg)

    def _toggle_pipeline(self, e):
        if self._is_running():
            self._stop_pipeline()
        else:
            self._start_pipeline()

    def _spawn(self, cmd: List[str], on_line, on_exit) -> subprocess.Popen:
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                text=True, encoding="utf-8", errors="replace", env=env)

        def reader():
            try:
                for line in iter(proc.stdout.readline, ""):
                    line = line.rstrip()
                    if line:
                        on_line(line)
            except Exception:
                pass
            proc.wait()
            on_exit(proc)

        threading.Thread(target=reader, daemon=True).start()
        return proc

    def _start_pipeline(self):
        if self._is_running():
            return
        if self._is_testing():
            self._stop_test(quiet=True)
            time.sleep(0.5)
        self.log_lines.clear()
        if self.log_view is not None:
            self.log_view.controls.clear()
        self._push_log(f"[GUI] Запуск: python run.py --config {PIPELINE_FILE}")
        self.proc = self._spawn([sys.executable, "-u", "run.py", "--config", PIPELINE_FILE],
                                self._push_log, self._on_pipeline_exit)
        self.proc_started = time.time()
        self._refresh_status()
        self._toast("Конвеєр запущено", "ok")

    def _stop_pipeline(self, quiet: bool = False):
        proc = self.proc
        if proc is not None:
            try:
                proc.terminate()
            except Exception:
                pass
        self.proc = None
        self._push_log("[GUI] Конвеєр зупинено користувачем.")
        self._refresh_status()
        if not quiet:
            self._toast("Конвеєр зупинено")

    def _on_pipeline_exit(self, proc: subprocess.Popen):
        if proc is self.proc:
            self.proc = None
            self._push_log(f"[GUI] Процес завершився (код {proc.returncode}).")
            self._refresh_status()
            if proc.returncode not in (0, None):
                self._toast("Конвеєр завершився з помилкою — див. консоль", "err")

    def _refresh_status(self):
        running = self._is_running()
        if self.status_dot is not None:
            self.status_dot.bgcolor = "#22c55e" if running else "#64748b"
            self.status_dot.opacity = 1
        if self.status_text is not None:
            self.status_text.value = "Працює" if running else "Зупинено"
            self.status_text.color = "#22c55e" if running else ft.Colors.ON_SURFACE_VARIANT
        if self.status_pill is not None:
            self.status_pill.border = ft.Border.all(1, op(0.55, "#22c55e") if running else op(0.14, ft.Colors.ON_SURFACE))
        if self.tab == 0 and self.power_btn is not None:
            self._style_power(running)
            self.power_btn.scale = 1
            self.power_ring.visible = running
            self.run_title.value = "Конвеєр працює" if running else "Конвеєр зупинено"
            self.run_sub.value = ("Розпізнавання активне — дивіться вікно прев'ю." if running
                                  else "Натисніть кнопку живлення, щоб запустити розпізнавання жестів.")
            if not running:
                self.uptime_text.value = "00:00:00"
        self._page_update()

    # ------------------------------------------------------------------
    # Фоновий «пульс»: статус, кнопка, плями, uptime
    # ------------------------------------------------------------------
    def _ticker(self):
        n = 0
        while self.alive:
            time.sleep(1)
            n += 1
            try:
                self._pulse_on = not self._pulse_on
                running = self._is_running()
                if self.animations:
                    if self.status_dot is not None and running:
                        self.status_dot.opacity = 1 if self._pulse_on else 0.3
                        self._safe(self.status_dot)
                    if running and self.tab == 0 and self.power_btn is not None:
                        self.power_btn.scale = 1.05 if self._pulse_on else 1.0
                        self.power_btn.shadow = ft.BoxShadow(
                            blur_radius=70 if self._pulse_on else 28, spread_radius=6 if self._pulse_on else 1,
                            color=op(0.6, "#22c55e"), offset=ft.Offset(0, 10))
                        self._safe(self.power_btn)
                if running and self.tab == 0 and self.uptime_text is not None:
                    self.uptime_text.value = fmt_uptime(time.time() - self.proc_started)
                    self._safe(self.uptime_text)
                if self.animations and self.settings["orbs"] and n % 9 == 0:
                    self._drift_orbs()
            except Exception:
                pass

    def _drift_orbs(self):
        w = self.page.width or 1300
        h = self.page.height or 800
        for orb in self.orbs:
            orb.left = random.uniform(-0.15, 0.85) * w
            orb.top = random.uniform(-0.15, 0.75) * h
            self._safe(orb)

    def _cleanup(self):
        self.alive = False
        self.face_embedded_running = False
        self.sim_running = False
        if self.audio_mgr is not None:
            try:
                self.audio_mgr.stop()
            except Exception:
                pass
        for proc in (self.proc, self.rec_proc, self.test_proc, self.face_proc):
            try:
                if proc is not None and proc.poll() is None:
                    proc.terminate()
            except Exception:
                pass

    def _audio_ui_ticker(self):
        while self.alive:
            time.sleep(0.15)
            try:
                if self.tab == 2 and self.audio_mgr and self.audio_mgr.running:
                    snap = self.audio_mgr.get_snapshot()
                    if self.audio_vu_bar is not None:
                        self.audio_vu_bar.value = snap["rms"]
                        self._safe(self.audio_vu_bar)
                    if self.audio_db_text is not None:
                        pct = int(snap["rms"] * 100)
                        self.audio_db_text.value = f"Рівень сигналу: {pct}%"
                        self._safe(self.audio_db_text)
                    if self.audio_events_col is not None and snap["top_events"]:
                        self._update_audio_events(snap["top_events"])
                    if self.voice_speaker_badge is not None and snap["speaker"]:
                        self._update_speaker_badge(snap["speaker"])
                    if not self.voice_is_checking and snap.get("spectrum"):
                        self._update_waveform_bars(snap["spectrum"])
            except Exception:
                pass

    # ==================================================================
    # ВКЛАДКА 2 — ЗАПИС
    # ==================================================================
    def _get_model_segments(self) -> List[ft.Segment]:
        if self.rec_type == "slider":
            return [
                ft.Segment(value="hands", label=ft.Text("Кисть"), icon=ft.Icon(ft.Icons.PAN_TOOL_ALT, size=18)),
                ft.Segment(value="face", label=ft.Text("Обличчя (вимкнено)"), icon=ft.Icon(ft.Icons.LOCK_OUTLINE, size=16), disabled=True),
                ft.Segment(value="pose", label=ft.Text("Тіло (вимкнено)"), icon=ft.Icon(ft.Icons.LOCK_OUTLINE, size=16), disabled=True),
            ]
        return [
            ft.Segment(value=m[0], label=ft.Text(m[2]), icon=ft.Icon(m[3], size=18), disabled=False)
            for m in MODELS
        ]

    def _on_model_seg_change(self, e):
        sel = list(e.control.selected)
        if sel:
            if self.rec_type == "slider":
                self.rec_model = "hands"
                if self.model_seg is not None:
                    self.model_seg.selected = ["hands"]
                    self._safe(self.model_seg)
            else:
                self.rec_model = sel[0]

    def _refresh_rec_btn(self):
        if self.rec_btn is None:
            return
        is_slider = self.rec_type == "slider"
        label = "Зберегти конфігурацію повзунка" if is_slider else "Відкрити камеру й записати"
        icon = ft.Icons.SAVE if is_slider else ft.Icons.VIDEOCAM
        try:
            self.rec_btn.content.controls[0].icon = icon
            self.rec_btn.content.controls[1].value = label
            self._safe(self.rec_btn)
        except Exception:
            pass
        if self.rec_camera_hint is not None:
            self.rec_camera_hint.value = (
                "Повзунок — числовий регулятор (0–100%). Камера для запису не потрібна: збережіть його та призначте у «Зв'язках»."
                if is_slider else
                "У вікні камери: ПРОБІЛ — зйомка, ESC / Q — скасування."
            )
            self._safe(self.rec_camera_hint)

    def _view_record(self) -> ft.Column:
        recording = self.rec_proc is not None and self.rec_proc.poll() is None

        # --- карти парадигм ---
        self.rec_cards = {}
        grid = ft.ResponsiveRow(spacing=14, run_spacing=14)
        for p in PARADIGMS:
            card = self._paradigm_card(p)
            self.rec_cards[p["type"]] = card
            grid.controls.append(card)

        # --- частина тіла ---
        if self.rec_type == "slider":
            self.rec_model = "hands"

        self.model_lock_info = ft.Container(
            visible=(self.rec_type == "slider"),
            padding=ft.Padding.symmetric(horizontal=10, vertical=7), border_radius=6,
            bgcolor=op(0.12, "#f59e0b"), border=ft.Border.all(1, op(0.35, "#f59e0b")),
            content=ft.Row(spacing=8, controls=[
                ft.Icon(ft.Icons.LOCK, size=16, color="#f59e0b"),
                ft.Text("Для повзунка доступна лише кисть руки. Обличчя та тіло вимкнено.",
                        size=11, color="#f59e0b", weight=ft.FontWeight.W_600, expand=True)
            ]))

        self.model_seg = ft.SegmentedButton(
            selected=[self.rec_model], allow_empty_selection=False,
            segments=self._get_model_segments(),
            on_change=self._on_model_seg_change)

        name_field = ft.TextField(
            label="Назва жесту (латиницею)", hint_text="fist_to_fingers, wave, volume_bar",
            value=self.rec_name, prefix_icon=ft.Icons.LABEL_OUTLINE,
            border=ft.OutlineInputBorder(border_radius=8),
            on_change=lambda e: setattr(self, "rec_name", e.control.value), on_submit=self._on_record_click)
        self.rec_name_field = name_field

        self.rec_params_switcher = ft.AnimatedSwitcher(
            content=self._rec_params_panel(), transition=ft.AnimatedSwitcherTransition.FADE, duration=300)

        self.rec_status = ft.Text("", size=12, color=ft.Colors.ON_SURFACE_VARIANT, max_lines=2)
        self.rec_ring = ft.ProgressRing(width=22, height=22, stroke_width=3, visible=recording)
        self.rec_btn = self._gradient_button("Відкрити камеру й записати", ft.Icons.VIDEOCAM, self._on_record_click)
        self.rec_btn.opacity = 0.5 if recording else 1

        self.rec_camera_hint = ft.Text(
            "Повзунок — числовий регулятор (0–100%). Камера для запису не потрібна: збережіть його та призначте у «Зв'язках»."
            if self.rec_type == "slider" else
            "У вікні камери: ПРОБІЛ — зйомка, ESC / Q — скасування.",
            size=11, color=ft.Colors.ON_SURFACE_VARIANT, expand=True)

        self._refresh_rec_btn()

        left = self._glass(
            hover=False, radius=10, col={"xs": 12, "lg": 7},
            content=ft.Column(spacing=14, controls=[
                self._section_title("1. Оберіть характер руху", ft.Icons.CATEGORY, "Шість фундаментальних парадигм"), grid]))

        right = self._glass(
            hover=False, radius=10, col={"xs": 12, "lg": 5},
            content=ft.Column(spacing=14, controls=[
                self._section_title("2. Налаштуйте і запишіть", ft.Icons.TUNE, "Параметри передаються рекордеру автоматично"),
                name_field,
                ft.Text("Частина тіла для відстеження", size=12, color=ft.Colors.ON_SURFACE_VARIANT),
                self.model_lock_info,
                self.model_seg, self.rec_params_switcher,
                ft.Row(spacing=12, controls=[self.rec_btn, self.rec_ring]),
                self.rec_status,
                self._glass(hover=False, blur=False, padding=10, radius=8, content=ft.Row(spacing=10, controls=[
                    ft.Icon(ft.Icons.KEYBOARD, size=16, color=self.c1),
                    self.rec_camera_hint]))]))

        return self._view([ft.ResponsiveRow(spacing=16, run_spacing=16, controls=[left, right])])

    def _paradigm_card(self, p: Dict[str, Any]) -> ft.Container:
        selected = p["type"] == self.rec_type
        check = ft.Icon(ft.Icons.CHECK_CIRCLE, color=p["color"], size=20, scale=1 if selected else 0,
                        animate_scale=ft.Animation(300, ft.AnimationCurve.EASE_OUT))

        def on_hover(e):
            if p["type"] == self.rec_type:
                return
            hot = str(e.data).lower() == "true"
            e.control.offset = ft.Offset(0, -0.010) if hot else ft.Offset(0, 0)
            e.control.border = ft.Border.all(1, op(0.4, p["color"]) if hot else op(0.12, ft.Colors.ON_SURFACE))
            self._safe(e.control)

        card = ft.Container(
            col={"xs": 12, "sm": 6, "xl": 4}, padding=12, border_radius=8, data=dict(check=check),
            bgcolor=op(0.16 if selected else 0.30, p["color"] if selected else ft.Colors.SURFACE_CONTAINER_HIGH) if selected else op(0.20, ft.Colors.SURFACE_CONTAINER_HIGH),
            border=ft.Border.all(1.4 if selected else 1, p["color"] if selected else op(0.12, ft.Colors.ON_SURFACE)),
            shadow=ft.BoxShadow(blur_radius=12, color=op(0.24, p["color"]), offset=ft.Offset(0, 3)) if selected else None,
            clip_behavior=ft.ClipBehavior.ANTI_ALIAS,
            animate=self._anim(200), animate_offset=self._anim(180, ft.AnimationCurve.EASE_OUT_CUBIC),
            on_hover=on_hover if self.animations else None, on_click=lambda e, t=p["type"]: self._select_paradigm(t),
            content=ft.Column(spacing=6, controls=[
                ft.Row(alignment=ft.MainAxisAlignment.SPACE_BETWEEN, controls=[self._icon_badge(p["icon"], p["color"], 36, 18), check]),
                ft.Text(f"{p['choice']}. {p['title']}", size=13, weight=ft.FontWeight.W_700),
                ft.Text(p["desc"], size=11, color=ft.Colors.ON_SURFACE_VARIANT)]))
        return card

    def _select_paradigm(self, ptype: str):
        self.rec_type = ptype
        if ptype == "slider":
            self.rec_model = "hands"
        for t, card in self.rec_cards.items():
            p = PARADIGM_BY_TYPE[t]
            sel = t == ptype
            card.bgcolor = op(0.16 if sel else 0.20, p["color"] if sel else ft.Colors.SURFACE_CONTAINER_HIGH)
            card.border = ft.Border.all(1.4 if sel else 1, p["color"] if sel else op(0.12, ft.Colors.ON_SURFACE))
            card.shadow = ft.BoxShadow(blur_radius=12, color=op(0.24, p["color"]), offset=ft.Offset(0, 3)) if sel else None
            card.scale = 1
            card.data["check"].scale = 1 if sel else 0
        if self.rec_params_switcher is not None:
            self.rec_params_switcher.content = self._rec_params_panel()
        if self.model_seg is not None:
            self.model_seg.segments = self._get_model_segments()
            self.model_seg.selected = [self.rec_model]
            self._safe(self.model_seg)
        if self.model_lock_info is not None:
            self.model_lock_info.visible = (ptype == "slider")
            self._safe(self.model_lock_info)
        self._refresh_rec_btn()
        self._page_update()

    def _slider_row(self, title: str, key: str, lo: float, hi: float, div: int, unit: str = "", as_int=False) -> ft.Control:
        value = self.rec_params[key]
        value_text = ft.Text(f"{int(value) if as_int else value:g}{unit}", size=13, weight=ft.FontWeight.W_700, color=self.c1)

        def change(e):
            v = round(e.control.value, 2)
            self.rec_params[key] = int(v) if as_int else v
            value_text.value = f"{self.rec_params[key]:g}{unit}"
            self._safe(value_text)

        return ft.Column(spacing=0, controls=[
            ft.Row(alignment=ft.MainAxisAlignment.SPACE_BETWEEN, controls=[
                ft.Text(title, size=12, color=ft.Colors.ON_SURFACE_VARIANT), value_text]),
            ft.Slider(min=lo, max=hi, divisions=div, value=value, on_change=change)])

    def _hint(self, text: str) -> ft.Control:
        return ft.Container(
            padding=ft.Padding.symmetric(horizontal=10, vertical=8), border_radius=6,
            bgcolor=op(0.06, ft.Colors.ON_SURFACE), border=ft.Border(left=ft.BorderSide(3, self.c1)),
            content=ft.Text(text, size=12, color=ft.Colors.ON_SURFACE_VARIANT))

    def _rec_params_panel(self) -> ft.Control:
        t = self.rec_type
        rows: List[ft.Control] = []
        if t in ("static", "trajectory", "transition", "hold"):
            rows.append(self._slider_row("Поріг схожості", "threshold", 0.5, 0.95, 9))
        if t == "static":
            rows.append(self._hint("Як це працює: ви показуєте позу в камеру, система запам'ятовує положення "
                                   "точок. Жест спрацьовує, коли поточна поза схожа на еталон не менше за поріг."))
        if t == "trajectory":
            rows.append(self._slider_row("Час виконання руху", "traj_time", 0.4, 3.0, 26, " с"))
            rows.append(self._hint("Натисніть ПРОБІЛ і проведіть рукою за вказаний час — це еталонна траєкторія. "
                                   "Під час роботи порівнюється рух за останні N секунд. Швидкий жест — менший час, "
                                   "повільне коло — більший."))
        if t == "transition":
            rows.append(self._slider_row("Макс. час перетворення А → Б", "trans_time", 0.3, 2.0, 17, " с"))
            rows.append(self._hint("Записуються дві пози. Жест спрацьовує, якщо після пози А за вказаний час "
                                   "з'явилась поза Б (кулак → розкриті пальці)."))
        if t == "hold":
            rows.append(self._slider_row("Час утримання", "hold_seconds", 0.5, 5.0, 9, " с"))
            rows.append(self._hint("Тримайте позу без зміни вказаний час — заповниться смуга прогресу і жест спрацює."))
        if t == "periodic":
            rows.append(ft.Text("Вісь коливань", size=12, color=ft.Colors.ON_SURFACE_VARIANT))
            rows.append(ft.SegmentedButton(
                selected=[self.rec_params["axis_choice"]], allow_empty_selection=False,
                segments=[ft.Segment(value="1", label=ft.Text("Горизонтально"), icon=ft.Icon(ft.Icons.SWAP_HORIZ, size=18)),
                          ft.Segment(value="2", label=ft.Text("Вертикально"), icon=ft.Icon(ft.Icons.SWAP_VERT, size=18))],
                on_change=lambda e: self.rec_params.__setitem__("axis_choice", list(e.control.selected)[0])))
            rows.append(self._slider_row("Кількість змін напрямку", "min_cycles", 2, 8, 6, "", as_int=True))
            rows.append(self._slider_row("Час на виконання руху", "period_window", 0.4, 4.0, 36, " с"))
            rows.append(self._slider_row("Мін. розмах руху (частка кадру)", "min_amplitude", 0.03, 0.30, 27))
            rows.append(self._hint("Еталон не записується: система рахує, скільки разів рука змінила напрямок "
                                   "по обраній осі за вказаний час із достатнім розмахом. Помахайте рукою — "
                                   "спрацює, коли набереться потрібна кількість змін."))
        if t == "slider":
            rows.append(ft.Text("Що керує повзунком", size=12, color=ft.Colors.ON_SURFACE_VARIANT))
            rows.append(ft.SegmentedButton(
                selected=[self.rec_params["slider_choice"]], allow_empty_selection=False,
                segments=[ft.Segment(value="1", label=ft.Text("Висота")),
                          ft.Segment(value="2", label=ft.Text("Горизонталь")),
                          ft.Segment(value="3", label=ft.Text("Щипок"))],
                on_change=lambda e: self.rec_params.__setitem__("slider_choice", list(e.control.selected)[0])))
            rows.append(self._hint("Не жест, а регулятор: положення руки (висота / ліво-право) або відстань між "
                                   "великим і вказівним пальцями перетворюється на значення 0–100%. Запис не потрібен — "
                                   "зв'яжіть його в «Зв'язках» з дією «Аналоговий індикатор»."))
        info = PARADIGM_BY_TYPE[t]
        return ft.Container(
            key=f"params_{t}", padding=14, border_radius=8, bgcolor=op(0.08, info["color"]),
            border=ft.Border.all(1, op(0.3, info["color"])),
            content=ft.Column(spacing=10, controls=[
                ft.Row(spacing=8, controls=[ft.Icon(info["icon"], color=info["color"], size=18),
                                            ft.Text(info["title"], size=13, weight=ft.FontWeight.W_700)]), *rows]))

    def _build_preset(self, name: str) -> Dict[str, Any]:
        model_choice = {m[0]: m[1] for m in MODELS}[self.rec_model]
        pr = self.rec_params
        return dict(
            name=name, gesture_choice=PARADIGM_BY_TYPE[self.rec_type]["choice"], model_choice=model_choice,
            threshold=pr["threshold"], trans_time=pr["trans_time"], hold_seconds=pr["hold_seconds"],
            min_cycles=pr["min_cycles"], axis_choice=pr["axis_choice"], slider_choice=pr["slider_choice"],
            traj_time=pr.get("traj_time", 1.0), period_window=pr.get("period_window", 1.0),
            min_amplitude=pr.get("min_amplitude", 0.09))

    def _on_record_click(self, e):
        if self.rec_proc is not None and self.rec_proc.poll() is None:
            return
        raw = (self.rec_name or "").strip()
        name = re.sub(r"[^A-Za-z0-9_]+", "_", raw).strip("_")
        if not name:
            self.rec_name_field.error_text = "Введіть назву латиницею (літери, цифри, _)"
            self._safe(self.rec_name_field)
            self._toast("Спершу введіть назву жесту", "err")
            return
        self.rec_name_field.error_text = None
        self._safe(self.rec_name_field)

        target = os.path.join(RECORDED_DIR, f"{name}.json")
        if self.rec_type == "slider":
            if os.path.exists(target):
                self._confirm("Повзунок уже існує", f"«{name}» буде перезаписано новим значенням. Продовжити?",
                              "Перезаписати", lambda: self._save_slider_preset(name))
            else:
                self._save_slider_preset(name)
            return

        if os.path.exists(target):
            self._confirm("Жест уже існує", f"«{name}» буде перезаписано новим записом. Продовжити?",
                          "Перезаписати", lambda: self._launch_recorder(name))
        else:
            self._launch_recorder(name)

    def _save_slider_preset(self, name: str):
        s_choice = self.rec_params.get("slider_choice", "1")
        s_map = {"1": "height", "2": "horizontal", "3": "pinch"}
        slider_mode = s_map.get(s_choice, "height")
        mode_titles = {
            "height": "Висота руки по вертикалі (0–100%)",
            "horizontal": "Позиція руки по горизонталі (0–100%)",
            "pinch": "Щипок пальцями (0–100%)",
        }
        payload = {
            "name": name,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "gesture_type": "slider",
            "model_type": "hands",
            "slider_mode": slider_mode,
        }
        os.makedirs(RECORDED_DIR, exist_ok=True)
        target = os.path.join(RECORDED_DIR, f"{name}.json")
        with open(target, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        self._toast(f"Повзунок «{name}» збережено ({mode_titles.get(slider_mode, slider_mode)})", "ok")
        self._offer_test(target.replace("\\", "/"), name)

    def _launch_recorder(self, name: str):
        if self._is_running():
            self._stop_pipeline(quiet=True)
            time.sleep(0.6)  # даємо камері звільнитись
        if self._is_testing():
            self._stop_test(quiet=True)
            time.sleep(0.6)
        preset = self._build_preset(name)
        cam = read_cfg().get("settings", {}).get("camera_id", 0)
        started = time.time()
        self.rec_btn.opacity = 0.5
        self.rec_ring.visible = True
        self.rec_status.value = "Запуск студії запису… дивіться у вікно камери."
        self._page_update()

        def on_line(line: str):
            if line.startswith("[Recorder]") or line.startswith(">>>") or line.startswith("ЖЕСТ"):
                self.rec_status.value = line.strip("> ").strip()
                self._safe(self.rec_status)

        def on_exit(proc: subprocess.Popen):
            target = os.path.join(RECORDED_DIR, f"{name}.json")
            ok = proc.returncode == 0 and os.path.exists(target) and os.path.getmtime(target) >= started - 1
            self.rec_btn.opacity = 1
            self.rec_ring.visible = False
            self.rec_status.value = ""
            self._page_update()
            self.rec_proc = None
            if ok:
                self._toast(f"Жест «{name}» записано та додано до бібліотеки", "ok")
                self._offer_test(target.replace("\\", "/"), name)
            else:
                self._toast("Запис скасовано або не вдався", "err")

        self.rec_proc = self._spawn(
            [sys.executable, "-u", "record_pose.py", "--camera", str(cam), "--preset", json.dumps(preset, ensure_ascii=False)],
            on_line, on_exit)

    # ==================================================================
    # ВКЛАДКА 3 — БІБЛІОТЕКА
    # ==================================================================
    def _view_library(self) -> ft.Column:
        search = ft.TextField(
            hint_text="Пошук жесту…", prefix_icon=ft.Icons.SEARCH,
            border=ft.OutlineInputBorder(border_radius=8), expand=True, value=self.lib_query,
            on_change=self._on_lib_search)
        flt = ft.SegmentedButton(
            selected=[self.lib_filter], allow_empty_selection=False,
            segments=[ft.Segment(value="all", label=ft.Text("Усі")),
                      ft.Segment(value="json", label=ft.Text("Зразки"), icon=ft.Icon(ft.Icons.FIBER_SMART_RECORD, size=16)),
                      ft.Segment(value="py", label=ft.Text("Код"), icon=ft.Icon(ft.Icons.CODE, size=16))],
            on_change=self._on_lib_filter)
        self.lib_count = ft.Text("", size=12, color=ft.Colors.ON_SURFACE_VARIANT)
        toolbar = self._glass(
            hover=False, padding=14, content=ft.ResponsiveRow(
                vertical_alignment=ft.CrossAxisAlignment.CENTER, spacing=12, run_spacing=12,
                controls=[
                    ft.Container(col={"xs": 12, "sm": 6, "md": 7}, content=search),
                    ft.Container(col={"xs": 9, "sm": 5, "md": 4}, content=flt),
                    ft.Container(col={"xs": 3, "sm": 1, "md": 1}, alignment=ft.Alignment.CENTER_RIGHT,
                                 content=ft.IconButton(ft.Icons.REFRESH, tooltip="Оновити", on_click=lambda e: self._refresh_library()))
                ]))

        self.test_banner_title = ft.Text("", size=15, weight=ft.FontWeight.W_700)
        self.test_banner_sub = ft.Text("", size=12, color=ft.Colors.ON_SURFACE_VARIANT)
        self.test_banner = self._glass(
            hover=False, tint="#22c55e", visible=self._is_testing(),
            content=ft.Row(spacing=16, vertical_alignment=ft.CrossAxisAlignment.CENTER, controls=[
                ft.ProgressRing(width=28, height=28, stroke_width=3, color="#22c55e"),
                ft.Column(expand=True, spacing=2, controls=[
                    self.test_banner_title, self.test_banner_sub,
                    ft.Text("Покажіть жест у вікні камери. ESC або Q у вікні — завершити.", size=11,
                            color=ft.Colors.ON_SURFACE_VARIANT)]),
                ft.FilledButton("Зупинити тест", icon=ft.Icons.STOP_CIRCLE,
                                style=ft.ButtonStyle(bgcolor="#dc2626", color="white"),
                                on_click=lambda e: self._stop_test())]))
        self._refresh_test_banner_text()

        self.lib_list = ft.Column(spacing=10, horizontal_alignment=ft.CrossAxisAlignment.STRETCH)
        self._refresh_library(update=False)
        return self._view([self.test_banner, toolbar, self.lib_count, self.lib_list])

    def _refresh_test_banner_text(self):
        if self._is_testing():
            self.test_banner_title.value = f"Перевірка жесту: {self.test_name}"
            self.test_banner_sub.value = f"{self.test_last}   |   Спрацювань: {self.test_hits}"

    def _on_lib_search(self, e):
        self.lib_query = e.control.value or ""
        self._refresh_library()

    def _on_lib_filter(self, e):
        self.lib_filter = list(e.control.selected)[0]
        self._refresh_library()

    def _refresh_library(self, update: bool = True):
        items = scan_gestures()
        q = self.lib_query.strip().lower()
        shown = [g for g in items if (self.lib_filter == "all" or g["kind"] == self.lib_filter)
                 and (not q or q in g["name"].lower() or q in g["file"].lower())]
        cards: List[ft.Control] = [self._library_card(g) for g in shown]
        if not cards:
            cards = [self._glass(hover=False, padding=34, content=ft.Column(
                horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=8, controls=[
                    ft.Icon(ft.Icons.SEARCH_OFF, size=46, color=ft.Colors.ON_SURFACE_VARIANT),
                    ft.Text("Нічого не знайдено", size=16, weight=ft.FontWeight.W_700),
                    ft.Text("Запишіть новий рух на вкладці «Запис» або змініть фільтр.", size=12,
                            color=ft.Colors.ON_SURFACE_VARIANT)]))]
        self._prepare_enter(cards)
        self.lib_list.controls = cards
        if self.lib_count is not None:
            self.lib_count.value = f"Показано {len(shown)} із {len(items)}"
        if update:
            self._safe(self.lib_list)
            self._safe(self.lib_count)
        self._play_enter(cards, delay=0.04)

    def _library_card(self, g: Dict[str, Any]) -> ft.Control:
        if g["kind"] == "py":
            badge = self._icon_badge(ft.Icons.CODE, "#38bdf8")
            title = g["file"]
            sub = ft.Text("Алгоритмічний модуль Python", size=12, color=ft.Colors.ON_SURFACE_VARIANT, no_wrap=True, overflow=ft.TextOverflow.ELLIPSIS)
            pills = [self._pill(".PY", "#38bdf8", ft.Icons.CODE)]
            trailing: ft.Control = ft.Icon(ft.Icons.LOCK_OUTLINE, size=18, color=ft.Colors.ON_SURFACE_VARIANT, tooltip="Системний модуль")
            tint = "#38bdf8"
        else:
            p = PARADIGM_BY_TYPE.get(g["gesture_type"], PARADIGM_BY_TYPE["static"])
            badge = self._icon_badge(p["icon"], p["color"])
            title = g["name"]
            sub = ft.Text(g["file"], size=12, color=ft.Colors.ON_SURFACE_VARIANT, no_wrap=True, overflow=ft.TextOverflow.ELLIPSIS)
            pills = [self._pill(p["title"], p["color"], p["icon"]), self._pill(str(g["model_type"]), "#94a3b8")]
            if g.get("threshold") is not None:
                pills.append(self._pill(f"поріг {g['threshold']}", "#94a3b8"))
            if g.get("created"):
                c_val = g["created"]
                if isinstance(c_val, (int, float)):
                    c_str = time.strftime("%d.%m.%Y %H:%M", time.localtime(c_val))
                else:
                    c_str = str(c_val)
                pills.append(self._pill(c_str, "#94a3b8", ft.Icons.SCHEDULE))
            trailing = ft.IconButton(ft.Icons.DELETE_OUTLINE, icon_color="#f87171", tooltip="Видалити зразок",
                                     on_click=lambda e, gg=g: self._ask_delete(gg))
            tint = p["color"]

        testing_this = self.test_proc is not None and self.test_proc.poll() is None and self.test_target == g["path"]
        test_btn = ft.FilledTonalButton(
            "Стоп" if testing_this else "Перевірити",
            icon=ft.Icons.STOP_CIRCLE if testing_this else ft.Icons.PLAY_CIRCLE,
            tooltip="Перевірити жест наживо перед камерою",
            on_click=lambda e, gg=g: self._toggle_test(gg))

        action_controls = [test_btn]
        if g["kind"] == "json":
            action_controls.append(ft.IconButton(
                ft.Icons.CODE, icon_color=self.c1, tooltip="Експортувати код жесту",
                on_click=lambda e, gg=g: self._export_single_gesture(gg["path"])))
        action_controls.append(trailing)

        actions = ft.Row(
            spacing=4, tight=True, vertical_alignment=ft.CrossAxisAlignment.CENTER,
            controls=action_controls)

        title_ctrl = ft.Text(title, size=15, weight=ft.FontWeight.W_700, no_wrap=True, overflow=ft.TextOverflow.ELLIPSIS)
        pills_row = ft.Row(wrap=True, spacing=6, run_spacing=6, controls=pills)

        info_col = ft.Column(
            expand=True, spacing=4, tight=True,
            controls=[title_ctrl, sub, pills_row])

        return self._glass(
            tint=tint, blur=False, padding=ft.Padding.symmetric(horizontal=14, vertical=10), radius=8,
            animate_opacity=ft.Animation(250, ft.AnimationCurve.EASE_OUT),
            content=ft.Row(
                spacing=12, vertical_alignment=ft.CrossAxisAlignment.CENTER,
                controls=[badge, info_col, actions]))

    # ------------------------------------------------------------------
    # Перевірка жесту наживо
    # ------------------------------------------------------------------
    def _is_testing(self) -> bool:
        return self.test_proc is not None and self.test_proc.poll() is None

    def _toggle_test(self, g: Dict[str, Any]):
        if self._is_testing() and self.test_target == g["path"]:
            self._stop_test()
        else:
            self._start_test(g["path"], g["name"])

    def _start_test(self, path: str, name: str):
        if self._is_testing():
            self._stop_test(quiet=True)
        if self._is_running():
            self._stop_pipeline(quiet=True)
        if self.rec_proc is not None and self.rec_proc.poll() is None:
            self._toast("Спершу завершіть запис жесту", "err")
            return
        time.sleep(0.4)  # даємо камері звільнитись
        cam = read_cfg().get("settings", {}).get("camera_id", 0)
        self.test_target, self.test_name = path, name
        self.test_hits = 0
        self.test_last = "Завантаження моделей та камери…"

        def on_line(line: str):
            if line.startswith("[Test]"):
                self.test_last = line.replace("[Test]", "").strip()
                if "спрацював" in line:
                    self.test_hits += 1
                self._refresh_test_banner()
            self._push_log(line)

        def on_exit(proc: subprocess.Popen):
            if self.test_proc is proc:
                self.test_proc = None
                self.test_target = None
                self._refresh_test_banner()
                self._refresh_library_safe()
                if proc.returncode not in (0, None):
                    self._toast("Тест завершився з помилкою — див. консоль на вкладці «Конвеєр»", "err")

        self.test_proc = self._spawn([sys.executable, "-u", "test_gesture.py", path, "--camera", str(cam)],
                                     on_line, on_exit)
        self._refresh_test_banner()
        self._refresh_library_safe()

    def _stop_test(self, quiet: bool = False):
        proc = self.test_proc
        self.test_proc = None
        self.test_target = None
        if proc is not None:
            try:
                proc.terminate()
            except Exception:
                pass
        self._refresh_test_banner()
        self._refresh_library_safe()
        if not quiet:
            self._toast("Перевірку зупинено")

    def _refresh_library_safe(self):
        if self.tab == 2 and self.lib_list is not None:
            self._refresh_library()

    def _refresh_test_banner(self):
        if self.test_banner is None:
            return
        testing = self._is_testing()
        self.test_banner.visible = testing
        if testing:
            self.test_banner_title.value = f"Перевірка жесту: {self.test_name}"
            self.test_banner_sub.value = f"{self.test_last}   |   Спрацювань: {self.test_hits}"
        self._safe(self.test_banner)

    def _offer_test(self, path: str, name: str):
        """Після запису пропонуємо одразу перевірити жест."""
        def close(_):
            self.page.pop_dialog()

        def go(_):
            self.page.pop_dialog()
            self._start_test(path, name)

        try:
            self.page.show_dialog(ft.AlertDialog(
                modal=True, shape=ft.RoundedRectangleBorder(radius=22),
                title=ft.Row(spacing=10, controls=[ft.Icon(ft.Icons.CHECK_CIRCLE, color="#22c55e"),
                                                   ft.Text(f"Жест «{name}» записано")]),
                content=ft.Text("Перевірити його наживо перед камерою? Спрацювання відображається зеленим спалахом "
                                "і лічильником, смуга показує схожість."),
                actions=[ft.TextButton("Пізніше", on_click=close),
                         ft.FilledButton("Перевірити зараз", icon=ft.Icons.PLAY_CIRCLE, on_click=go)],
                actions_alignment=ft.MainAxisAlignment.END))
        except Exception:
            pass

    def _ask_delete(self, g: Dict[str, Any]):
        self._confirm("Видалити жест?", f"Зразок «{g['name']}» ({g['file']}) буде видалено назавжди.", "Видалити",
                      lambda: self._delete_gesture(g))

    def _delete_gesture(self, g: Dict[str, Any]):
        try:
            os.remove(g["path"])
        except FileNotFoundError:
            pass
        except Exception as ex:
            self._toast(f"Не вдалося видалити: {ex}", "err")
            return
        self._toast(f"Жест «{g['name']}» видалено", "ok")
        self._refresh_library()

    # ==================================================================
    # ВКЛАДКА 4 — ЄДИНИЙ МУЛЬТИМОДАЛЬНИЙ КОНСТРУКТОР ЗВ'ЯЗКІВ
    # ==================================================================
    def _collect_multimodal_triggers(self) -> List[Dict[str, str]]:
        """Збирає всі доступні тригери: Жести (.json/.py) + Обличчя (Face ID) + Голос (Voice ID) + Звуки (YAMNet)."""
        triggers: List[Dict[str, str]] = []
        for g in scan_gestures():
            prefix = "[ЖЕСТ .PY] " if g["kind"] == "py" else "[ЖЕСТ ЗРАЗОК] "
            triggers.append({"key": g["path"], "label": prefix + g["name"], "kind": "gesture"})

        # Додаємо профілі облич Face ID
        if self.face_engine:
            for fp in self.face_engine.load_profiles():
                fname = fp.get("name", "User")
                triggers.append({"key": f"face:{fname}", "label": f"[ОБЛИЧЧЯ Face ID] З'явився: {fname}", "kind": "face"})
        triggers.append({"key": "face:*", "label": "[ОБЛИЧЧЯ Face ID] Будь-яка авторизована особа", "kind": "face"})

        # Додаємо профілі дикторів Voice ID
        if self.audio_mgr:
            for vp in self.audio_mgr.voice_id.load_profiles():
                vname = vp.get("name", "Speaker")
                triggers.append({"key": f"voice:{vname}", "label": f"[ГОЛОС Voice ID] Говорить диктор: {vname}", "kind": "voice"})

        # Додаємо акустичні тригери Google YAMNet
        yamnet_presets = [
            ("Clapping", "Плескання в долоні (Clapping)"),
            ("Finger snapping", "Клацання пальцями (Finger snapping)"),
            ("Whistling", "Свист (Whistling)"),
            ("Laughter", "Сміх (Laughter)"),
            ("Speech", "Людська мова (Speech)"),
            ("Knock", "Стукіт у двері (Knock)"),
        ]
        for eng, ua in yamnet_presets:
            triggers.append({"key": f"yamnet:{eng}", "label": f"[ЗВУК YAMNet] {ua}", "kind": "yamnet"})

        return triggers

    @staticmethod
    def _format_trigger_pill(spec: str) -> Tuple[str, str, Any]:
        """Повертає (короткий підпис, колір, іконку) для будь-якого мультимодального тригера."""
        s = str(spec or "—")
        if s.startswith("face:"):
            who = s.split(":", 1)[1]
            return (f"Face ID: {'Всі свої' if who == '*' else who}", "#38bdf8", ft.Icons.FACE_RETOUCHING_NATURAL)
        if s.startswith("voice:"):
            who = s.split(":", 1)[1]
            return (f"Voice ID: {who}", "#22c55e", ft.Icons.RECORD_VOICE_OVER)
        if s.startswith("yamnet:"):
            snd = s.split(":", 1)[1]
            return (f"YAMNet: {snd}", "#f59e0b", ft.Icons.WAVES)
        return (os.path.splitext(os.path.basename(s))[0], "#a855f7", ft.Icons.GESTURE)

    def _view_bindings(self) -> ft.Column:
        triggers = self._collect_multimodal_triggers()
        actions = scan_actions()

        valid_g = {t["key"] for t in triggers}
        if self.bld["gesture"] not in valid_g:
            self.bld["gesture"] = triggers[0]["key"] if triggers else None
        valid_a = {a["path"] for a in actions}
        if self.bld["action"] not in valid_a:
            self.bld["action"] = actions[0]["path"] if actions else None

        g_dd = ft.Dropdown(
            label="Мультимодальний тригер (Жест / Обличчя Face ID / Голос Voice ID / Звук YAMNet)",
            value=self.bld["gesture"], border=ft.OutlineInputBorder(border_radius=8), expand=True,
            options=[ft.dropdown.Option(key=t["key"], text=t["label"]) for t in triggers],
            on_select=lambda e: self._on_bld("gesture", e.control.value))
        a_dd = ft.Dropdown(
            label="Дія при спрацюванні", value=self.bld["action"], border=ft.OutlineInputBorder(border_radius=8), expand=True,
            options=[ft.dropdown.Option(key=a["path"], text=a["label"]) for a in actions],
            on_select=lambda e: self._on_bld("action", e.control.value))
        self.key_field = ft.TextField(
            label="Клавіша клавіатури", value=self.bld["key"], prefix_icon=ft.Icons.KEYBOARD,
            border=ft.OutlineInputBorder(border_radius=8),
            hint_text="space, enter, up, down, w, a, s, d…", on_change=lambda e: self._on_bld("key", e.control.value))
        self.key_field.visible = "keyboard" in str(self.bld["action"])

        cd_text = ft.Text(f"{self.bld['cooldown']:g} с", size=13, weight=ft.FontWeight.W_700, color=self.c1, font_family="Consolas")

        def cd_change(e):
            self.bld["cooldown"] = round(e.control.value, 2)
            cd_text.value = f"{self.bld['cooldown']:g} с"
            self._safe(cd_text)

        cooldown = ft.Column(spacing=0, controls=[
            ft.Row(alignment=ft.MainAxisAlignment.SPACE_BETWEEN, controls=[
                ft.Text("Пауза між спрацюваннями (cooldown)", size=12, color=ft.Colors.ON_SURFACE_VARIANT), cd_text]),
            ft.Slider(min=0, max=2, divisions=20, value=self.bld["cooldown"], on_change=cd_change)])

        self.flow_row = ft.Row(alignment=ft.MainAxisAlignment.CENTER, spacing=12, wrap=True)
        self._refresh_flow(update=False)

        builder = self._glass(
            hover=False, radius=10, col={"xs": 12, "lg": 6},
            content=ft.Column(spacing=14, controls=[
                self._section_title(
                    "Єдиний мультимодальний конструктор зв'язок",
                    ft.Icons.ADD_LINK,
                    "Прив'язуйте Жести, Обличчя (Face ID), Голос (Voice ID) або Звуки (YAMNet: оплески, свист) до будь-якої дії"),
                self._glass(hover=False, blur=False, radius=8, padding=12, content=self.flow_row),
                ft.Row(controls=[g_dd]), ft.Row(controls=[a_dd]), self.key_field, cooldown,
                self._gradient_button("Зберегти мультимодальну зв'язку", ft.Icons.SAVE, self._save_binding)]))

        self.bind_list = ft.Column(spacing=10, horizontal_alignment=ft.CrossAxisAlignment.STRETCH)
        self._refresh_bindings(update=False)
        active = self._glass(
            hover=False, radius=10, col={"xs": 12, "lg": 6},
            content=ft.Column(spacing=14, controls=[
                self._section_title("Активні зв'язки конвеєра", ft.Icons.HUB, "Виконуються в реальному часі при запуску конвеєра"), self.bind_list]))

        return self._view([ft.ResponsiveRow(spacing=16, run_spacing=16, controls=[builder, active])])

    def _on_bld(self, key: str, value: Any):
        self.bld[key] = value
        if key == "action" and self.key_field is not None:
            self.key_field.visible = "keyboard" in str(value)
            self._safe(self.key_field)
        if key in ("gesture", "action", "key"):
            self._refresh_flow()

    def _refresh_flow(self, update: bool = True):
        if self.flow_row is None:
            return
        t_lbl, t_col, t_icon = self._format_trigger_pill(str(self.bld["gesture"] or "—"))
        a = os.path.splitext(os.path.basename(str(self.bld["action"] or "—")))[0]
        label, icon = ACTION_LABELS.get(a, (a, ft.Icons.BOLT))
        if "keyboard" in a:
            label = f"{label}: {self.bld['key'] or '?'}"
        gesture_chip = self._pill(t_lbl, t_col, t_icon)
        action_chip = self._pill(label, self.c1, icon)
        for chip in (gesture_chip, action_chip):
            chip.scale = 1.15
        self.flow_row.controls = [gesture_chip, ft.Icon(ft.Icons.ARROW_FORWARD_ROUNDED, color=ft.Colors.ON_SURFACE_VARIANT), action_chip]
        if update:
            self._safe(self.flow_row)

    def _refresh_bindings(self, update: bool = True):
        bindings = read_cfg().get("bindings", []) or []
        items: List[ft.Control] = []
        if not bindings:
            items.append(ft.Container(padding=24, alignment=ft.Alignment.CENTER, content=ft.Column(
                horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=6, controls=[
                    ft.Icon(ft.Icons.LINK_OFF, size=40, color=ft.Colors.ON_SURFACE_VARIANT),
                    ft.Text("Зв'язок ще немає", weight=ft.FontWeight.W_700),
                    ft.Text("Створіть першу в конструкторі ліворуч (можна обрати жест, обличчя або звук).", size=12, color=ft.Colors.ON_SURFACE_VARIANT)])))
        for idx, b in enumerate(bindings):
            raw_g = b.get("gesture", {}).get("file", "")
            t_lbl, t_col, t_icon = self._format_trigger_pill(raw_g)
            a = os.path.splitext(os.path.basename(b.get("action", {}).get("file", "")))[0]
            label, icon = ACTION_LABELS.get(a, (a, ft.Icons.BOLT))
            key = b.get("action", {}).get("config", {}).get("key")
            cd = b.get("filter", {}).get("cooldown")
            actions = ft.Row(
                spacing=4, tight=True, vertical_alignment=ft.CrossAxisAlignment.CENTER,
                controls=[ft.IconButton(ft.Icons.DELETE_OUTLINE, icon_color="#f87171", tooltip="Видалити зв'язку",
                                        on_click=lambda e, i=idx: self._delete_binding(i))])
            row = self._glass(
                blur=False, radius=8, padding=ft.Padding.symmetric(horizontal=12, vertical=10),
                animate_opacity=ft.Animation(250, ft.AnimationCurve.EASE_OUT),
                content=ft.Row(spacing=12, vertical_alignment=ft.CrossAxisAlignment.CENTER, controls=[
                    self._icon_badge(t_icon, t_col, 38, 20),
                    ft.Column(expand=True, spacing=4, tight=True, controls=[
                        ft.Row(wrap=True, spacing=8, run_spacing=6, controls=[
                            self._pill(t_lbl, t_col, t_icon),
                            ft.Icon(ft.Icons.ARROW_FORWARD_ROUNDED, size=16, color=ft.Colors.ON_SURFACE_VARIANT),
                            self._pill(f"{label}: {key}" if key else label, self.c1, icon)]),
                        ft.Text(f"cooldown {cd} с" if cd is not None else "", size=11, color=ft.Colors.ON_SURFACE_VARIANT)]),
                    actions]))
            items.append(row)
        self.bind_list.controls = items
        if update:
            self._safe(self.bind_list)

    def _delete_binding(self, index: int):
        cfg = read_cfg()
        bindings = cfg.get("bindings", []) or []
        if 0 <= index < len(bindings):
            removed = bindings.pop(index)
            cfg["bindings"] = bindings
            write_cfg(cfg)
            self._refresh_bindings()
            self._toast(f"Зв'язку «{removed.get('name', index + 1)}» видалено", "ok")

    def _save_binding(self, e):
        g_file, a_file = self.bld["gesture"], self.bld["action"]
        if not g_file or not a_file:
            self._toast("Оберіть і тригер, і дію", "err")
            return
        cfg = read_cfg()
        bindings = cfg.get("bindings", []) or []
        t_lbl, _, _ = self._format_trigger_pill(g_file)
        a_name = os.path.splitext(os.path.basename(a_file))[0]
        action_config: Dict[str, Any] = {}
        if "keyboard" in a_name:
            action_config["key"] = (self.bld["key"] or "space").strip() or "space"
        bindings.append({
            "id": f"bind_{len(bindings) + 1}",
            "name": f"{t_lbl} -> {a_name}",
            "gesture": {"file": g_file, "config": {}},
            "filter": {"cooldown": float(self.bld["cooldown"]), "trigger_on": "rising"},
            "action": {"file": a_file, "config": action_config},
        })
        cfg["bindings"] = bindings
        write_cfg(cfg)
        self._refresh_bindings()
        self._toast(f"Мультимодальну зв'язку «{t_lbl}» збережено у pipeline.yaml", "ok")

    # ==================================================================
    # МОДАЛЬНІСТЬ 2 — ОБЛИЧЧЯ (FACE ID + ВБУДОВАНЕ ПРЕВ'Ю + LIVENESS)
    # ==================================================================
    def _view_face(self) -> ft.Column:
        guide = self._guide_banner([
            ("1", "Увімкніть прев'ю або введіть ім'я", "Камера працює прямо всередині вікна Flet з діагностикою освітлення та відстані."),
            ("2", "Захопіть 5 зрізів обличчя", "Коли індикатор покаже «Ідеальна позиція», натисніть кнопку зйомки — створиться 128-d профіль."),
            ("3", "Перевірте Liveness (Анти-Фото)", "Увімкніть захист від фотографії з телефона — система вимагатиме легкого повороту голови."),
        ])

        self.face_name_field = ft.TextField(
            label="Ім'я особи (наприклад: Max, Олександр, Адміністратор)",
            hint_text="Введіть ім'я та дивіться прямо в об'єктив камери...",
            prefix_icon=ft.Icons.BADGE, border=ft.OutlineInputBorder(border_radius=8), value=self.face_reg_name,
            on_change=lambda e: setattr(self, "face_reg_name", e.control.value))

        self.face_reg_ring = ft.ProgressRing(width=22, height=22, stroke_width=3, visible=False)
        self.face_reg_btn = self._gradient_button("Захопити обличчя з камери (5 зрізів)", ft.Icons.CAMERA_ALT, self._on_face_register_click)
        self.face_status_text = ft.Text("Подивіться прямо в камеру при нормальному освітленні (відстань 40–80 см).", size=12, color=ft.Colors.ON_SURFACE_VARIANT)

        # Індикатор якості освітлення та відстані в реальному часі
        self.face_light_pill = self._pill("Освітлення: Очікування камери", "#94a3b8", ft.Icons.WB_SUNNY_OUTLINED)
        self.face_dist_pill = self._pill("Відстань: Очікування камери", "#94a3b8", ft.Icons. STRAIGHTEN)
        self.face_quality_text = ft.Text("Увімкніть вбудований сканер праворуч для оцінки світла та позиції наживо", size=12, weight=ft.FontWeight.W_600)
        self.face_quality_badge = self._glass(
            hover=False, blur=False, radius=8, padding=12,
            content=ft.Column(spacing=6, controls=[
                ft.Row(spacing=8, wrap=True, controls=[
                    ft.Icon(ft.Icons.TIPS_AND_UPDATES, size=18, color=self.c1),
                    ft.Text("Діагностика кадру (Освітлення та Відстань):", size=12, weight=ft.FontWeight.W_700),
                    self.face_light_pill,
                    self.face_dist_pill]),
                self.face_quality_text]))

        # Прев'ю захоплення під час реєстрації
        self.face_reg_preview_img = ft.Image(
            src=BLANK_FRAME_JPEG, width=440, height=260, fit="contain",
            gapless_playback=True, border_radius=8, visible=False)
        self.face_reg_preview_box = ft.Container(
            content=self.face_reg_preview_img,
            alignment=ft.Alignment.CENTER,
            visible=False,
            height=260,
            border_radius=8,
            bgcolor=op(0.20, ft.Colors.SURFACE_CONTAINER_HIGHEST),
            border=ft.Border.all(1.5, self.c1))

        reg_card = self._glass(
            hover=False, radius=10,
            content=ft.Column(spacing=14, controls=[
                self._section_title("1. Реєстрація в Face DB", ft.Icons.PERSON_ADD, "Створення мульти-кадрового біометричного дескриптора (128-d SFace)"),
                self.face_quality_badge,
                self.face_name_field,
                ft.Row(spacing=12, controls=[self.face_reg_btn, self.face_reg_ring]),
                self.face_status_text,
                self.face_reg_preview_box,
                self._hint("Порада: Під час зйомки дивіться прямо в камеру. Вікно вище покаже прев'ю процесу в реальному часі.")]))

        self.face_profiles_list = ft.Column(spacing=8, horizontal_alignment=ft.CrossAxisAlignment.STRETCH)
        self._refresh_face_profiles(update=False)

        profiles_card = self._glass(
            hover=False, radius=10,
            content=ft.Column(spacing=12, controls=[
                ft.Row(alignment=ft.MainAxisAlignment.SPACE_BETWEEN, controls=[
                    self._section_title("2. Зареєстровані особи", ft.Icons.PEOPLE_ALT, "База облич для зіставлення"),
                    ft.OutlinedButton("Експорт Face ID", icon=ft.Icons.FOLDER_ZIP,
                                      on_click=lambda e: self._open_export_modal(default_modality="face"))]),
                self.face_profiles_list]))

        left = ft.Column(spacing=16, col={"xs": 12, "lg": 6}, controls=[reg_card, profiles_card])

        scanning = self.face_embedded_running
        self.face_scan_ring = ft.ProgressRing(width=22, height=22, stroke_width=3, visible=scanning)
        self.face_scan_btn = self._gradient_button(
            "Зупинити вбудовану камеру" if scanning else "Увімкнути вбудований сканер",
            ft.Icons.STOP if scanning else ft.Icons.VIDEOCAM,
            self._toggle_face_scan,
            colors=["#dc2626", "#b91c1c"] if scanning else [self.c1, self.c2])

        # Вбудоване прев'ю камери всередині картки Flet
        self.face_preview_img = ft.Image(
            src=BLANK_FRAME_JPEG, width=520, height=320, fit="contain",
            gapless_playback=True, border_radius=10, visible=scanning)
        self.face_preview_placeholder = ft.Container(
            height=320, border_radius=10, alignment=ft.Alignment.CENTER, visible=not scanning,
            bgcolor=op(0.18, ft.Colors.SURFACE_CONTAINER_HIGHEST),
            border=ft.Border.all(1, op(0.14, ft.Colors.ON_SURFACE)),
            content=ft.Column(horizontal_alignment=ft.CrossAxisAlignment.CENTER, alignment=ft.MainAxisAlignment.CENTER, spacing=8, controls=[
                ft.Icon(ft.Icons.VIDEOCAM_OFF_OUTLINED, size=44, color=ft.Colors.ON_SURFACE_VARIANT),
                ft.Text("Вбудоване прев'ю камери вимкнено", size=14, weight=ft.FontWeight.W_700),
                ft.Text("Натисніть «Увімкнути вбудований сканер» — відео з'явиться прямо тут без зовнішніх вікон OpenCV", size=11, color=ft.Colors.ON_SURFACE_VARIANT, text_align=ft.TextAlign.CENTER)]))

        self.face_live_title = ft.Text("Сканер не активний" if not scanning else "Очікування обличчя...", size=17, weight=ft.FontWeight.W_800)
        self.face_live_sub = ft.Text("Натисніть кнопку для запуску вбудованого прев'ю наживо." if not scanning else "Дивіться в об'єктив камери", size=12, color=ft.Colors.ON_SURFACE_VARIANT)

        self.face_live_badge = self._glass(
            hover=False, radius=10, padding=14,
            content=ft.Row(spacing=14, vertical_alignment=ft.CrossAxisAlignment.CENTER, controls=[
                self._icon_badge(ft.Icons.SECURITY, self.c1, size=42, icon_size=22),
                ft.Column(spacing=2, expand=True, controls=[self.face_live_title, self.face_live_sub])]))

        # Liveness / Anti-Spoofing (Захист від фото з телефона)
        def on_liveness_toggle(e):
            self.face_liveness_required = bool(e.control.value)
            if self.face_engine:
                self.face_engine.liveness_required = self.face_liveness_required
                self.face_engine.reset_liveness()
            self._toast("Захист від фото (Liveness 3D) увімкнено" if self.face_liveness_required else "Перевірку Liveness вимкнено", "info")

        liveness_sw = ft.Switch(
            label="Захист від фото з телефона (3D Liveness Anti-Spoofing)",
            value=self.face_liveness_required, on_change=on_liveness_toggle)
        self.face_liveness_bar = ft.ProgressBar(value=0.0, height=6, border_radius=3, color="#22c55e", bgcolor=op(0.12, ft.Colors.ON_SURFACE))
        self.face_liveness_text = ft.Text("Статус живості (Liveness): Очікування 3D-руху голови...", size=11, color=ft.Colors.ON_SURFACE_VARIANT)

        liveness_box = self._glass(
            hover=False, blur=False, padding=12, radius=8,
            content=ft.Column(spacing=6, controls=[
                ft.Row(alignment=ft.MainAxisAlignment.SPACE_BETWEEN, wrap=True, controls=[
                    liveness_sw,
                    ft.TextButton("Скинути Liveness", icon=ft.Icons.REFRESH,
                                  on_click=lambda e: (self.face_engine.reset_liveness() if self.face_engine else None))]),
                self.face_liveness_text,
                self.face_liveness_bar]))

        curr_f_thresh = self.face_engine.cosine_thresh if self.face_engine else 0.363
        f_thresh_lbl = ft.Text(f"Поріг суворості Face ID (Cosine Similarity): {curr_f_thresh:.2f}", size=12, weight=ft.FontWeight.W_600)

        def on_face_thresh(e):
            val = round(float(e.control.value), 2)
            if self.face_engine:
                self.face_engine.cosine_thresh = val
            f_thresh_lbl.value = f"Поріг суворості Face ID (Cosine Similarity): {val:.2f}"
            self._safe(f_thresh_lbl)

        f_thresh_slider = ft.Slider(
            min=0.25, max=0.60, divisions=35, value=curr_f_thresh,
            active_color=self.c1, on_change=on_face_thresh,
            tooltip="Вище значення = суворіша перевірка (менший ризик хибного спрацювання)")

        right = self._glass(
            hover=False, radius=10, col={"xs": 12, "lg": 6},
            content=ft.Column(spacing=14, controls=[
                self._section_title("3. Вбудований живий сканер та Liveness", ft.Icons.SECURITY, "Відеопотік YuNet + SFace прямо всередині вікна Flet"),
                ft.Row(spacing=12, controls=[self.face_scan_btn, self.face_scan_ring]),
                ft.Container(alignment=ft.Alignment.CENTER, content=ft.Column(
                    horizontal_alignment=ft.CrossAxisAlignment.CENTER, controls=[self.face_preview_placeholder, self.face_preview_img])),
                self.face_live_badge,
                liveness_box,
                ft.Column(spacing=2, controls=[
                    f_thresh_lbl,
                    f_thresh_slider,
                    ft.Text("Рекомендовано: 0.36 (баланс точності)  •  0.45+ (висока безпека для робота)", size=11, color=ft.Colors.ON_SURFACE_VARIANT)])]))

        return self._view([guide, ft.ResponsiveRow(spacing=16, run_spacing=16, controls=[left, right])])

    def _refresh_face_profiles(self, update: bool = True):
        if self.face_engine is None or self.face_profiles_list is None:
            return
        profiles = self.face_engine.load_profiles()
        items: List[ft.Control] = []
        if not profiles:
            items.append(ft.Container(padding=16, alignment=ft.Alignment.CENTER, content=ft.Column(
                horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=6, controls=[
                    ft.Icon(ft.Icons.NO_ACCOUNTS, size=36, color=ft.Colors.ON_SURFACE_VARIANT),
                    ft.Text("Осіб ще не зареєстровано", size=13, weight=ft.FontWeight.W_600),
                    ft.Text("Введіть ім'я вище та захопіть обличчя з камери.", size=11, color=ft.Colors.ON_SURFACE_VARIANT)])))
        for prof in profiles:
            p_id = prof.get("id", "")
            p_name = prof.get("name", "Без імені")
            p_date = prof.get("created_at", "")
            p_crop = prof.get("crop_path", "")
            p_cnt = prof.get("samples_count", len(prof.get("features", [])) or 1)

            avatar: ft.Control
            if p_crop and os.path.exists(p_crop):
                avatar = ft.Image(src=p_crop, width=38, height=38, border_radius=6, fit="cover")
            else:
                avatar = self._icon_badge(ft.Icons.PERSON, self.c1, 38, 20)

            row = self._glass(
                blur=False, radius=8, padding=ft.Padding.symmetric(horizontal=12, vertical=8),
                content=ft.Row(spacing=12, vertical_alignment=ft.CrossAxisAlignment.CENTER, controls=[
                    avatar,
                    ft.Column(expand=True, spacing=2, tight=True, controls=[
                        ft.Row(spacing=8, controls=[
                            ft.Text(p_name, size=13, weight=ft.FontWeight.W_700),
                            self._pill(f"{p_cnt} зріз(ів)", "#22c55e", ft.Icons.VERIFIED)]),
                        ft.Text(f"ID: {p_id}  •  {p_date}", size=11, font_family="Consolas", color=ft.Colors.ON_SURFACE_VARIANT)]),
                    ft.IconButton(ft.Icons.DELETE_OUTLINE, icon_color="#f87171", tooltip="Видалити профіль",
                                  on_click=lambda e, pid=p_id, n=p_name: self._confirm(
                                      "Видалити профіль", f"Видалити «{n}» з Face DB?", "Видалити",
                                      lambda: self._delete_face_profile(pid)))]))
            items.append(row)
        self.face_profiles_list.controls = items
        if update:
            self._safe(self.face_profiles_list)

    def _delete_face_profile(self, profile_id: str):
        if self.face_engine and self.face_engine.delete_profile(profile_id):
            self._toast("Профіль обличчя успішно видалено", "ok")
            self._refresh_face_profiles()

    def _update_face_quality_ui(self, q: Dict[str, Any]):
        if self.face_quality_text is not None:
            self.face_quality_text.value = q.get("hint", "")
            self.face_quality_text.color = q.get("color", ft.Colors.ON_SURFACE)
        if self.face_light_pill is not None:
            l_col = "#22c55e" if q.get("light_ok") else "#f59e0b"
            self.face_light_pill.bgcolor = op(0.14, l_col)
            self.face_light_pill.border = ft.Border.all(1, op(0.32, l_col))
            try:
                self.face_light_pill.content.controls[-1].value = f"Світло: {q.get('light_status', '')} ({q.get('brightness', 0)})"
                self.face_light_pill.content.controls[-1].color = l_col
            except Exception:
                pass
        if self.face_dist_pill is not None:
            d_col = "#22c55e" if q.get("distance_ok") else "#f59e0b"
            self.face_dist_pill.bgcolor = op(0.14, d_col)
            self.face_dist_pill.border = ft.Border.all(1, op(0.32, d_col))
            try:
                self.face_dist_pill.content.controls[-1].value = f"Дистанція: {q.get('distance_status', '')}"
                self.face_dist_pill.content.controls[-1].color = d_col
            except Exception:
                pass
        self._safe(self.face_quality_badge)

    def _on_face_register_click(self, e):
        name = (self.face_reg_name or "").strip()
        if not name:
            self._toast("Введіть ім'я особи для реєстрації", "err")
            return
        cam_id = read_cfg().get("settings", {}).get("camera_id", 0)
        was_scanning = self.face_embedded_running
        if was_scanning:
            self.face_embedded_running = False
            time.sleep(0.25)

        if self.face_reg_preview_box:
            self.face_reg_preview_box.visible = True
            self._safe(self.face_reg_preview_box)
        if self.face_reg_preview_img:
            self.face_reg_preview_img.visible = True
            self._safe(self.face_reg_preview_img)

        self.face_reg_ring.visible = True
        self.face_status_text.value = "Серійна зйомка (5 біометричних зрізів з камери + перевірка якості)..."
        self._safe(self.face_reg_ring)
        self._safe(self.face_status_text)

        def run():
            cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap = cv2.VideoCapture(cam_id)
            if not cap.isOpened():
                self.face_status_text.value = f"Помилка: не вдалося відкрити камеру ID={cam_id}"
                self.face_reg_ring.visible = False
                self._safe(self.face_status_text)
                self._safe(self.face_reg_ring)
                return

            frames: List[np.ndarray] = []
            warmup_count = 10
            # Попередній прогрів та показ камери оператору (Warmup)
            for w_idx in range(warmup_count):
                ret, frm = cap.read()
                if ret and frm is not None:
                    flipped = cv2.flip(frm, 1)
                    disp = flipped.copy()
                    if self.face_engine:
                        q = self.face_engine.assess_frame_quality(flipped)
                        self._update_face_quality_ui(q)
                        faces = self.face_engine.detect_raw(flipped)
                        if faces is not None and len(faces) > 0:
                            for f in faces:
                                fx, fy, fw, fh = [int(v) for v in f[:4]]
                                cv2.rectangle(disp, (fx, fy), (fx + fw, fy + fh), (0, 200, 255), 2)
                    cv2.putText(disp, f"PIDSOTUVANNYA: {warmup_count - w_idx}", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 220, 255), 2)
                    uri = frame_to_data_uri(disp, quality=80)
                    if uri and self.face_reg_preview_img:
                        self.face_reg_preview_img.src = uri
                        self.face_reg_preview_img.visible = True
                        self._safe(self.face_reg_preview_img)
                time.sleep(0.08)

            # Серійна зйомка 5 якісних зрізів
            for idx in range(5):
                ret, frm = cap.read()
                if ret and frm is not None:
                    flipped = cv2.flip(frm, 1)
                    frames.append(flipped)
                    disp = flipped.copy()
                    if self.face_engine:
                        last_q = self.face_engine.assess_frame_quality(flipped)
                        self._update_face_quality_ui(last_q)
                        faces = self.face_engine.detect_raw(flipped)
                        if faces is not None and len(faces) > 0:
                            for f in faces:
                                fx, fy, fw, fh = [int(v) for v in f[:4]]
                                cv2.rectangle(disp, (fx, fy), (fx + fw, fy + fh), (0, 255, 100), 2)
                    cv2.putText(disp, f"ZAPYS ZRYSY: {idx + 1}/5", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 100), 2)
                    uri = frame_to_data_uri(disp, quality=80)
                    if uri and self.face_reg_preview_img:
                        self.face_reg_preview_img.src = uri
                        self.face_reg_preview_img.visible = True
                        self._safe(self.face_reg_preview_img)
                time.sleep(0.16)

            cap.release()
            self.face_reg_ring.visible = False
            self._safe(self.face_reg_ring)
            if not frames:
                self.face_status_text.value = "Помилка читання кадрів з камери"
                self._safe(self.face_status_text)
                return
            try:
                prof = self.face_engine.register_face_multi(name, frames)
                cnt = prof.get("samples_count", 1)
                self.face_status_text.value = f"✅ Успішно збережено: {prof['name']} (ID: {prof['id']}, зрізів: {cnt})"
                self.face_reg_name = ""
                if self.face_name_field:
                    self.face_name_field.value = ""
                    self._safe(self.face_name_field)
                self._toast(f"Особу «{name}» додано до Face DB ({cnt} зрізів)", "ok")
                self._refresh_face_profiles()
            except Exception as ex:
                self.face_status_text.value = f"Помилка: {ex}"
                self._toast(str(ex), "err")
            self._safe(self.face_status_text)
            if was_scanning:
                self._toggle_face_scan()

        threading.Thread(target=run, daemon=True).start()

    def _toggle_face_scan(self, e=None):
        if self.face_embedded_running:
            self.face_embedded_running = False
            self._style_face_scan(False)
            self._toast("Вбудований сканер Face ID зупинено")
            return

        if self._is_running():
            self._stop_pipeline(quiet=True)
            time.sleep(0.4)
        if self._is_testing():
            self._stop_test(quiet=True)
            time.sleep(0.4)

        self.face_embedded_running = True
        if self.face_engine:
            self.face_engine.reset_liveness()
        self._style_face_scan(True)
        self._toast("Вбудований сканер Face ID запущено прямо у вікні Flet!", "ok")
        threading.Thread(target=self._face_embedded_loop, daemon=True).start()

    def _face_embedded_loop(self):
        cam_id = read_cfg().get("settings", {}).get("camera_id", 0)
        cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(cam_id)
        if not cap.isOpened():
            self.face_embedded_running = False
            self._style_face_scan(False)
            self._toast(f"Не вдалося відкрити камеру {cam_id}", "err")
            return

        frame_idx = 0
        try:
            while self.alive and self.face_embedded_running and self.tab == 1:
                ret, frame = cap.read()
                if not ret or frame is None:
                    time.sleep(0.03)
                    continue
                frame = cv2.flip(frame, 1)
                frame_idx += 1

                if self.face_engine is not None:
                    # 1. Оцінка освітлення та дистанції (кожні 3 кадри)
                    if frame_idx % 3 == 0:
                        q = self.face_engine.assess_frame_quality(frame)
                        self._update_face_quality_ui(q)

                    # 2. Розпізнавання + Liveness
                    results = self.face_engine.recognize(frame, check_liveness=self.face_liveness_required)
                    frame = self.face_engine.draw_results(frame, results)

                    if results:
                        best = results[0]
                        liv_prog = float(best.get("liveness_progress", 0.0))
                        liv_ok = bool(best.get("liveness_ok", False))
                        liv_hint = best.get("liveness_hint", "")

                        if self.face_liveness_bar is not None:
                            self.face_liveness_bar.value = liv_prog
                            self.face_liveness_bar.color = "#22c55e" if liv_ok else "#f59e0b"
                            self._safe(self.face_liveness_bar)
                        if self.face_liveness_text is not None:
                            self.face_liveness_text.value = f"Статус живості (Liveness): {liv_hint} ({int(liv_prog * 100)}%)"
                            self.face_liveness_text.color = "#22c55e" if liv_ok else "#f59e0b"
                            self._safe(self.face_liveness_text)

                        if best["is_known"] and best["is_authorized"]:
                            if self.face_live_title:
                                self.face_live_title.value = f"{best['name']} ({best['confidence_pct']}%)"
                                self.face_live_title.color = "#22c55e"
                            if self.face_live_sub:
                                self.face_live_sub.value = "✅ ОСОБУ ТА ЖИВІСТЬ (3D LIVENESS) ПІДТВЕРДЖЕНО"
                            if self.face_live_badge:
                                self.face_live_badge.border = ft.Border.all(1.5, "#22c55e")
                                self.face_live_badge.bgcolor = op(0.15, "#22c55e")
                        elif best["is_known"] and not best["is_authorized"]:
                            if self.face_live_title:
                                self.face_live_title.value = f"{best['name']} — Очікування Liveness"
                                self.face_live_title.color = "#38bdf8"
                            if self.face_live_sub:
                                self.face_live_sub.value = f"🔄 {liv_hint} (захист від фотографії з телефона)"
                            if self.face_live_badge:
                                self.face_live_badge.border = ft.Border.all(1.5, "#38bdf8")
                                self.face_live_badge.bgcolor = op(0.14, "#38bdf8")
                        else:
                            if self.face_live_title:
                                self.face_live_title.value = "Невідома особа"
                                self.face_live_title.color = "#f59e0b"
                            if self.face_live_sub:
                                self.face_live_sub.value = "⚠️ Обличчя відсутнє в Face DB"
                            if self.face_live_badge:
                                self.face_live_badge.border = ft.Border.all(1.5, "#f59e0b")
                                self.face_live_badge.bgcolor = op(0.12, "#f59e0b")
                        self._safe(self.face_live_badge)

                uri = frame_to_data_uri(frame, quality=80)
                if uri and self.face_preview_img is not None:
                    self.face_preview_img.src = uri
                    if not self.face_preview_img.visible:
                        self.face_preview_img.visible = True
                        if hasattr(self, "face_preview_placeholder") and self.face_preview_placeholder:
                            self.face_preview_placeholder.visible = False
                            self._safe(self.face_preview_placeholder)
                    self._safe(self.face_preview_img)

                time.sleep(0.035)
        finally:
            cap.release()
            self.face_embedded_running = False
            self._style_face_scan(False)

    def _style_face_scan(self, scanning: bool):
        if self.face_scan_btn is not None:
            self.face_scan_btn.content.controls[0].icon = ft.Icons.STOP if scanning else ft.Icons.VIDEOCAM
            self.face_scan_btn.content.controls[1].value = "Зупинити вбудовану камеру" if scanning else "Увімкнути вбудований сканер"
            self.face_scan_btn.gradient = ft.LinearGradient(
                begin=ft.Alignment.CENTER_LEFT, end=ft.Alignment.CENTER_RIGHT,
                colors=["#dc2626", "#b91c1c"] if scanning else [self.c1, self.c2])
            self._safe(self.face_scan_btn)
        if self.face_scan_ring is not None:
            self.face_scan_ring.visible = scanning
            self._safe(self.face_scan_ring)
        if not scanning:
            if self.face_preview_img is not None:
                self.face_preview_img.visible = False
                self._safe(self.face_preview_img)
            if hasattr(self, "face_preview_placeholder") and self.face_preview_placeholder is not None:
                self.face_preview_placeholder.visible = True
                self._safe(self.face_preview_placeholder)
            if self.face_live_title:
                self.face_live_title.value = "Сканер не активний"
                self.face_live_title.color = ft.Colors.ON_SURFACE
            if self.face_live_sub:
                self.face_live_sub.value = "Натисніть кнопку для запуску вбудованого прев'ю наживо."
            if self.face_live_badge:
                self.face_live_badge.border = self._glass_border(False)
                self.face_live_badge.bgcolor = op(0.60 if self._is_dark() else 0.85, ft.Colors.SURFACE_CONTAINER_LOW)
                self._safe(self.face_live_badge)

    # ==================================================================
    # МОДАЛЬНІСТЬ 3 — ГОЛОС (AUDIO & VOICE)
    # ==================================================================
    def _view_voice(self) -> ft.Column:
        running = self.audio_mgr is not None and self.audio_mgr.running

        guide = self._guide_banner([
            ("1", "Зареєструйте голос", "Введіть ім'я внизу ліворуч та запишіть 3.5 с мови (наприклад: «Система, активуй робота»)."),
            ("2", "Запустіть перевірку", "Натисніть «Запустити перевірку» вгорі — мікрофон запише звук лише на обраний час."),
            ("3", "Подвійний результат", "Ви одразу побачите: 1) Тип звуку (Google YAMNet) та 2) Кому належить голос (Voice ID)."),
        ])

        # ==============================================================
        # ГОЛОВНИЙ БЛОК: АКТИВАЦІЯ ПЕРЕВІРКИ ГОЛОСУ (ON-DEMAND SPOT CHECK)
        # ==============================================================
        self.voice_spot_btn = self._gradient_button(
            f"Запустити перевірку ({self.voice_spot_duration} с)",
            ft.Icons.RECORD_VOICE_OVER,
            self._on_spot_check_click,
            height=46,
            colors=[self.c1, self.c2])

        self.voice_spot_ring = ft.ProgressRing(width=22, height=22, stroke_width=3, visible=False)
        self.voice_spot_status_text = ft.Text(
            "Готовий до перевірки. Натисніть кнопку для миттєвого аналізу типу звуку та диктора.",
            size=13, color=ft.Colors.ON_SURFACE_VARIANT)

        self.voice_spot_progress_bar = ft.ProgressBar(
            value=0.0, color=self.c1, bgcolor=op(0.12, ft.Colors.ON_SURFACE), height=6, border_radius=3)
        self.voice_spot_vu_bar = ft.ProgressBar(
            value=0.0, color=self.c2, bgcolor=op(0.12, ft.Colors.ON_SURFACE), height=6, border_radius=3)

        # Жива 16-смугова осцилограма (Спектральний еквалайзер FFT)
        self.voice_waveform_bars = []
        bar_palette = [self.c1, self.c2, "#38bdf8", "#22c55e"]
        for b_idx in range(16):
            b_col = bar_palette[b_idx % len(bar_palette)]
            bar = ft.Container(
                width=14, height=6, border_radius=4,
                bgcolor=b_col,
                animate=ft.Animation(90, ft.AnimationCurve.EASE_OUT))
            self.voice_waveform_bars.append(bar)

        self.voice_waveform_row = ft.Row(
            alignment=ft.MainAxisAlignment.SPACE_EVENLY,
            vertical_alignment=ft.CrossAxisAlignment.END,
            height=54,
            controls=self.voice_waveform_bars)

        waveform_box = self._glass(
            hover=False, blur=False, radius=8, padding=ft.Padding.symmetric(horizontal=14, vertical=8),
            content=ft.Column(spacing=4, controls=[
                ft.Row(alignment=ft.MainAxisAlignment.SPACE_BETWEEN, controls=[
                    ft.Row(spacing=6, controls=[
                        ft.Icon(ft.Icons.GRAPHIC_EQ, size=15, color=self.c1),
                        ft.Text("Жива осцилограма (16-смуговий FFT спектр 80–7500 Гц):", size=11, weight=ft.FontWeight.W_600)]),
                    ft.Text("Реагує під час тесту та запису голосу", size=10, color=ft.Colors.ON_SURFACE_VARIANT)]),
                self.voice_waveform_row]))

        dur_seg = ft.SegmentedButton(
            selected=[str(self.voice_spot_duration)],
            allow_empty_selection=False,
            segments=[
                ft.Segment(value="2", label=ft.Text("2 с")),
                ft.Segment(value="3", label=ft.Text("3 с (реком.)")),
                ft.Segment(value="5", label=ft.Text("5 с")),
            ],
            on_change=self._on_spot_duration_change)

        self.voice_spot_result_card = ft.Container()
        self._refresh_spot_result_ui()

        spot_check_block = self._glass(
            hover=False, radius=12, padding=18,
            content=ft.Column(spacing=14, controls=[
                ft.Row(alignment=ft.MainAxisAlignment.SPACE_BETWEEN, wrap=True, controls=[
                    self._section_title(
                        "Активація перевірки звуку та голосу (на вимогу)",
                        ft.Icons.RECORD_VOICE_OVER,
                        "Натисніть кнопку, скажіть фразу або видайте звук — система класифікує тип звуку (YAMNet) та власника голосу (Voice ID)"),
                    ft.OutlinedButton("Експорт модуля Голосу", icon=ft.Icons.FOLDER_ZIP,
                                      on_click=lambda e: self._open_export_modal(default_modality="voice"))]),
                ft.Row(
                    wrap=True, spacing=14, vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    controls=[
                        self.voice_spot_btn,
                        dur_seg,
                        ft.Row(spacing=8, controls=[self.voice_spot_ring, self.voice_spot_status_text])]),
                waveform_box,
                ft.Column(spacing=6, controls=[
                    ft.Row(alignment=ft.MainAxisAlignment.SPACE_BETWEEN, controls=[
                        ft.Text("Прогрес запису:", size=11, color=ft.Colors.ON_SURFACE_VARIANT),
                        ft.Text("Рівень мікрофона під час тесту (VU):", size=11, color=ft.Colors.ON_SURFACE_VARIANT)]),
                    ft.ResponsiveRow(spacing=12, controls=[
                        ft.Column(col={"xs": 12, "md": 6}, spacing=2, controls=[self.voice_spot_progress_bar]),
                        ft.Column(col={"xs": 12, "md": 6}, spacing=2, controls=[self.voice_spot_vu_bar])])]),
                ft.Divider(height=1, color=op(0.10, ft.Colors.ON_SURFACE)),
                self.voice_spot_result_card]))

        # ==============================================================
        # НИЖНІЙ БЛОК: РЕЄСТРАЦІЯ ДИКТОРІВ + НАЛАШТУВАННЯ ПОРОГУ + ФОНОВИЙ ПОТІК
        # ==============================================================
        self.voice_name_field = ft.TextField(
            label="Ім'я диктора для реєстрації",
            hint_text="Наприклад: Max (говоріть спокійним голосом 3.5 с)...",
            prefix_icon=ft.Icons.BADGE,
            border=ft.OutlineInputBorder(border_radius=8),
            value=self.voice_reg_name, on_change=lambda e: setattr(self, "voice_reg_name", e.control.value))

        self.voice_rec_ring = ft.ProgressRing(width=22, height=22, stroke_width=3, visible=False)
        self.voice_rec_btn = self._gradient_button("Записати зразок голосу (3.5 с)", ft.Icons.VOICE_CHAT, self._on_record_speaker_click)

        curr_v_thresh = self.audio_mgr.voice_id.threshold if self.audio_mgr else 0.70
        v_thresh_lbl = ft.Text(f"Поріг схожості голосу (Voice ID): {int(curr_v_thresh * 100)}%", size=12, weight=ft.FontWeight.W_600)

        def on_voice_thresh(e):
            val = round(float(e.control.value), 2)
            if self.audio_mgr:
                self.audio_mgr.voice_id.threshold = val
            v_thresh_lbl.value = f"Поріг схожості голосу (Voice ID): {int(val * 100)}%"
            self._safe(v_thresh_lbl)

        v_thresh_slider = ft.Slider(
            min=0.50, max=0.90, divisions=40, value=curr_v_thresh,
            active_color=self.c2, on_change=on_voice_thresh,
            tooltip="Вище значення = суворіша перевірка тембру голосу")

        self.voice_speaker_title = ft.Text("Очікування мови...", size=16, weight=ft.FontWeight.W_700)
        self.voice_speaker_sub = ft.Text("Скажіть щось у мікрофон для верифікації диктора", size=12, color=ft.Colors.ON_SURFACE_VARIANT)
        self.voice_speaker_badge = self._glass(
            hover=False, radius=8, padding=12,
            content=ft.Row(spacing=12, controls=[
                self._icon_badge(ft.Icons.RECORD_VOICE_OVER, self.c2, 36, 18),
                ft.Column(spacing=2, expand=True, controls=[self.voice_speaker_title, self.voice_speaker_sub])]))

        self.voice_profiles_list = ft.Column(spacing=8, horizontal_alignment=ft.CrossAxisAlignment.STRETCH)
        self._refresh_voice_profiles(update=False)

        block_speakers = self._glass(
            hover=False, radius=10, col={"xs": 12, "lg": 6},
            content=ft.Column(spacing=14, controls=[
                self._section_title("База дикторів (Voice ID)", ft.Icons.PEOPLE_OUTLINE, "Реєстрація власників голосу за допомогою тембрального зліпка 1024-d"),
                self.voice_name_field,
                ft.Row(spacing=12, controls=[self.voice_rec_btn, self.voice_rec_ring]),
                ft.Column(spacing=2, controls=[
                    v_thresh_lbl,
                    v_thresh_slider]),
                ft.Text("Зареєстровані диктори в системі:", size=12, weight=ft.FontWeight.W_600),
                self.voice_profiles_list,
                self._hint("Порада: Під час запису промовте ціле речення (наприклад: «Привіт, це мій голосовий ключ доступу до робота»). Уникайте пауз і шуму.")]))

        self.audio_toggle_btn = self._gradient_button(
            "Вимкнути мікрофон" if running else "Увімкнути постійний моніторинг",
            ft.Icons.MIC_OFF if running else ft.Icons.MIC,
            self._toggle_audio_stream,
            colors=["#dc2626", "#b91c1c"] if running else [self.c1, self.c2])

        self.audio_vu_bar = ft.ProgressBar(value=0.0, color=self.c1, bgcolor=op(0.12, ft.Colors.ON_SURFACE), height=8, border_radius=4)
        self.audio_db_text = ft.Text("Рівень сигналу: 0%", size=12, color=ft.Colors.ON_SURFACE_VARIANT)
        self.audio_events_col = ft.Column(spacing=8, horizontal_alignment=ft.CrossAxisAlignment.STRETCH)
        self._refresh_audio_events_placeholder()

        block_stream = self._glass(
            hover=False, radius=10, col={"xs": 12, "lg": 6},
            content=ft.Column(spacing=14, controls=[
                self._section_title("Постійний моніторинг (опціонально)", ft.Icons.STREAM, "Якщо потрібен не разовий клік, а неперервний потік у фоні"),
                ft.Row(spacing=12, controls=[self.audio_toggle_btn]),
                ft.Column(spacing=4, controls=[self.audio_db_text, self.audio_vu_bar]),
                ft.Divider(height=1, color=op(0.10, ft.Colors.ON_SURFACE)),
                ft.Text("Поточний стан live-потоку:", size=12, weight=ft.FontWeight.W_600),
                self.audio_events_col,
                self.voice_speaker_badge,
                self._hint("За замовчуванням вимкнений, щоб не витрачати ресурси ПК. Для швидких перевірок використовуйте кнопку вгорі.")]))

        return self._view([
            guide,
            spot_check_block,
            ft.ResponsiveRow(spacing=16, run_spacing=16, controls=[block_speakers, block_stream])
        ])

    def _update_waveform_bars(self, bands: List[float]):
        if not self.voice_waveform_bars or self.voice_waveform_row is None:
            return
        for idx, bar in enumerate(self.voice_waveform_bars):
            val = float(bands[idx]) if idx < len(bands) else 0.0
            bar.height = max(5, min(52, int(5 + val * 47)))
        self._safe(self.voice_waveform_row)

    def _on_spot_duration_change(self, e):
        sel = list(e.control.selected)
        if sel:
            self.voice_spot_duration = int(sel[0])
            if self.voice_spot_btn and len(self.voice_spot_btn.content.controls) > 1:
                self.voice_spot_btn.content.controls[1].value = f"Запустити перевірку ({self.voice_spot_duration} с)"
                self._safe(self.voice_spot_btn)

    def _on_spot_check_click(self, e):
        if self.voice_is_checking:
            return
        self.voice_is_checking = True
        duration = float(self.voice_spot_duration)

        if self.voice_spot_btn:
            self.voice_spot_btn.opacity = 0.5
            self._safe(self.voice_spot_btn)
        if self.voice_spot_ring:
            self.voice_spot_ring.visible = True
            self._safe(self.voice_spot_ring)
        if self.voice_spot_status_text:
            self.voice_spot_status_text.value = f"🎙️ Слухаю... Говоріть у мікрофон або видайте звук ({duration:.0f} с)"
            self.voice_spot_status_text.color = self.c1
            self._safe(self.voice_spot_status_text)
        if self.voice_spot_progress_bar:
            self.voice_spot_progress_bar.value = 0.0
            self._safe(self.voice_spot_progress_bar)
        if self.voice_spot_vu_bar:
            self.voice_spot_vu_bar.value = 0.0
            self._safe(self.voice_spot_vu_bar)

        def progress_cb(prog: float, rms: float):
            if self.voice_spot_progress_bar:
                self.voice_spot_progress_bar.value = prog
                self._safe(self.voice_spot_progress_bar)
            if self.voice_spot_vu_bar:
                self.voice_spot_vu_bar.value = rms
                self._safe(self.voice_spot_vu_bar)

        def spectrum_cb(bands: List[float]):
            self._update_waveform_bars(bands)

        def run():
            try:
                if self.audio_mgr is None:
                    self.audio_mgr = AudioStreamManager()

                result = self.audio_mgr.run_spot_check(
                    duration_sec=duration,
                    progress_cb=progress_cb,
                    spectrum_cb=spectrum_cb)
                self.last_spot_result = result
                if self.voice_spot_status_text:
                    self.voice_spot_status_text.value = "✅ Перевірку завершено! Результати відображено нижче."
                    self.voice_spot_status_text.color = "#22c55e"
                    self._safe(self.voice_spot_status_text)
                self._refresh_spot_result_ui()
                self._toast("Аналіз завершено: тип звуку та диктора визначено!", "ok")
            except Exception as ex:
                if self.voice_spot_status_text:
                    self.voice_spot_status_text.value = f"Помилка: {ex}"
                    self.voice_spot_status_text.color = "#ef4444"
                    self._safe(self.voice_spot_status_text)
                self._toast(f"Помилка перевірки аудіо: {ex}", "err")
            finally:
                self.voice_is_checking = False
                self._update_waveform_bars([0.05] * 16)
                if self.voice_spot_btn:
                    self.voice_spot_btn.opacity = 1.0
                    self._safe(self.voice_spot_btn)
                if self.voice_spot_ring:
                    self.voice_spot_ring.visible = False
                    self._safe(self.voice_spot_ring)
                if self.voice_spot_progress_bar:
                    self.voice_spot_progress_bar.value = 1.0
                    self._safe(self.voice_spot_progress_bar)

        threading.Thread(target=run, daemon=True).start()

    def _refresh_spot_result_ui(self):
        if self.voice_spot_result_card is None:
            return

        if self.last_spot_result is None:
            self.voice_spot_result_card.content = self._glass(
                blur=False, radius=10, padding=14,
                content=ft.Row(spacing=12, vertical_alignment=ft.CrossAxisAlignment.CENTER, controls=[
                    ft.Icon(ft.Icons.INFO_OUTLINE, size=24, color=self.c1),
                    ft.Column(spacing=2, expand=True, controls=[
                        ft.Text("Результати аналізу з'являться після натискання кнопки перевірки", size=13, weight=ft.FontWeight.W_600),
                        ft.Text("Система одночасно визначить тип звуку (мова, свист, плескання тощо) та ідентифікує, кому належить голос.", size=11, color=ft.Colors.ON_SURFACE_VARIANT)])]))
        else:
            res = self.last_spot_result
            top_events = res.get("top_events", [])
            spk = res.get("speaker", {})
            primary = res.get("primary_event")

            # 1. Панель типу звуку (Google YAMNet)
            event_items: List[ft.Control] = []
            palette = [self.c1, self.c2, "#38bdf8", "#f59e0b"]
            if top_events:
                for idx, ev in enumerate(top_events[:3]):
                    pct = ev.get("confidence_pct", 0)
                    col = palette[idx % len(palette)]
                    event_items.append(
                        ft.Row(alignment=ft.MainAxisAlignment.SPACE_BETWEEN, controls=[
                            ft.Row(spacing=6, controls=[
                                ft.Icon(ft.Icons.VOLUME_UP, size=15, color=col),
                                ft.Text(ev.get("label", ev.get("name")), size=12, weight=ft.FontWeight.W_600)]),
                            ft.Row(spacing=6, controls=[
                                ft.ProgressBar(value=pct / 100.0, width=90, height=5, color=col, border_radius=3),
                                ft.Text(f"{pct}%", size=11, weight=ft.FontWeight.W_700, font_family="Consolas", color=col)])]))
            else:
                event_items.append(
                    ft.Text("Чітких звукових подій не розпізнано (тиша або слабкий сигнал)", size=12, color=ft.Colors.ON_SURFACE_VARIANT))

            p_title = primary.get("label", "Тиша / Не визначено") if primary else "Тиша / Не визначено"
            p_pct = primary.get("confidence_pct", 0) if primary else 0

            sound_panel = self._glass(
                blur=False, radius=10, padding=14, col={"xs": 12, "lg": 6},
                content=ft.Column(spacing=10, controls=[
                    ft.Row(spacing=10, controls=[
                        self._icon_badge(ft.Icons.WAVES, self.c1, 38, 20),
                        ft.Column(spacing=1, expand=True, controls=[
                            ft.Text("1. ТИП ЗВУКУ (Google YAMNet)", size=11, weight=ft.FontWeight.W_700, color=self.c1),
                            ft.Text(f"{p_title} ({p_pct}%)", size=15, weight=ft.FontWeight.W_800)])]),
                    ft.Divider(height=1, color=op(0.10, ft.Colors.ON_SURFACE)),
                    ft.Text("Топ розпізнаних акустичних подій у фрагменті:", size=11, color=ft.Colors.ON_SURFACE_VARIANT),
                    ft.Column(spacing=6, controls=event_items)]))

            # 2. Панель власника голосу (Voice ID)
            speech_det = spk.get("speech_detected", False)
            is_verified = spk.get("is_verified", False)
            spk_name = spk.get("name", "Невідомий")
            spk_pct = spk.get("confidence_pct", 0)

            if not speech_det:
                v_color = "#94a3b8"
                v_icon = ft.Icons.MIC_OFF
                v_title = "Мови не виявлено"
                v_status = "Акустичний фрагмент не містить людського голосу (звуки оточення або тиша)"
                v_bg = op(0.08, ft.Colors.ON_SURFACE)
                v_border = ft.Border.all(1, op(0.15, ft.Colors.ON_SURFACE))
            elif is_verified:
                v_color = "#22c55e"
                v_icon = ft.Icons.VERIFIED_USER
                v_title = f"Диктор: {spk_name}"
                v_status = f"✅ ГОЛОС АВТОРИЗОВАНО (Збіг біометрії: {spk_pct}%)"
                v_bg = op(0.15, "#22c55e")
                v_border = ft.Border.all(1.5, "#22c55e")
            else:
                v_color = "#f59e0b"
                v_icon = ft.Icons.GPP_MAYBE
                v_title = "Невідомий голос"
                v_status = f"⚠️ ГОЛОС НЕ ЗБІГАЄТЬСЯ з зареєстрованими дикторами ({spk_pct}% схожості)"
                v_bg = op(0.15, "#f59e0b")
                v_border = ft.Border.all(1.5, "#f59e0b")

            voice_panel = self._glass(
                blur=False, radius=10, padding=14, col={"xs": 12, "lg": 6},
                content=ft.Column(spacing=10, controls=[
                    ft.Row(spacing=10, controls=[
                        self._icon_badge(v_icon, v_color, 38, 20),
                        ft.Column(spacing=1, expand=True, controls=[
                            ft.Text("2. КОМУ НАЛЕЖИТЬ ГОЛОС (Voice ID)", size=11, weight=ft.FontWeight.W_700, color=v_color),
                            ft.Text(v_title, size=15, weight=ft.FontWeight.W_800, color=v_color)])]),
                    ft.Divider(height=1, color=op(0.10, ft.Colors.ON_SURFACE)),
                    ft.Container(
                        padding=ft.Padding.symmetric(horizontal=12, vertical=10),
                        border_radius=8, bgcolor=v_bg, border=v_border,
                        content=ft.Text(v_status, size=12, color=v_color, weight=ft.FontWeight.W_600))]))

            self.voice_spot_result_card.content = ft.ResponsiveRow(spacing=12, run_spacing=12, controls=[sound_panel, voice_panel])

        self._safe(self.voice_spot_result_card)

    def _refresh_audio_events_placeholder(self):
        if self.audio_events_col is None:
            return
        running = self.audio_mgr is not None and self.audio_mgr.running
        self.audio_events_col.controls = [
            self._glass(blur=False, radius=8, padding=12, content=ft.Row(spacing=10, controls=[
                ft.Icon(ft.Icons.GRAPHIC_EQ, size=18, color=self.c1),
                ft.Text("Увімкніть мікрофон для детекції звуків оточення YAMNet" if not running else "Очікування звуків оточення...",
                        size=12, color=ft.Colors.ON_SURFACE_VARIANT)]))
        ]

    def _update_audio_events(self, events: List[Dict[str, Any]]):
        if self.audio_events_col is None or not events:
            return
        cards: List[ft.Control] = []
        palette = [self.c1, self.c2, "#38bdf8", "#f59e0b"]
        for idx, ev in enumerate(events[:3]):
            pct = ev.get("confidence_pct", 0)
            color = palette[idx % len(palette)]
            row = self._glass(
                blur=False, radius=8, padding=ft.Padding.symmetric(horizontal=12, vertical=8),
                content=ft.Row(alignment=ft.MainAxisAlignment.SPACE_BETWEEN, controls=[
                    ft.Row(spacing=8, controls=[
                        ft.Icon(ft.Icons.VOLUME_UP, size=16, color=color),
                        ft.Text(ev.get("label", ev.get("name", "")), size=13, weight=ft.FontWeight.W_600)]),
                    ft.Row(spacing=8, controls=[
                        ft.ProgressBar(value=pct / 100.0, width=120, height=6, color=color, border_radius=3),
                        ft.Text(f"{pct}%", size=12, weight=ft.FontWeight.W_700, font_family="Consolas", color=color)])]))
            cards.append(row)
        self.audio_events_col.controls = cards
        self._safe(self.audio_events_col)

    def _update_speaker_badge(self, spk: Dict[str, Any]):
        if self.voice_speaker_badge is None or self.voice_speaker_title is None or self.voice_speaker_sub is None:
            return
        if not spk.get("speech_detected", False):
            self.voice_speaker_title.value = "Тиша / Немає мови"
            self.voice_speaker_title.color = ft.Colors.ON_SURFACE_VARIANT
            self.voice_speaker_sub.value = "Скажіть щось у мікрофон для верифікації диктора"
            self.voice_speaker_badge.border = self._glass_border(False)
            self.voice_speaker_badge.bgcolor = op(0.60 if self._is_dark() else 0.85, ft.Colors.SURFACE_CONTAINER_LOW)
        elif spk.get("is_verified", False):
            self.voice_speaker_title.value = f"Диктор: {spk.get('name')} ({spk.get('confidence_pct')}%)"
            self.voice_speaker_title.color = "#22c55e"
            self.voice_speaker_sub.value = "✅ ГОЛОС ПІДТВЕРДЖЕНО (Власник)"
            self.voice_speaker_badge.border = ft.Border.all(1.5, "#22c55e")
            self.voice_speaker_badge.bgcolor = op(0.15, "#22c55e")
        else:
            self.voice_speaker_title.value = "Невідомий голос"
            self.voice_speaker_title.color = "#f59e0b"
            self.voice_speaker_sub.value = "⚠️ Голос не відповідає зареєстрованим дикторам"
            self.voice_speaker_badge.border = ft.Border.all(1.5, "#f59e0b")
            self.voice_speaker_badge.bgcolor = op(0.12, "#f59e0b")
        self._safe(self.voice_speaker_badge)

    def _refresh_voice_profiles(self, update: bool = True):
        if self.audio_mgr is None or self.voice_profiles_list is None:
            return
        profiles = self.audio_mgr.voice_id.load_profiles()
        items: List[ft.Control] = []
        if not profiles:
            items.append(ft.Container(padding=16, alignment=ft.Alignment.CENTER, content=ft.Column(
                horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=6, controls=[
                    ft.Icon(ft.Icons.MIC_OFF, size=36, color=ft.Colors.ON_SURFACE_VARIANT),
                    ft.Text("Дикторів ще не зареєстровано", size=13, weight=ft.FontWeight.W_600),
                    ft.Text("Введіть ім'я та запишіть 3.5 с мови.", size=11, color=ft.Colors.ON_SURFACE_VARIANT)])))
        for prof in profiles:
            p_id = prof.get("id", "")
            p_name = prof.get("name", "Без імені")
            p_date = prof.get("created_at", "")

            row = self._glass(
                blur=False, radius=8, padding=ft.Padding.symmetric(horizontal=12, vertical=8),
                content=ft.Row(spacing=12, vertical_alignment=ft.CrossAxisAlignment.CENTER, controls=[
                    self._icon_badge(ft.Icons.RECORD_VOICE_OVER, self.c2, 38, 20),
                    ft.Column(expand=True, spacing=2, tight=True, controls=[
                        ft.Text(p_name, size=13, weight=ft.FontWeight.W_700),
                        ft.Text(f"ID: {p_id}  •  {p_date}", size=11, font_family="Consolas", color=ft.Colors.ON_SURFACE_VARIANT)]),
                    ft.IconButton(ft.Icons.DELETE_OUTLINE, icon_color="#f87171", tooltip="Видалити диктора",
                                  on_click=lambda e, pid=p_id, n=p_name: self._confirm(
                                      "Видалити диктора", f"Видалити профіль «{n}» з Voice ID?", "Видалити",
                                      lambda: self._delete_voice_profile(pid)))]))
            items.append(row)
        self.voice_profiles_list.controls = items
        if update:
            self._safe(self.voice_profiles_list)

    def _delete_voice_profile(self, profile_id: str):
        if self.audio_mgr and self.audio_mgr.voice_id.delete_profile(profile_id):
            self._toast("Профіль диктора успішно видалено", "ok")
            self._refresh_voice_profiles()

    def _toggle_audio_stream(self, e=None):
        if self.audio_mgr is None:
            self.audio_mgr = AudioStreamManager()
        if self.audio_mgr.running:
            self.audio_mgr.stop()
            self._toast("Моніторинг мікрофона вимкнено")
            self._style_audio_toggle(False)
            self._refresh_audio_events_placeholder()
        else:
            self.audio_mgr.start()
            self._toast("Моніторинг мікрофона активовано (YAMNet + Voice ID)", "ok")
            self._style_audio_toggle(True)

    def _style_audio_toggle(self, running: bool):
        if self.audio_toggle_btn is not None:
            self.audio_toggle_btn.content.controls[0].icon = ft.Icons.MIC_OFF if running else ft.Icons.MIC
            self.audio_toggle_btn.content.controls[1].value = "Вимкнути мікрофон" if running else "Увімкнути моніторинг мікрофона"
            self.audio_toggle_btn.gradient = ft.LinearGradient(
                begin=ft.Alignment.CENTER_LEFT, end=ft.Alignment.CENTER_RIGHT,
                colors=["#dc2626", "#b91c1c"] if running else [self.c1, self.c2])
            self._safe(self.audio_toggle_btn)

    def _on_record_speaker_click(self, e):
        name = (self.voice_reg_name or "").strip()
        if not name:
            self._toast("Введіть ім'я диктора для реєстрації", "err")
            return
        if self.audio_mgr is None:
            self.audio_mgr = AudioStreamManager()

        self.voice_rec_btn.opacity = 0.5
        self.voice_rec_ring.visible = True
        self._safe(self.voice_rec_btn)
        self._safe(self.voice_rec_ring)
        self._toast(f"Говоріть у мікрофон 3.5 сек... (реєстрація «{name}»)", "info")

        def run():
            samples = self.audio_mgr.record_speaker_sample(
                duration_sec=3.5,
                spectrum_cb=self._update_waveform_bars)
            self._update_waveform_bars([0.05] * 16)
            try:
                prof = self.audio_mgr.voice_id.register_speaker(name, samples)
                self.voice_reg_name = ""
                if self.voice_name_field:
                    self.voice_name_field.value = ""
                    self._safe(self.voice_name_field)
                self._toast(f"Диктора «{name}» успішно зареєстровано!", "ok")
                self._refresh_voice_profiles()
            except Exception as ex:
                self._toast(str(ex), "err")
            self.voice_rec_btn.opacity = 1
            self.voice_rec_ring.visible = False
            self._safe(self.voice_rec_btn)
            self._safe(self.voice_rec_ring)

        threading.Thread(target=run, daemon=True).start()

    # ==================================================================
    # МОДАЛЬНІСТЬ 4 — ТЕСТОВИЙ ПОЛІГОН 2FA РОБОТА (СИМУЛЯТОР У ДОДАТКУ)
    # ==================================================================
    def _view_robot_sim(self) -> ft.Column:
        guide = self._guide_banner([
            ("1", "Крок 1: Face ID + Liveness", "Увімкніть камеру полігону — система перевірить обличчя власника та захист від фото (3D поворот голови)."),
            ("2", "Крок 2: Голосовий ключ (Voice ID)", "Натисніть «Сканувати голосовий ключ (3 с)» та скажіть фразу власника — відкриється зелений замок ARMED."),
            ("3", "Крок 3: Керування віртуальним роботом", "Показуйте жести з вашої Бібліотеки в камеру або тестуйте маневри на 2D-арені перед експортом на залізо!"),
        ])

        running = self.sim_running
        self.sim_cam_btn = self._gradient_button(
            "Зупинити камеру полігону" if running else "1. Запустити камеру полігону",
            ft.Icons.STOP if running else ft.Icons.VIDEOCAM,
            self._toggle_sim_camera,
            colors=["#dc2626", "#b91c1c"] if running else [self.c1, self.c2])

        self.sim_voice_btn = self._gradient_button(
            "2. Сканувати голосовий ключ (3 с)",
            ft.Icons.RECORD_VOICE_OVER,
            self._on_sim_voice_verify,
            colors=["#0284c7", "#2563eb"])

        lock_btn = ft.OutlinedButton(
            "Заблокувати (LOCK)", icon=ft.Icons.LOCK_RESET,
            on_click=lambda e: self._sim_reset_lock())

        bypass_btn = ft.TextButton(
            "Тестовий обхід 2FA (Demo ARMED)", icon=ft.Icons.BOLT,
            tooltip="Миттєво розблокувати полігон для швидкого тестування жестів без очікування",
            on_click=lambda e: self._sim_force_arm())

        # Вбудоване прев'ю камери полігону
        self.sim_preview_img = ft.Image(
            src=BLANK_FRAME_JPEG, width=520, height=320, fit="contain",
            gapless_playback=True, border_radius=10, visible=running)
        self.sim_preview_placeholder = ft.Container(
            height=320, border_radius=10, alignment=ft.Alignment.CENTER, visible=not running,
            bgcolor=op(0.18, ft.Colors.SURFACE_CONTAINER_HIGHEST),
            border=ft.Border.all(1, op(0.14, ft.Colors.ON_SURFACE)),
            content=ft.Column(horizontal_alignment=ft.CrossAxisAlignment.CENTER, alignment=ft.MainAxisAlignment.CENTER, spacing=8, controls=[
                ft.Icon(ft.Icons.SMART_TOY_OUTLINED, size=48, color=ft.Colors.ON_SURFACE_VARIANT),
                ft.Text("Камера тестового полігону вимкнена", size=14, weight=ft.FontWeight.W_700),
                ft.Text("Натисніть «1. Запустити камеру полігону», щоб протестувати 2FA розблокування та жестове керування роботом",
                        size=11, color=ft.Colors.ON_SURFACE_VARIANT, text_align=ft.TextAlign.CENTER)]))

        # 3 картки ланцюжка безпеки 2FA
        self.sim_step1_card = ft.Container()
        self.sim_step2_card = ft.Container()
        self.sim_step3_card = ft.Container()
        self.sim_status_banner = ft.Container()

        # 2D Віртуальна арена робота (420x250)
        arena_w, arena_h = 420, 240
        bx = int(self.sim_robot_x * (arena_w - 56))
        by = int(self.sim_robot_y * (arena_h - 56))

        self.sim_arena_bot = ft.Container(
            width=52, height=52, border_radius=26,
            left=bx, top=by,
            alignment=ft.Alignment.CENTER,
            bgcolor=self.c1,
            border=ft.Border.all(2.5, "white"),
            shadow=ft.BoxShadow(blur_radius=16, color=op(0.6, self.c1), offset=ft.Offset(0, 4)),
            animate_position=ft.Animation(240, ft.AnimationCurve.EASE_OUT_CUBIC),
            animate=ft.Animation(240, ft.AnimationCurve.EASE_OUT),
            content=ft.Icon(ft.Icons.SMART_TOY, color="white", size=28))

        arena_stack = ft.Container(
            width=arena_w, height=arena_h, border_radius=12,
            bgcolor="#0b1324" if self._is_dark() else "#e2e8f0",
            border=ft.Border.all(1.5, op(0.28, self.c1)),
            clip_behavior=ft.ClipBehavior.ANTI_ALIAS,
            content=ft.Stack(width=arena_w, height=arena_h, controls=[
                # Координатна сітка арени
                ft.Container(left=arena_w // 2, top=0, width=1, height=arena_h, bgcolor=op(0.10, ft.Colors.ON_SURFACE)),
                ft.Container(left=0, top=arena_h // 2, width=arena_w, height=1, bgcolor=op(0.10, ft.Colors.ON_SURFACE)),
                ft.Container(left=10, top=8, content=ft.Text("2D АРЕНА СИМУЛЯЦІЇ ШАСІ РОБОТА", size=10, font_family="Consolas", color=ft.Colors.ON_SURFACE_VARIANT)),
                self.sim_arena_bot]))

        self.sim_cmd_text = ft.Text(self.sim_robot_cmd, size=13, weight=ft.FontWeight.W_800, font_family="Consolas", color=self.c1)

        # Кнопки ручного тесту маневрів на арені
        dpad_row = ft.Row(alignment=ft.MainAxisAlignment.CENTER, spacing=8, wrap=True, controls=[
            ft.OutlinedButton("⬆️ Вперед", on_click=lambda e: self._sim_move_robot(0.0, -0.18, "MOVE_FORWARD (Вперед)")),
            ft.OutlinedButton("⬇️ Назад", on_click=lambda e: self._sim_move_robot(0.0, 0.18, "MOVE_BACKWARD (Назад)")),
            ft.OutlinedButton("⬅️ Вліво", on_click=lambda e: self._sim_move_robot(-0.18, 0.0, "TURN_LEFT (Поворот вліво)")),
            ft.OutlinedButton("➡️ Вправо", on_click=lambda e: self._sim_move_robot(0.18, 0.0, "TURN_RIGHT (Поворот вправо)")),
            ft.OutlinedButton("🛑 Стоп / Центр", on_click=lambda e: self._sim_center_robot()),
        ])

        self._refresh_sim_ui(update=False)

        left_col = self._glass(
            hover=False, radius=10, col={"xs": 12, "lg": 6},
            content=ft.Column(spacing=14, controls=[
                ft.Row(alignment=ft.MainAxisAlignment.SPACE_BETWEEN, wrap=True, controls=[
                    self._section_title("Вбудована камера 2FA + Жести", ft.Icons.VIDEOCAM, "Одночасна перевірка Face ID, Liveness та жестів керування"),
                    ft.OutlinedButton("Експорт 2FA-Робота (ZIP)", icon=ft.Icons.FOLDER_ZIP,
                                      on_click=lambda e: self._open_export_modal(default_modality="robot_2fa"))]),
                ft.Row(spacing=10, wrap=True, controls=[self.sim_cam_btn, self.sim_voice_btn, lock_btn]),
                ft.Container(alignment=ft.Alignment.CENTER, content=ft.Column(
                    horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                    controls=[self.sim_preview_placeholder, self.sim_preview_img])),
                self.sim_status_banner,
                ft.Row(alignment=ft.MainAxisAlignment.SPACE_BETWEEN, controls=[
                    ft.Text("Порада: Будь-який жест із вашої Бібліотеки автоматично рухає робота в режимі ARMED.", size=11, color=ft.Colors.ON_SURFACE_VARIANT, expand=True),
                    bypass_btn])]))

        right_col = self._glass(
            hover=False, radius=10, col={"xs": 12, "lg": 6},
            content=ft.Column(spacing=12, horizontal_alignment=ft.CrossAxisAlignment.CENTER, controls=[
                self._section_title("Ланцюжок безпеки 2FA та 2D-Арена Робота", ft.Icons.SECURITY, "Face ID OK + Voice ID OK = Зелений замок відкрито (ARMED)"),
                self.sim_step1_card,
                self.sim_step2_card,
                self.sim_step3_card,
                ft.Divider(height=1, color=op(0.10, ft.Colors.ON_SURFACE)),
                arena_stack,
                ft.Container(
                    padding=ft.Padding.symmetric(horizontal=14, vertical=8), border_radius=8,
                    bgcolor=op(0.12, self.c1), border=ft.Border.all(1, op(0.3, self.c1)),
                    content=ft.Row(spacing=8, alignment=ft.MainAxisAlignment.CENTER, controls=[
                        ft.Icon(ft.Icons.TERMINAL, size=16, color=self.c1),
                        self.sim_cmd_text])),
                dpad_row]))

        return self._view([guide, ft.ResponsiveRow(spacing=16, run_spacing=16, controls=[left_col, right_col])])

    def _refresh_sim_ui(self, update: bool = True):
        # Визначаємо загальний стан 2FA
        if self.sim_face_ok and self.sim_voice_ok:
            self.sim_state = "ARMED"
        elif self.sim_face_ok:
            self.sim_state = "FACE_OK"
        else:
            self.sim_state = "LOCKED"

        # 1. Картка Крок 1: Face ID + Liveness
        f_col = "#22c55e" if self.sim_face_ok else ("#38bdf8" if self.sim_running else "#94a3b8")
        f_icon = ft.Icons.VERIFIED_USER if self.sim_face_ok else ft.Icons.FACE_RETOUCHING_NATURAL
        f_title = f"Крок 1: Face ID + 3D Liveness — {'ПІДТВЕРДЖЕНО (' + (self.sim_owner or 'Власник') + ')' if self.sim_face_ok else 'Очікування обличчя власника'}"
        f_sub = "Жива людина в кадрі підтверджена, захист від фото пройдено." if self.sim_face_ok else "Подивіться в камеру та злегка поверніть голову вліво-вправо."
        if self.sim_step1_card is not None:
            self.sim_step1_card.content = self._glass(
                blur=False, radius=8, padding=10,
                border=ft.Border.all(1.5 if self.sim_face_ok else 1, f_col),
                content=ft.Row(spacing=12, controls=[
                    self._icon_badge(f_icon, f_col, 36, 18),
                    ft.Column(spacing=1, expand=True, controls=[
                        ft.Text(f_title, size=12, weight=ft.FontWeight.W_700, color=f_col),
                        ft.Text(f_sub, size=11, color=ft.Colors.ON_SURFACE_VARIANT)])]))

        # 2. Картка Крок 2: Voice ID
        v_col = "#22c55e" if self.sim_voice_ok else ("#f59e0b" if self.sim_face_ok else "#94a3b8")
        v_icon = ft.Icons.VERIFIED if self.sim_voice_ok else ft.Icons.RECORD_VOICE_OVER
        v_title = f"Крок 2: Голосовий ключ (Voice ID) — {'ПІДТВЕРДЖЕНО (' + (self.sim_voice_name or 'Диктор') + ')' if self.sim_voice_ok else 'Очікування голосу'}"
        v_sub = "Тембр голосу збігається з базою Voice DB." if self.sim_voice_ok else "Натисніть «2. Сканувати голосовий ключ (3 с)» та промовте парольну фразу."
        if self.sim_step2_card is not None:
            self.sim_step2_card.content = self._glass(
                blur=False, radius=8, padding=10,
                border=ft.Border.all(1.5 if self.sim_voice_ok else 1, v_col),
                content=ft.Row(spacing=12, controls=[
                    self._icon_badge(v_icon, v_col, 36, 18),
                    ft.Column(spacing=1, expand=True, controls=[
                        ft.Text(v_title, size=12, weight=ft.FontWeight.W_700, color=v_col),
                        ft.Text(v_sub, size=11, color=ft.Colors.ON_SURFACE_VARIANT)])]))

        # 3. Картка Крок 3: Статус замка та жестів
        armed = self.sim_state == "ARMED"
        a_col = "#22c55e" if armed else "#ef4444"
        a_icon = ft.Icons.LOCK_OPEN if armed else ft.Icons.LOCK_OUTLINE
        a_title = f"Крок 3: Привід Робота — {'🔓 РОЗБЛОКОВАНО (ARMED)' if armed else '🔒 ЗАБЛОКОВАНО (LOCKED)'}"
        a_sub = f"Активний жест: «{self.sim_last_gesture}» → Команди передаються на шасі!" if armed else "Виконайте Кроки 1 та 2, щоб відкрити замок керування."
        if self.sim_step3_card is not None:
            self.sim_step3_card.content = self._glass(
                blur=False, radius=8, padding=10,
                border=ft.Border.all(1.5, a_col),
                content=ft.Row(spacing=12, controls=[
                    self._icon_badge(a_icon, a_col, 36, 18),
                    ft.Column(spacing=1, expand=True, controls=[
                        ft.Text(a_title, size=12, weight=ft.FontWeight.W_800, color=a_col),
                        ft.Text(a_sub, size=11, color=ft.Colors.ON_SURFACE_VARIANT)])]))

        if self.sim_status_banner is not None:
            b_col = "#22c55e" if armed else ("#38bdf8" if self.sim_face_ok else "#f59e0b")
            b_txt = (
                f"✅ 2FA ПРОЙДЕНО ({self.sim_owner}) — РОБОТ АКТИВНИЙ! Покажіть жест у камеру."
                if armed else
                ("🔄 Крок 1 виконано! Тепер натисніть «2. Сканувати голосовий ключ (3 с)»." if self.sim_face_ok
                 else "🔒 Робот заблокований. Увімкніть камеру полігону та пройдіть верифікацію.")
            )
            self.sim_status_banner.content = ft.Container(
                padding=ft.Padding.symmetric(horizontal=14, vertical=10), border_radius=8,
                bgcolor=op(0.15, b_col), border=ft.Border.all(1.5, b_col),
                content=ft.Row(spacing=10, controls=[
                    ft.Icon(ft.Icons.SHIELD if armed else ft.Icons.SECURITY, color=b_col, size=20),
                    ft.Text(b_txt, size=12, weight=ft.FontWeight.W_700, color=b_col, expand=True)]))

        if self.sim_arena_bot is not None:
            self.sim_arena_bot.bgcolor = "#22c55e" if armed else ("#38bdf8" if self.sim_face_ok else "#ef4444")

        if update:
            self._safe(self.sim_step1_card)
            self._safe(self.sim_step2_card)
            self._safe(self.sim_step3_card)
            self._safe(self.sim_status_banner)
            self._safe(self.sim_arena_bot)

    def _sim_reset_lock(self):
        self.sim_face_ok = False
        self.sim_liveness_ok = False
        self.sim_voice_ok = False
        self.sim_owner = None
        self.sim_voice_name = None
        self.sim_last_gesture = "Немає"
        self.sim_robot_cmd = "🔒 ЗАБЛОКОВАНО ОПЕРАТОРОМ (LOCKED)"
        if self.face_engine:
            self.face_engine.reset_liveness()
        if self.sim_cmd_text:
            self.sim_cmd_text.value = self.sim_robot_cmd
            self._safe(self.sim_cmd_text)
        self._refresh_sim_ui()
        self._toast("Робота переведено у захищений стан LOCKED", "info")

    def _sim_force_arm(self):
        self.sim_face_ok = True
        self.sim_liveness_ok = True
        self.sim_voice_ok = True
        self.sim_owner = "Оператор (Demo)"
        self.sim_voice_name = "Оператор (Demo)"
        self.sim_robot_cmd = "🔓 ГОТОВИЙ ДО КОМАНД (ARMED)"
        if self.sim_cmd_text:
            self.sim_cmd_text.value = self.sim_robot_cmd
            self._safe(self.sim_cmd_text)
        self._refresh_sim_ui()
        self._toast("2FA розблоковано в демо-режимі! Показуйте жести або натискайте кнопки руху.", "ok")

    def _sim_move_robot(self, dx: float, dy: float, cmd_label: str):
        if self.sim_state != "ARMED":
            self._toast("Спочатку розблокуйте робота (Face ID + Voice ID або кнопка Demo ARMED)!", "err")
            return
        self.sim_robot_x = max(0.05, min(0.95, self.sim_robot_x + dx))
        self.sim_robot_y = max(0.05, min(0.95, self.sim_robot_y + dy))
        self.sim_robot_cmd = f"🤖 КОМАНДА: {cmd_label}"
        if self.sim_arena_bot is not None:
            self.sim_arena_bot.left = int(self.sim_robot_x * (420 - 56))
            self.sim_arena_bot.top = int(self.sim_robot_y * (240 - 56))
            self._safe(self.sim_arena_bot)
        if self.sim_cmd_text is not None:
            self.sim_cmd_text.value = self.sim_robot_cmd
            self._safe(self.sim_cmd_text)

    def _sim_center_robot(self):
        self.sim_robot_x = 0.50
        self.sim_robot_y = 0.50
        self.sim_robot_cmd = "🛑 EMERGENCY_STOP / ЦЕНТР АРЕНИ"
        if self.sim_arena_bot is not None:
            self.sim_arena_bot.left = int(self.sim_robot_x * (420 - 56))
            self.sim_arena_bot.top = int(self.sim_robot_y * (240 - 56))
            self._safe(self.sim_arena_bot)
        if self.sim_cmd_text is not None:
            self.sim_cmd_text.value = self.sim_robot_cmd
            self._safe(self.sim_cmd_text)

    def _on_sim_voice_verify(self, e=None):
        if self.voice_is_checking:
            return
        self.voice_is_checking = True
        self._toast("🎙️ Говоріть голосовий пароль (3 с)...", "info")

        def run():
            try:
                if self.audio_mgr is None:
                    self.audio_mgr = AudioStreamManager()
                res = self.audio_mgr.run_spot_check(duration_sec=3.0)
                spk = res.get("speaker", {})
                profiles = self.audio_mgr.voice_id.load_profiles()
                if spk.get("is_verified"):
                    self.sim_voice_ok = True
                    self.sim_voice_name = spk.get("name", "Власник")
                    self._toast(f"Голосовий ключ прийнято: {self.sim_voice_name} ({spk.get('confidence_pct')}%)", "ok")
                elif not profiles and spk.get("speech_detected"):
                    # Якщо база голосів порожня, дозволяємо тест за фактом живої мови
                    self.sim_voice_ok = True
                    self.sim_voice_name = "Голос оператора (БД порожня)"
                    self._toast("Голос виявлено (база Voice ID порожня — увімкнено гостьовий дозвіл)", "ok")
                else:
                    self._toast("Голос не розпізнано або мови не виявлено! Зареєструйте голос у вкладці «Голос».", "err")
                self._refresh_sim_ui()
            except Exception as ex:
                self._toast(f"Помилка перевірки голосу: {ex}", "err")
            finally:
                self.voice_is_checking = False

        threading.Thread(target=run, daemon=True).start()

    def _toggle_sim_camera(self, e=None):
        if self.sim_running:
            self.sim_running = False
            self._style_sim_cam(False)
            self._toast("Камеру полігону зупинено")
            return

        if self._is_running():
            self._stop_pipeline(quiet=True)
            time.sleep(0.4)
        if self.face_embedded_running:
            self.face_embedded_running = False
            time.sleep(0.3)

        self.sim_running = True
        if self.face_engine:
            self.face_engine.reset_liveness()
        self._style_sim_cam(True)
        self._toast("Камеру 2FA-Полігону запущено!", "ok")
        threading.Thread(target=self._sim_camera_loop, daemon=True).start()

    def _style_sim_cam(self, running: bool):
        if self.sim_cam_btn is not None:
            self.sim_cam_btn.content.controls[0].icon = ft.Icons.STOP if running else ft.Icons.VIDEOCAM
            self.sim_cam_btn.content.controls[1].value = "Зупинити камеру полігону" if running else "1. Запустити камеру полігону"
            self.sim_cam_btn.gradient = ft.LinearGradient(
                begin=ft.Alignment.CENTER_LEFT, end=ft.Alignment.CENTER_RIGHT,
                colors=["#dc2626", "#b91c1c"] if running else [self.c1, self.c2])
            self._safe(self.sim_cam_btn)
        if not running:
            if self.sim_preview_img is not None:
                self.sim_preview_img.visible = False
                self._safe(self.sim_preview_img)
            if hasattr(self, "sim_preview_placeholder") and self.sim_preview_placeholder is not None:
                self.sim_preview_placeholder.visible = True
                self._safe(self.sim_preview_placeholder)

    def _sim_camera_loop(self):
        cam_id = read_cfg().get("settings", {}).get("camera_id", 0)
        cap = cv2.VideoCapture(cam_id)
        if not cap.isOpened():
            self.sim_running = False
            self._style_sim_cam(False)
            self._toast(f"Не вдалося відкрити камеру {cam_id}", "err")
            return

        # Завантажуємо жести з бібліотеки для реакції віртуального робота
        from core.models import FrameContext
        from core.model_manager import ModelManager
        from core.module_loader import load_gesture_module
        gestures = list_gestures()
        detectors = []
        req_models = set()
        for g in gestures:
            try:
                if g["type"] == "slider":
                    continue
                det, reqs, gname = load_gesture_module(g["path"], {})
                detectors.append((det, gname))
                for r in reqs:
                    req_models.add(r)
            except Exception:
                pass

        model_mgr = None
        if req_models:
            try:
                model_mgr = ModelManager(req_models)
            except Exception:
                model_mgr = None

        # Мапінг жестів на вектори руху віртуального робота на 2D арені
        move_vectors = [
            (0.0, -0.14, "MOVE_FORWARD (Вперед)"),
            (0.14, 0.0, "TURN_RIGHT (Праворуч)"),
            (0.0, 0.14, "MOVE_BACKWARD (Назад)"),
            (-0.14, 0.0, "TURN_LEFT (Ліворуч)"),
        ]
        last_move_time = 0.0
        start_time = time.time()
        frame_id = 0

        try:
            while self.alive and self.sim_running and self.tab == 3:
                ret, frame = cap.read()
                if not ret or frame is None:
                    time.sleep(0.03)
                    continue
                frame = cv2.flip(frame, 1)
                frame_id += 1
                now = time.time()

                # 1. Перевірка Face ID + Liveness
                if self.face_engine is not None:
                    f_profiles = self.face_engine.load_profiles()
                    f_res = self.face_engine.recognize(frame, check_liveness=True)
                    frame = self.face_engine.draw_results(frame, f_res)
                    if f_res:
                        best = f_res[0]
                        if best["is_known"] and best["is_authorized"]:
                            if not self.sim_face_ok:
                                self.sim_face_ok = True
                                self.sim_liveness_ok = True
                                self.sim_owner = best["name"]
                                self._refresh_sim_ui()
                        elif not f_profiles and best.get("liveness_ok"):
                            # Якщо Face DB ще порожня, дозволяємо тест за умови живості Liveness
                            if not self.sim_face_ok:
                                self.sim_face_ok = True
                                self.sim_liveness_ok = True
                                self.sim_owner = "Оператор (Liveness OK)"
                                self._refresh_sim_ui()

                # 2. Якщо 2FA розблоковано (ARMED) — розпізнаємо жести для руху робота
                if self.sim_state == "ARMED" and model_mgr is not None and detectors:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    h, w = frame.shape[:2]
                    ts_ms = int((now - start_time) * 1000)
                    landmarks = model_mgr.process(rgb, ts_ms)
                    ctx = FrameContext(
                        frame=frame, frame_rgb=rgb, timestamp=now,
                        frame_id=frame_id, width=w, height=h, landmarks=landmarks)
                    for idx, (det, gname) in enumerate(detectors):
                        det_res = det.process(ctx)
                        if det_res.triggered and (now - last_move_time) > 0.45:
                            last_move_time = now
                            self.sim_last_gesture = gname
                            vx, vy, v_lbl = move_vectors[idx % len(move_vectors)]
                            self._sim_move_robot(vx, vy, f"{gname} -> {v_lbl}")
                            self._refresh_sim_ui()

                # Малюємо статус 2FA на самому кадрі
                hud_col = (0, 220, 100) if self.sim_state == "ARMED" else ((255, 190, 0) if self.sim_face_ok else (60, 70, 230))
                cv2.rectangle(frame, (8, 8), (380, 42), (15, 20, 30), -1)
                cv2.rectangle(frame, (8, 8), (380, 42), hud_col, 2)
                cv2.putText(frame, f"2FA ROBOT: {self.sim_state} | Gesture: {self.sim_last_gesture}",
                            (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.52, hud_col, 2)

                uri = frame_to_data_uri(frame, quality=80)
                if uri and self.sim_preview_img is not None:
                    self.sim_preview_img.src = uri
                    if not self.sim_preview_img.visible:
                        self.sim_preview_img.visible = True
                        if hasattr(self, "sim_preview_placeholder") and self.sim_preview_placeholder:
                            self.sim_preview_placeholder.visible = False
                            self._safe(self.sim_preview_placeholder)
                    self._safe(self.sim_preview_img)

                time.sleep(0.035)
        finally:
            cap.release()
            if model_mgr is not None:
                try:
                    model_mgr.close()
                except Exception:
                    pass
            self.sim_running = False
            self._style_sim_cam(False)

    # ==================================================================
    # ВКЛАДКА 5 — ВИГЛЯД
    # ==================================================================
    def _view_appearance(self) -> ft.Column:
        mode_seg = ft.SegmentedButton(
            selected=[self.settings["mode"]], allow_empty_selection=False,
            segments=[ft.Segment(value="dark", label=ft.Text("Темна"), icon=ft.Icon(ft.Icons.DARK_MODE, size=18)),
                      ft.Segment(value="light", label=ft.Text("Світла"), icon=ft.Icon(ft.Icons.LIGHT_MODE, size=18)),
                      ft.Segment(value="system", label=ft.Text("Системна"), icon=ft.Icon(ft.Icons.SETTINGS_BRIGHTNESS, size=18))],
            on_change=lambda e: self._set_mode(list(e.control.selected)[0]))

        swatches = ft.Row(wrap=True, spacing=14, run_spacing=14, controls=[self._swatch(n, c) for n, c in PALETTES])

        theme_card = self._glass(hover=False, content=ft.Column(spacing=16, controls=[
            self._section_title("Тема", ft.Icons.PALETTE, "Перемикається миттєво, Ctrl+T — швидка зміна"), mode_seg,
            ft.Divider(height=1, color=op(0.12, ft.Colors.ON_SURFACE)),
            ft.Text("Акцентний колір", size=12, color=ft.Colors.ON_SURFACE_VARIANT), swatches]))

        def toggle(key: str, label: str, desc: str, icon) -> ft.Control:
            return ft.Row(alignment=ft.MainAxisAlignment.SPACE_BETWEEN, controls=[
                ft.Row(spacing=12, controls=[
                    ft.Icon(icon, color=self.c1), ft.Column(spacing=0, tight=True, controls=[
                        ft.Text(label, weight=ft.FontWeight.W_600),
                        ft.Text(desc, size=12, color=ft.Colors.ON_SURFACE_VARIANT)])]),
                ft.Switch(value=bool(self.settings[key]), on_change=lambda e, k=key: self._on_effect(k, bool(e.control.value)))])

        effects_card = self._glass(hover=False, content=ft.Column(spacing=14, controls=[
            self._section_title("Ефекти", ft.Icons.AUTO_AWESOME, "Вимкніть, якщо потрібна максимальна швидкодія"),
            toggle("animations", "Анімації інтерфейсу", "Каскадна поява, hover-підйом, пульсація", ft.Icons.ANIMATION),
            toggle("orbs", "Світлові плями на фоні", "Повільний дрейф кольорових туманностей", ft.Icons.BLUR_ON)]))

        keys = [
            ("Ctrl + 1 … 5", "Перемикання вкладок (Поза, Обличчя, Голос, Полігон 2FA, Вигляд)"),
            ("Ctrl + E", "Відкрити вікно експорту коду та ZIP-пакетів"),
            ("Ctrl + H", "Відкрити інтерактивний довідник і підказки"),
            ("Ctrl + T", "Перемкнути світлу / темну тему"),
        ]
        keys_card = self._glass(hover=False, content=ft.Column(spacing=12, controls=[
            self._section_title("Гарячі клавіші", ft.Icons.KEYBOARD),
            *[ft.Row(controls=[self._pill(k, self.c1), ft.Text(d, size=13)]) for k, d in keys]]))

        return self._view([ft.ResponsiveRow(spacing=18, run_spacing=18, controls=[
            ft.Container(col={"xs": 12, "lg": 7}, content=theme_card),
            ft.Container(col={"xs": 12, "lg": 5}, content=ft.Column(spacing=18, controls=[effects_card, keys_card]))])])

    def _swatch(self, name: str, color: str) -> ft.Control:
        selected = color == self.settings["accent"]

        def on_hover(e):
            hot = str(e.data).lower() == "true"
            e.control.scale = 1.18 if hot else 1
            self._safe(e.control)

        return ft.Container(
            width=64, height=64, border_radius=32, tooltip=name, alignment=ft.Alignment.CENTER,
            gradient=ft.LinearGradient(begin=ft.Alignment.TOP_LEFT, end=ft.Alignment.BOTTOM_RIGHT,
                                       colors=[color, shift_hue(color, 0.12)]),
            border=ft.Border.all(3, ft.Colors.ON_SURFACE) if selected else ft.Border.all(3, "transparent"),
            shadow=ft.BoxShadow(blur_radius=18, color=op(0.5, color), offset=ft.Offset(0, 6)),
            animate=self._anim(250), animate_scale=self._anim(220, ft.AnimationCurve.EASE_OUT_BACK),
            on_hover=on_hover if self.animations else None, on_click=lambda e, c=color: self._set_accent(c),
            content=ft.Icon(ft.Icons.CHECK, color="white", size=26) if selected else None)

    def _on_effect(self, key: str, value: bool):
        self.settings[key] = value
        if key == "animations":
            self.animations = value
        if key == "orbs":
            for orb in self.orbs:
                orb.visible = value
        self._save_settings()
        self._page_update()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="V-Pose Combine Studio (Flet GUI & Web)")
    parser.add_argument("--web", action="store_true", help="Запустити у веб-браузері (Web Browser Mode)")
    parser.add_argument("--host", default="0.0.0.0", help="IP адреса сервера (за замовчуванням 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8550, help="Порт веб-сервера (за замовчуванням 8550)")
    args, _ = parser.parse_known_args()

    def entry(page: ft.Page):
        PoseStudioApp(page)

    if args.web:
        print("\n" + "=" * 60)
        print("   V-POSE COMBINE STUDIO — WEB VERSION")
        print("=" * 60)
        print(f" -> Локальна адреса:   http://localhost:{args.port}")
        print(f" -> Адреса в мережі:   http://{args.host}:{args.port}")
        print(" -> Відкриваємо ваш веб-браузер за замовчуванням...")
        print("=" * 60 + "\n")
        if hasattr(ft, "run"):
            ft.run(entry, host=args.host, port=args.port, view=ft.AppView.WEB_BROWSER)
        else:
            ft.app(target=entry, host=args.host, port=args.port, view=ft.AppView.WEB_BROWSER)
    else:
        if hasattr(ft, "run"):
            ft.run(entry)
        else:
            ft.app(target=entry)


if __name__ == "__main__":
    main()
