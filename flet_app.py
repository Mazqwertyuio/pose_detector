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
from typing import Any, Dict, List, Optional

import flet as ft

PIPELINE_FILE = "pipeline.yaml"
SETTINGS_FILE = "gui_settings.json"
GESTURES_DIR = os.path.join("modules", "gestures")
RECORDED_DIR = os.path.join("modules", "gestures", "recorded")
ACTIONS_DIR = os.path.join("modules", "actions")

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
    (ft.Icons.DASHBOARD_OUTLINED, ft.Icons.DASHBOARD, "Конвеєр", "Запуск, статус і консоль"),
    (ft.Icons.FIBER_MANUAL_RECORD_OUTLINED, ft.Icons.FIBER_MANUAL_RECORD, "Запис", "Створення власних рухів (6 парадигм)"),
    (ft.Icons.FOLDER_COPY_OUTLINED, ft.Icons.FOLDER_COPY, "Бібліотека", "Усі жести: код і записані зразки"),
    (ft.Icons.ACCOUNT_TREE_OUTLINED, ft.Icons.ACCOUNT_TREE, "Зв'язки", "Жест → дія: конструктор"),
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
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) / 255 for i in (0, 2, 4))
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    r, g, b = colorsys.hsv_to_rgb((h + delta) % 1.0, s, v)
    return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))


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
            axis_choice="1", slider_choice="1",
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
        self.lib_list: Optional[ft.Column] = None
        self.lib_count: Optional[ft.Text] = None
        self.bind_list: Optional[ft.Column] = None
        self.flow_row: Optional[ft.Row] = None
        self.key_field: Optional[ft.TextField] = None

        self._pulse_on = False
        self._compute_colors()
        self._setup_page()
        self._build_ui()

        atexit.register(self._cleanup)
        threading.Thread(target=self._ticker, daemon=True).start()

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
            return ["#070b18", "#120d26", "#07202b"]
        return ["#f4f7ff", "#f6f1ff", "#e8faf7"]

    def _on_key(self, e: ft.KeyboardEvent):
        if not e.ctrl:
            return
        key = str(e.key).upper()
        if key in ("1", "2", "3", "4", "5"):
            self._goto(int(key) - 1)
        elif key == "T":
            self._set_mode("light" if self._is_dark() else "dark")

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
            blur_radius=34 if lifted else 18, spread_radius=0,
            color=op(0.38 if lifted else 0.16, color or self.c1),
            offset=ft.Offset(0, 14 if lifted else 6))

    def _glass_border(self, lifted: bool = False, color: Optional[str] = None):
        if lifted:
            return ft.Border.all(1.4, op(0.65, color or self.c1))
        return ft.Border.all(1, op(0.14, ft.Colors.ON_SURFACE))

    def _glass(self, content: ft.Control, padding=18, radius=22, hover=True, blur=True,
               on_click=None, tint: Optional[str] = None, **kwargs) -> ft.Container:
        def on_hover(e):
            lifted = str(e.data).lower() == "true"
            e.control.offset = ft.Offset(0, -0.012) if lifted else ft.Offset(0, 0)
            e.control.shadow = self._shadow(lifted, tint)
            e.control.border = self._glass_border(lifted, tint)
            self._safe(e.control)

        return ft.Container(
            content=content, padding=padding, border_radius=radius,
            bgcolor=op(0.52, ft.Colors.SURFACE_CONTAINER),
            blur=ft.Blur(16, 16) if blur else None,
            clip_behavior=ft.ClipBehavior.ANTI_ALIAS,
            border=self._glass_border(False), shadow=self._shadow(False, tint),
            animate=self._anim(260), animate_offset=self._anim(220, ft.AnimationCurve.EASE_OUT_CUBIC),
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

    def _icon_badge(self, icon, color: str, size=46, icon_size=24) -> ft.Container:
        return ft.Container(
            width=size, height=size, border_radius=size / 3.2, alignment=ft.Alignment.CENTER,
            clip_behavior=ft.ClipBehavior.ANTI_ALIAS,
            gradient=ft.LinearGradient(begin=ft.Alignment.TOP_LEFT, end=ft.Alignment.BOTTOM_RIGHT,
                                       colors=[color, shift_hue(color, 0.1)]),
            shadow=ft.BoxShadow(blur_radius=14, color=op(0.45, color), offset=ft.Offset(0, 5)),
            content=ft.Icon(icon, color="white", size=icon_size))

    def _pill(self, text: str, color: str, icon=None) -> ft.Container:
        row: List[ft.Control] = []
        if icon:
            row.append(ft.Icon(icon, size=13, color=color))
        row.append(ft.Text(text, size=11, weight=ft.FontWeight.W_700, color=color))
        return ft.Container(
            content=ft.Row(tight=True, spacing=4, controls=row),
            padding=ft.Padding.symmetric(horizontal=10, vertical=4), border_radius=20,
            bgcolor=op(0.14, color), border=ft.Border.all(1, op(0.35, color)))

    def _gradient_button(self, label: str, icon, on_click, colors: Optional[List[str]] = None,
                         height=54) -> ft.Container:
        colors = colors or [self.c1, self.c2]

        def on_hover(e):
            hot = str(e.data).lower() == "true"
            e.control.scale = 1.035 if hot else 1
            self._safe(e.control)

        return ft.Container(
            height=height, padding=ft.Padding.symmetric(horizontal=26), border_radius=height / 2.6,
            alignment=ft.Alignment.CENTER, clip_behavior=ft.ClipBehavior.ANTI_ALIAS,
            gradient=ft.LinearGradient(begin=ft.Alignment.CENTER_LEFT, end=ft.Alignment.CENTER_RIGHT, colors=colors),
            shadow=ft.BoxShadow(blur_radius=22, color=op(0.42, colors[0]), offset=ft.Offset(0, 8)),
            animate=self._anim(250), animate_scale=self._anim(220, ft.AnimationCurve.EASE_OUT_BACK),
            animate_opacity=self._anim(250),
            on_hover=on_hover if self.animations else None, on_click=on_click,
            content=ft.Row(tight=True, alignment=ft.MainAxisAlignment.CENTER, spacing=10, controls=[
                ft.Icon(icon, color="white", size=22),
                ft.Text(label, color="white", size=15, weight=ft.FontWeight.W_700)]))

    def _section_title(self, text: str, icon=None, subtitle: Optional[str] = None) -> ft.Control:
        left: List[ft.Control] = []
        if icon:
            left.append(ft.Icon(icon, color=self.c1, size=22))
        col = [ft.Text(text, size=16, weight=ft.FontWeight.W_700)]
        if subtitle:
            col.append(ft.Text(subtitle, size=12, color=ft.Colors.ON_SURFACE_VARIANT))
        left.append(ft.Column(spacing=1, tight=True, controls=col))
        return ft.Row(spacing=10, controls=left)

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
        orb_specs = [(self.c1, 520, 0.05, 0.05), (self.c2, 460, 0.60, 0.40), ("#38bdf8", 380, 0.30, 0.65)]
        self.orbs = []
        for color, size, fx, fy in orb_specs:
            self.orbs.append(ft.Container(
                width=size, height=size, border_radius=size / 2, left=w * fx, top=h * fy,
                visible=bool(self.settings["orbs"]),
                animate_position=ft.Animation(11000, ft.AnimationCurve.EASE_IN_OUT),
                gradient=ft.RadialGradient(colors=[op(0.34 if self._is_dark() else 0.26, color), op(0, color)])))

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
            content=ft.Row(spacing=12, controls=[self.collapse_icon]), padding=14, border_radius=14, ink=True,
            tooltip="Згорнути / розгорнути панель", on_click=self._toggle_sidebar)
        self._apply_collapse_icon()

        return ft.Container(
            width=76 if collapsed else 238, animate=ft.Animation(380, ft.AnimationCurve.EASE_OUT_CUBIC),
            padding=ft.Padding(14, 22, 14, 16), bgcolor=op(0.5, ft.Colors.SURFACE_CONTAINER_LOW),
            blur=ft.Blur(18, 18), border=ft.Border(right=ft.BorderSide(1, op(0.12, ft.Colors.ON_SURFACE))),
            content=ft.Column(spacing=6, controls=[
                ft.Container(content=brand, padding=ft.Padding(2, 0, 0, 14)),
                *self.nav_items, ft.Container(expand=True), self.collapse_btn]))

    def _apply_collapse_icon(self):
        self.collapse_icon.icon = (ft.Icons.KEYBOARD_DOUBLE_ARROW_RIGHT if self.settings["collapsed"]
                                   else ft.Icons.KEYBOARD_DOUBLE_ARROW_LEFT)

    def _nav_item(self, index: int) -> ft.Container:
        icon, sel_icon, label, _ = NAV[index]
        selected = index == self.tab
        collapsed = bool(self.settings["collapsed"])
        icon_ctrl = ft.Icon(sel_icon if selected else icon, size=22,
                            color=self.c1 if selected else ft.Colors.ON_SURFACE_VARIANT,
                            animate_size=self._anim(200))
        text_ctrl = ft.Text(label, size=14, weight=ft.FontWeight.W_700 if selected else ft.FontWeight.W_500,
                            color=ft.Colors.ON_SURFACE if selected else ft.Colors.ON_SURFACE_VARIANT,
                            visible=not collapsed)

        def on_hover(e):
            if index == self.tab:
                return
            hot = str(e.data).lower() == "true"
            e.control.bgcolor = op(0.08, ft.Colors.ON_SURFACE) if hot else None
            e.control.scale = 1.03 if hot else 1
            self._safe(e.control)

        return ft.Container(
            data=dict(icon=icon_ctrl, text=text_ctrl), tooltip=f"{label}  (Ctrl+{index + 1})",
            padding=ft.Padding.symmetric(horizontal=14, vertical=13), border_radius=14,
            bgcolor=op(0.16, self.c1) if selected else None,
            border=ft.Border(left=ft.BorderSide(3, self.c1 if selected else "transparent")),
            animate=self._anim(240), animate_scale=self._anim(200),
            on_hover=on_hover, on_click=lambda e, i=index: self._goto(i),
            content=ft.Row(spacing=14, controls=[icon_ctrl, text_ctrl]))

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
        self.title_text = ft.Text(NAV[self.tab][2], size=26, weight=ft.FontWeight.W_800)
        self.subtitle_text = ft.Text(NAV[self.tab][3], size=13, color=ft.Colors.ON_SURFACE_VARIANT)

        self.status_dot = ft.Container(width=10, height=10, border_radius=5, bgcolor="#64748b",
                                       animate=self._anim(500), animate_opacity=ft.Animation(900, ft.AnimationCurve.EASE_IN_OUT))
        self.status_text = ft.Text("Зупинено", size=12, weight=ft.FontWeight.W_700)
        self.status_pill = ft.Container(
            padding=ft.Padding.symmetric(horizontal=14, vertical=8), border_radius=20,
            bgcolor=op(0.5, ft.Colors.SURFACE_CONTAINER), border=ft.Border.all(1, op(0.14, ft.Colors.ON_SURFACE)),
            animate=self._anim(400),
            content=ft.Row(tight=True, spacing=8, controls=[self.status_dot, self.status_text]))

        self.theme_icon_switcher = ft.AnimatedSwitcher(
            content=self._theme_icon(), transition=ft.AnimatedSwitcherTransition.ROTATION, duration=450)
        theme_btn = ft.Container(
            content=self.theme_icon_switcher, padding=11, border_radius=22, ink=True,
            bgcolor=op(0.5, ft.Colors.SURFACE_CONTAINER), border=ft.Border.all(1, op(0.14, ft.Colors.ON_SURFACE)),
            tooltip="Світла / темна тема (Ctrl+T)", on_click=lambda e: self._set_mode("light" if self._is_dark() else "dark"))

        return ft.Container(
            padding=ft.Padding(30, 22, 30, 12),
            content=ft.Row(alignment=ft.MainAxisAlignment.SPACE_BETWEEN, controls=[
                ft.Column(spacing=2, tight=True, controls=[self.title_text, self.subtitle_text]),
                ft.Row(spacing=12, controls=[self.status_pill, theme_btn])]))

    def _theme_icon(self) -> ft.Icon:
        dark = self._is_dark()
        return ft.Icon(ft.Icons.DARK_MODE if dark else ft.Icons.LIGHT_MODE, size=22,
                       color="#a5b4fc" if dark else "#f59e0b", key="dark" if dark else "light")

    def _set_mode(self, mode: str):
        self.settings["mode"] = mode
        self.page.theme_mode = self._theme_mode()
        self.bg.gradient = ft.LinearGradient(begin=ft.Alignment.TOP_LEFT, end=ft.Alignment.BOTTOM_RIGHT,
                                             colors=self._bg_colors())
        strength = 0.34 if self._is_dark() else 0.26
        for orb, color in zip(self.orbs, [self.c1, self.c2, "#38bdf8"]):
            orb.gradient = ft.RadialGradient(colors=[op(strength, color), op(0, color)])
        if self.theme_icon_switcher:
            self.theme_icon_switcher.content = self._theme_icon()
        self._save_settings()
        self._page_update()
        if self.tab == 4:
            self._show_tab(4)

    def _set_accent(self, color: str):
        self.settings["accent"] = color
        self._compute_colors()
        self.page.theme = ft.Theme(color_scheme_seed=self.c1, use_material3=True)
        self.page.dark_theme = ft.Theme(color_scheme_seed=self.c1, use_material3=True)
        self._save_settings()
        self._build_ui()
        self._toast("Акцентний колір змінено", "ok")

    # ------------------------------------------------------------------
    # Навігація
    # ------------------------------------------------------------------
    def _goto(self, index: int):
        if index == self.tab and self.switcher.content is not None and getattr(self.switcher.content, "key", None) == f"tab{index}":
            return
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
        builders = [self._view_pipeline, self._view_record, self._view_library,
                    self._view_bindings, self._view_appearance]
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
    # ВКЛАДКА 1 — КОНВЕЄР
    # ==================================================================
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
            padding=26, radius=28, tint=self.c1,
            content=ft.ResponsiveRow(
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
                spacing=24, run_spacing=20,
                controls=[
                    ft.Container(col={"xs": 12, "md": 4, "lg": 3}, content=power_stack, alignment=ft.Alignment.CENTER),
                    ft.Container(
                        col={"xs": 12, "md": 8, "lg": 9},
                        content=ft.Column(spacing=10, controls=[
                            self.run_title, self.run_sub, self.uptime_text,
                            ft.Container(height=4), camera_seg,
                            ft.Row(wrap=True, spacing=20, run_spacing=8, controls=[preview_sw, marks_sw, mirror_sw])
                        ])
                    )
                ]
            )
        )

        gestures = scan_gestures()
        tiles = ft.ResponsiveRow(spacing=16, run_spacing=16, controls=[
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
            hover=False, content=ft.Column(spacing=12, controls=[
                ft.Row(alignment=ft.MainAxisAlignment.SPACE_BETWEEN, controls=[
                    self._section_title("Консоль конвеєра", ft.Icons.TERMINAL, "Вивід процесу run.py у реальному часі"),
                    ft.IconButton(ft.Icons.DELETE_SWEEP, tooltip="Очистити консоль", on_click=self._clear_log)]),
                ft.Container(height=250, padding=14, border_radius=16, bgcolor=op(0.42, "#000000"),
                             border=ft.Border.all(1, op(0.1, ft.Colors.ON_SURFACE)), content=self.log_view)]))

        return self._view([hero, tiles, console])

    def _stat_tile(self, icon, label: str, value: int, color: str) -> ft.Control:
        number = ft.Text("0", size=30, weight=ft.FontWeight.W_800)
        tile = self._glass(
            tint=color, padding=18, col={"xs": 6, "md": 3},
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
        for proc in (self.proc, self.rec_proc, self.test_proc):
            try:
                if proc is not None and proc.poll() is None:
                    proc.terminate()
            except Exception:
                pass

    # ==================================================================
    # ВКЛАДКА 2 — ЗАПИС
    # ==================================================================
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
        model_seg = ft.SegmentedButton(
            selected=[self.rec_model], allow_empty_selection=False,
            segments=[ft.Segment(value=m[0], label=ft.Text(m[2]), icon=ft.Icon(m[3], size=18)) for m in MODELS],
            on_change=lambda e: setattr(self, "rec_model", list(e.control.selected)[0]))

        name_field = ft.TextField(
            label="Назва жесту (латиницею)", hint_text="fist_to_fingers, wave, volume_bar",
            value=self.rec_name, prefix_icon=ft.Icons.LABEL_OUTLINE, border_radius=14,
            on_change=lambda e: setattr(self, "rec_name", e.control.value), on_submit=self._on_record_click)
        self.rec_name_field = name_field

        self.rec_params_switcher = ft.AnimatedSwitcher(
            content=self._rec_params_panel(), transition=ft.AnimatedSwitcherTransition.FADE, duration=300)

        self.rec_status = ft.Text("", size=12, color=ft.Colors.ON_SURFACE_VARIANT, max_lines=2)
        self.rec_ring = ft.ProgressRing(width=22, height=22, stroke_width=3, visible=recording)
        self.rec_btn = self._gradient_button("Відкрити камеру й записати", ft.Icons.VIDEOCAM, self._on_record_click)
        self.rec_btn.opacity = 0.5 if recording else 1

        left = self._glass(
            hover=False, col={"xs": 12, "lg": 7},
            content=ft.Column(spacing=14, controls=[
                self._section_title("1. Оберіть характер руху", ft.Icons.CATEGORY, "Шість фундаментальних парадигм"), grid]))

        right = self._glass(
            hover=False, col={"xs": 12, "lg": 5},
            content=ft.Column(spacing=16, controls=[
                self._section_title("2. Налаштуйте і запишіть", ft.Icons.TUNE, "Параметри передаються рекордеру автоматично"),
                name_field,
                ft.Text("Частина тіла для відстеження", size=12, color=ft.Colors.ON_SURFACE_VARIANT),
                model_seg, self.rec_params_switcher,
                ft.Row(spacing=12, controls=[self.rec_btn, self.rec_ring]),
                self.rec_status,
                self._glass(hover=False, blur=False, padding=12, radius=14, content=ft.Row(spacing=10, controls=[
                    ft.Icon(ft.Icons.KEYBOARD, size=18, color=self.c1),
                    ft.Text("У вікні камери: ПРОБІЛ — зйомка, ESC / Q — скасування.", size=12,
                            color=ft.Colors.ON_SURFACE_VARIANT, expand=True)]))]))

        return self._view([ft.ResponsiveRow(spacing=18, run_spacing=18, controls=[left, right])])

    def _paradigm_card(self, p: Dict[str, Any]) -> ft.Container:
        selected = p["type"] == self.rec_type
        check = ft.Icon(ft.Icons.CHECK_CIRCLE, color=p["color"], size=22, scale=1 if selected else 0,
                        animate_scale=ft.Animation(350, ft.AnimationCurve.EASE_OUT_BACK))

        def on_hover(e):
            if p["type"] == self.rec_type:
                return
            hot = str(e.data).lower() == "true"
            e.control.offset = ft.Offset(0, -0.015) if hot else ft.Offset(0, 0)
            e.control.border = ft.Border.all(1, op(0.5, p["color"]) if hot else op(0.14, ft.Colors.ON_SURFACE))
            self._safe(e.control)

        card = ft.Container(
            col={"xs": 12, "sm": 6, "xl": 4}, padding=14, border_radius=18, data=dict(check=check),
            bgcolor=op(0.18 if selected else 0.35, p["color"] if selected else ft.Colors.SURFACE_CONTAINER_HIGH),
            border=ft.Border.all(1.6 if selected else 1, p["color"] if selected else op(0.14, ft.Colors.ON_SURFACE)),
            shadow=ft.BoxShadow(blur_radius=24, color=op(0.35, p["color"]), offset=ft.Offset(0, 8)) if selected else None,
            clip_behavior=ft.ClipBehavior.ANTI_ALIAS,
            animate=self._anim(280), animate_offset=self._anim(220, ft.AnimationCurve.EASE_OUT_CUBIC),
            on_hover=on_hover if self.animations else None, on_click=lambda e, t=p["type"]: self._select_paradigm(t),
            content=ft.Column(spacing=8, controls=[
                ft.Row(alignment=ft.MainAxisAlignment.SPACE_BETWEEN, controls=[self._icon_badge(p["icon"], p["color"], 42, 22), check]),
                ft.Text(f"{p['choice']}. {p['title']}", size=14, weight=ft.FontWeight.W_700),
                ft.Text(p["desc"], size=11.5, color=ft.Colors.ON_SURFACE_VARIANT)]))
        return card

    def _select_paradigm(self, ptype: str):
        self.rec_type = ptype
        for t, card in self.rec_cards.items():
            p = PARADIGM_BY_TYPE[t]
            sel = t == ptype
            card.bgcolor = op(0.18 if sel else 0.35, p["color"] if sel else ft.Colors.SURFACE_CONTAINER_HIGH)
            card.border = ft.Border.all(1.6 if sel else 1, p["color"] if sel else op(0.14, ft.Colors.ON_SURFACE))
            card.shadow = ft.BoxShadow(blur_radius=24, color=op(0.35, p["color"]), offset=ft.Offset(0, 8)) if sel else None
            card.scale = 1
            card.data["check"].scale = 1 if sel else 0
        if self.rec_params_switcher is not None:
            self.rec_params_switcher.content = self._rec_params_panel()
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

    def _rec_params_panel(self) -> ft.Control:
        t = self.rec_type
        rows: List[ft.Control] = []
        if t in ("static", "trajectory", "transition", "hold"):
            rows.append(self._slider_row("Поріг схожості", "threshold", 0.5, 0.95, 9))
        if t == "transition":
            rows.append(self._slider_row("Макс. час перетворення А → Б", "trans_time", 0.3, 2.0, 17, " с"))
        if t == "hold":
            rows.append(self._slider_row("Час утримання", "hold_seconds", 0.5, 5.0, 9, " с"))
        if t == "periodic":
            rows.append(ft.Text("Вісь коливань", size=12, color=ft.Colors.ON_SURFACE_VARIANT))
            rows.append(ft.SegmentedButton(
                selected=[self.rec_params["axis_choice"]], allow_empty_selection=False,
                segments=[ft.Segment(value="1", label=ft.Text("Горизонтально"), icon=ft.Icon(ft.Icons.SWAP_HORIZ, size=18)),
                          ft.Segment(value="2", label=ft.Text("Вертикально"), icon=ft.Icon(ft.Icons.SWAP_VERT, size=18))],
                on_change=lambda e: self.rec_params.__setitem__("axis_choice", list(e.control.selected)[0])))
            rows.append(self._slider_row("Кількість змін напрямку", "min_cycles", 2, 8, 6, "", as_int=True))
        if t == "slider":
            rows.append(ft.Text("Що керує повзунком", size=12, color=ft.Colors.ON_SURFACE_VARIANT))
            rows.append(ft.SegmentedButton(
                selected=[self.rec_params["slider_choice"]], allow_empty_selection=False,
                segments=[ft.Segment(value="1", label=ft.Text("Висота")),
                          ft.Segment(value="2", label=ft.Text("Горизонталь")),
                          ft.Segment(value="3", label=ft.Text("Щипок"))],
                on_change=lambda e: self.rec_params.__setitem__("slider_choice", list(e.control.selected)[0])))
        if t == "trajectory":
            rows.append(ft.Text("Рух записується 1 секунду після натискання ПРОБІЛ.", size=12,
                                color=ft.Colors.ON_SURFACE_VARIANT))
        info = PARADIGM_BY_TYPE[t]
        return ft.Container(
            key=f"params_{t}", padding=14, border_radius=16, bgcolor=op(0.1, info["color"]),
            border=ft.Border.all(1, op(0.3, info["color"])),
            content=ft.Column(spacing=8, controls=[
                ft.Row(spacing=8, controls=[ft.Icon(info["icon"], color=info["color"], size=18),
                                            ft.Text(info["title"], size=13, weight=ft.FontWeight.W_700)]), *rows]))

    def _build_preset(self, name: str) -> Dict[str, Any]:
        model_choice = {m[0]: m[1] for m in MODELS}[self.rec_model]
        pr = self.rec_params
        return dict(
            name=name, gesture_choice=PARADIGM_BY_TYPE[self.rec_type]["choice"], model_choice=model_choice,
            threshold=pr["threshold"], trans_time=pr["trans_time"], hold_seconds=pr["hold_seconds"],
            min_cycles=pr["min_cycles"], axis_choice=pr["axis_choice"], slider_choice=pr["slider_choice"])

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
        if os.path.exists(target):
            self._confirm("Жест уже існує", f"«{name}» буде перезаписано новим записом. Продовжити?",
                          "Перезаписати", lambda: self._launch_recorder(name))
        else:
            self._launch_recorder(name)

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
            hint_text="Пошук жесту…", prefix_icon=ft.Icons.SEARCH, border_radius=16, expand=True, value=self.lib_query,
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
                pills.append(self._pill(time.strftime("%d.%m.%Y %H:%M", time.localtime(g["created"])), "#94a3b8", ft.Icons.SCHEDULE))
            trailing = ft.IconButton(ft.Icons.DELETE_OUTLINE, icon_color="#f87171", tooltip="Видалити зразок",
                                     on_click=lambda e, gg=g: self._ask_delete(gg))
            tint = p["color"]

        testing_this = self.test_proc is not None and self.test_proc.poll() is None and self.test_target == g["path"]
        test_btn = ft.FilledTonalButton(
            "Стоп" if testing_this else "Перевірити",
            icon=ft.Icons.STOP_CIRCLE if testing_this else ft.Icons.PLAY_CIRCLE,
            tooltip="Перевірити жест наживо перед камерою",
            on_click=lambda e, gg=g: self._toggle_test(gg))

        actions = ft.Row(
            spacing=8, tight=True, vertical_alignment=ft.CrossAxisAlignment.CENTER,
            controls=[test_btn, trailing])

        title_ctrl = ft.Text(title, size=15, weight=ft.FontWeight.W_700, no_wrap=True, overflow=ft.TextOverflow.ELLIPSIS)
        pills_row = ft.Row(wrap=True, spacing=6, run_spacing=6, controls=pills)

        info_col = ft.Column(
            expand=True, spacing=4, tight=True,
            controls=[title_ctrl, sub, pills_row])

        return self._glass(
            tint=tint, blur=False, padding=ft.Padding.symmetric(horizontal=16, vertical=14), radius=18,
            animate_opacity=ft.Animation(300, ft.AnimationCurve.EASE_OUT),
            content=ft.Row(
                spacing=14, vertical_alignment=ft.CrossAxisAlignment.CENTER,
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
    # ВКЛАДКА 4 — ЗВ'ЯЗКИ
    # ==================================================================
    def _view_bindings(self) -> ft.Column:
        gestures = scan_gestures()
        actions = scan_actions()

        valid_g = {g["path"] for g in gestures}
        if self.bld["gesture"] not in valid_g:
            self.bld["gesture"] = gestures[0]["path"] if gestures else None
        valid_a = {a["path"] for a in actions}
        if self.bld["action"] not in valid_a:
            self.bld["action"] = actions[0]["path"] if actions else None

        g_dd = ft.Dropdown(
            label="Жест (тригер)", value=self.bld["gesture"], border_radius=14, expand=True,
            options=[ft.dropdown.Option(key=g["path"], text=("[PY] " if g["kind"] == "py" else "[ЗРАЗОК] ") + g["name"]) for g in gestures],
            on_select=lambda e: self._on_bld("gesture", e.control.value))
        a_dd = ft.Dropdown(
            label="Дія", value=self.bld["action"], border_radius=14, expand=True,
            options=[ft.dropdown.Option(key=a["path"], text=a["label"]) for a in actions],
            on_select=lambda e: self._on_bld("action", e.control.value))
        self.key_field = ft.TextField(
            label="Клавіша клавіатури", value=self.bld["key"], prefix_icon=ft.Icons.KEYBOARD, border_radius=14,
            hint_text="space, enter, up, down, w, a, s, d…", on_change=lambda e: self._on_bld("key", e.control.value))
        self.key_field.visible = "keyboard" in str(self.bld["action"])

        cd_text = ft.Text(f"{self.bld['cooldown']:g} с", size=13, weight=ft.FontWeight.W_700, color=self.c1)

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
            hover=False, col={"xs": 12, "lg": 6},
            content=ft.Column(spacing=16, controls=[
                self._section_title("Нова зв'язка", ft.Icons.ADD_LINK, "Збережеться у pipeline.yaml"),
                self._glass(hover=False, blur=False, radius=16, padding=14, content=self.flow_row),
                ft.Row(controls=[g_dd]), ft.Row(controls=[a_dd]), self.key_field, cooldown,
                self._gradient_button("Зберегти зв'язку", ft.Icons.SAVE, self._save_binding)]))

        self.bind_list = ft.Column(spacing=10, horizontal_alignment=ft.CrossAxisAlignment.STRETCH)
        self._refresh_bindings(update=False)
        active = self._glass(
            hover=False, col={"xs": 12, "lg": 6},
            content=ft.Column(spacing=14, controls=[
                self._section_title("Активні зв'язки", ft.Icons.HUB, "Саме вони запускаються конвеєром"), self.bind_list]))

        return self._view([ft.ResponsiveRow(spacing=18, run_spacing=18, controls=[builder, active])])

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
        g = os.path.splitext(os.path.basename(str(self.bld["gesture"] or "—")))[0]
        a = os.path.splitext(os.path.basename(str(self.bld["action"] or "—")))[0]
        label, icon = ACTION_LABELS.get(a, (a, ft.Icons.BOLT))
        if "keyboard" in a:
            label = f"{label}: {self.bld['key'] or '?'}"
        gesture_chip = self._pill(g, "#a855f7", ft.Icons.GESTURE)
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
                    ft.Text("Створіть першу в конструкторі ліворуч.", size=12, color=ft.Colors.ON_SURFACE_VARIANT)])))
        for idx, b in enumerate(bindings):
            g = os.path.splitext(os.path.basename(b.get("gesture", {}).get("file", "")))[0]
            a = os.path.splitext(os.path.basename(b.get("action", {}).get("file", "")))[0]
            label, icon = ACTION_LABELS.get(a, (a, ft.Icons.BOLT))
            key = b.get("action", {}).get("config", {}).get("key")
            cd = b.get("filter", {}).get("cooldown")
            actions = ft.Row(
                spacing=4, tight=True, vertical_alignment=ft.CrossAxisAlignment.CENTER,
                controls=[ft.IconButton(ft.Icons.DELETE_OUTLINE, icon_color="#f87171", tooltip="Видалити зв'язку",
                                        on_click=lambda e, i=idx: self._delete_binding(i))])
            row = self._glass(
                blur=False, radius=16, padding=ft.Padding.symmetric(horizontal=14, vertical=12),
                animate_opacity=ft.Animation(300, ft.AnimationCurve.EASE_OUT),
                content=ft.Row(spacing=12, vertical_alignment=ft.CrossAxisAlignment.CENTER, controls=[
                    self._icon_badge(ft.Icons.HUB, self.c1, 38, 20),
                    ft.Column(expand=True, spacing=4, tight=True, controls=[
                        ft.Row(wrap=True, spacing=8, run_spacing=6, controls=[
                            self._pill(g, "#a855f7", ft.Icons.GESTURE),
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
            self._toast("Оберіть і жест, і дію", "err")
            return
        cfg = read_cfg()
        bindings = cfg.get("bindings", []) or []
        g_name = os.path.splitext(os.path.basename(g_file))[0]
        a_name = os.path.splitext(os.path.basename(a_file))[0]
        action_config: Dict[str, Any] = {}
        if "keyboard" in a_name:
            action_config["key"] = (self.bld["key"] or "space").strip() or "space"
        bindings.append({
            "id": f"{g_name}_{len(bindings) + 1}",
            "name": f"{g_name} -> {a_name}",
            "gesture": {"file": g_file, "config": {}},
            "filter": {"cooldown": float(self.bld["cooldown"]), "trigger_on": "rising"},
            "action": {"file": a_file, "config": action_config},
        })
        cfg["bindings"] = bindings
        write_cfg(cfg)
        self._refresh_bindings()
        self._toast("Зв'язку збережено у pipeline.yaml", "ok")

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

        keys = [("Ctrl + 1 … 5", "Перемикання вкладок"), ("Ctrl + T", "Світла / темна тема")]
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
