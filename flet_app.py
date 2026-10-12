"""
V-Pose Combine Studio — графічний інтерфейс на Flet (Material 3).

Author: M. Korolkov
License: MIT

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
