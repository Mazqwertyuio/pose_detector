"""
Data Transfer Objects (DTO) для модульної системи комп'ютерного зору.
"""
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import numpy as np


@dataclass
class FrameContext:
    """Контекст поточного кадру, який передається у всі детектори жестів."""
    frame: np.ndarray             # BGR кадр OpenCV
    frame_rgb: np.ndarray         # RGB кадр для моделей
    timestamp: float              # Поточний час (в секундах)
    frame_id: int                 # Порядковий номер кадру
    width: int                    # Ширина кадру
    height: int                   # Висота кадру

    # Словник з результатами інференсу моделей:
    # "face": об'єкт результатів FaceLandmarker (face_landmarks, face_blendshapes)
    # "hands": об'єкт результатів HandLandmarker
    # "pose": об'єкт результатів PoseLandmarker
    landmarks: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DetectionResult:
    """Результат роботи детектора жесту."""
    triggered: bool                         # Чи активний жест прямо зараз
    value: float = 0.0                      # Значення (0.0 .. 1.0 або дискретне 1.0/0.0)
    confidence: float = 1.0                 # Впевненість детектора
    extra_data: Dict[str, Any] = field(default_factory=dict)  # Додаткові показники (EAR, кути тощо)


@dataclass
class CommandEvent:
    """Подія/команда, яка передається до модуля виконання дії."""
    event_id: str                           # Унікальний ідентифікатор зв'язки/події
    gesture_name: str                       # Назва жесту
    timestamp: float                        # Час виникнення
    value: float                            # Передане значення
    params: Dict[str, Any] = field(default_factory=dict)  # Параметри дії з pipeline.yaml
