"""
Модуль нормалізації точок та розрахунку схожості зразків (Pose Matching).
Забезпечує інваріантність до положення в кадрі та відстані до камери.
"""
import math
from typing import List, Tuple, Dict, Any, Optional
import numpy as np


def extract_landmarks_array(landmark_list) -> np.ndarray:
    """Перетворює список NormalizedLandmark у масив numpy формату (N, 3)."""
    points = []
    for lm in landmark_list:
        points.append([lm.x, lm.y, getattr(lm, "z", 0.0)])
    return np.array(points, dtype=np.float32)


def normalize_landmarks(points: np.ndarray, model_type: str) -> np.ndarray:
    """
    Нормалізує точки:
    1. Центрування (віднімання базової кореневої точки).
    2. Масштабування (ділення на характерний розмір частини тіла).
    """
    pts = points.copy()
    num_pts = len(pts)

    if num_pts == 0:
        return pts

    # 1. Центрування відносно базової точки
    if model_type == "hands":
        # Коренева точка — зап'ястя (індекс 0)
        root = pts[0].copy()
        # Характерний масштаб — відстань від зап'ястя (0) до основи середнього пальця (9)
        ref_dist = float(np.linalg.norm(pts[9][:2] - pts[0][:2]))
        if ref_dist < 1e-4:
            ref_dist = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
    elif model_type == "face":
        # Коренева точка — кінчик носа (індекс 1)
        root = pts[1].copy()
        # Характерний масштаб — ширина обличчя між щоками (234 і 454)
        if num_pts > 454:
            ref_dist = float(np.linalg.norm(pts[454][:2] - pts[234][:2]))
        else:
            ref_dist = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
    elif model_type == "pose":
        # Коренева точка — середина між плечима (11 і 12)
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

    # Зсув до центру та масштабування
    norm_pts = (pts - root) / ref_dist
    return norm_pts


def compute_similarity(
    current_norm: np.ndarray,
    template_norm: np.ndarray,
    tolerance: float = 0.35
) -> Tuple[float, float]:
    """
    Обчислює схожість між двома наборами нормалізованих точок.
    Повертає: (відсоток схожості від 0.0 до 1.0, середня евклідова відстань).
    """
    if len(current_norm) != len(template_norm):
        return 0.0, 999.0

    # Використовуємо 2D координати (x, y) для максимальної стійкості до шуму камери
    diff = current_norm[:, :2] - template_norm[:, :2]
    dists = np.linalg.norm(diff, axis=1)
    mean_dist = float(np.mean(dists))

    # Схожість лінійно спадає від 1.0 (при dist=0) до 0.0 (при dist>=tolerance)
    similarity = max(0.0, min(1.0, 1.0 - (mean_dist / tolerance)))
    return round(similarity, 3), round(mean_dist, 4)


def resample_trajectory(points: List[Tuple[float, float]], num_samples: int = 16) -> Optional[np.ndarray]:
    """
    Нормалізує та ресемплює 2D траєкторію (довжина дуги) до фіксованої кількості точок.
    Зсуває початок траєкторії в точку (0, 0).
    """
    if len(points) < 4:
        return None

    pts = np.array(points, dtype=np.float32)
    # Зсуваємо початок у (0, 0)
    pts = pts - pts[0]

    # Розрахунок кумулятивних відстаней вздовж кривої
    diffs = np.diff(pts, axis=0)
    step_dists = np.linalg.norm(diffs, axis=1)
    cum_dists = np.insert(np.cumsum(step_dists), 0, 0.0)
    total_length = float(cum_dists[-1])

    # Якщо рука майже не рухалася (менше 8% екрана) — це не рух
    if total_length < 0.08:
        return None

    # Рівномірний ресемплінг 16 точок
    target_dists = np.linspace(0.0, total_length, num_samples)
    resampled_x = np.interp(target_dists, cum_dists, pts[:, 0])
    resampled_y = np.interp(target_dists, cum_dists, pts[:, 1])

    return np.column_stack((resampled_x, resampled_y)).astype(np.float32)


def compute_trajectory_similarity(
    current_traj: np.ndarray,
    template_traj: np.ndarray,
    tolerance: float = 0.30
) -> Tuple[float, float]:
    """
    Обчислює схожість двох траєкторій однакової довжини (після ресемплінгу).
    """
    if len(current_traj) != len(template_traj):
        return 0.0, 999.0

    dists = np.linalg.norm(current_traj - template_traj, axis=1)
    mean_dist = float(np.mean(dists))

    similarity = max(0.0, min(1.0, 1.0 - (mean_dist / tolerance)))
    return round(similarity, 3), round(mean_dist, 4)


def count_oscillations(values: List[float], min_amplitude: float = 0.08) -> int:
    """
    Рахує кількість коливань (змін напрямку руху) у часовому ряді.
    Повертає кількість значущих розворотів (піків/впадин).
    """
    if len(values) < 5:
        return 0

    # Згладжування шумів (ковзне середнє з вікном 3)
    smoothed = []
    for i in range(1, len(values) - 1):
        smoothed.append((values[i-1] + values[i] + values[i+1]) / 3.0)

    if len(smoothed) < 3:
        return 0

    reversals = 0
    last_extreme = smoothed[0]
    direction = 0  # +1 вверх, -1 вниз

    for i in range(1, len(smoothed)):
        diff = smoothed[i] - smoothed[i-1]
        cur_dir = 1 if diff > 0 else (-1 if diff < 0 else 0)

        if cur_dir != 0:
            if direction != 0 and cur_dir != direction:
                amp = abs(smoothed[i-1] - last_extreme)
                if amp >= min_amplitude:
                    reversals += 1
                    last_extreme = smoothed[i-1]
            direction = cur_dir

    return reversals


