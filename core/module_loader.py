"""
Динамічний завантажувач модулів (Gestures та Actions).
Завантажує та валідує файли .py відповідно до єдиного стандарту.
"""
import os
import sys
import importlib.util
from typing import Dict, Any, Tuple


def _import_file_as_module(filepath: str):
    """Динамічно імпортує Python файл за вказаним шляхом."""
    abs_path = os.path.abspath(filepath)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"Файл модуля не знайдено: {abs_path}")

    module_name = f"dynamic_{os.path.splitext(os.path.basename(abs_path))[0]}_{abs(hash(abs_path))}"
    spec = importlib.util.spec_from_file_location(module_name, abs_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Неможливо завантажити специфікацію модуля з {abs_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_gesture_module(filepath: str, user_config: Dict[str, Any]) -> Tuple[Any, list, str]:
    """
    Завантажує модуль детектора жесту (.py або .json записаного зразка).
    Повертає: (екземпляр GestureDetector, список REQUIRES, ім'я модуля)
    """
    abs_path = os.path.abspath(filepath)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"Файл жесту не знайдено: {abs_path}")

    # Якщо це файл записаного зразка JSON
    if abs_path.endswith(".json"):
        import json
        from core.template_detector import TemplateGestureDetector

        with open(abs_path, "r", encoding="utf-8") as f:
            template_data = json.load(f)

        module_name = template_data.get("name", os.path.splitext(os.path.basename(abs_path))[0])
        model_type = template_data.get("model_type", "hands")
        requires = [model_type]

        detector_instance = TemplateGestureDetector(template_data, user_config)
        return detector_instance, requires, module_name

    # Якщо це звичайний Python файл .py
    module = _import_file_as_module(filepath)

    if not hasattr(module, "GestureDetector"):
        raise AttributeError(f"Модуль {filepath} повинен містити клас 'GestureDetector'")

    module_name = getattr(module, "MODULE_NAME", os.path.basename(filepath))
    requires = getattr(module, "REQUIRES", [])
    default_config = getattr(module, "DEFAULT_CONFIG", {})

    merged_config = {**default_config, **user_config}
    detector_instance = module.GestureDetector(merged_config)

    return detector_instance, requires, module_name


def load_action_module(filepath: str, user_config: Dict[str, Any]) -> Tuple[Any, str]:
    """
    Завантажує модуль виконання дії.
    Повертає: (екземпляр ActionExecutor, ім'я модуля)
    """
    module = _import_file_as_module(filepath)

    if not hasattr(module, "ActionExecutor"):
        raise AttributeError(f"Модуль {filepath} повинен містити клас 'ActionExecutor'")

    module_name = getattr(module, "MODULE_NAME", os.path.basename(filepath))
    default_config = getattr(module, "DEFAULT_CONFIG", {})

    merged_config = {**default_config, **user_config}
    action_instance = module.ActionExecutor(merged_config)

    return action_instance, module_name


def load_source_module(filepath: str, user_config: Dict[str, Any]) -> Tuple[Any, str]:
    """
    Завантажує модуль джерела відеопотоку.
    Повертає: (екземпляр VideoSource, ім'я модуля)
    """
    module = _import_file_as_module(filepath)

    if not hasattr(module, "VideoSource"):
        raise AttributeError(f"Модуль {filepath} повинен містити клас 'VideoSource'")

    module_name = getattr(module, "MODULE_NAME", os.path.basename(filepath))
    default_config = getattr(module, "DEFAULT_CONFIG", {})

    merged_config = {**default_config, **user_config}
    source_instance = module.VideoSource(merged_config)

    return source_instance, module_name
