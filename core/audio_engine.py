import csv
import json
import os
import threading
import time
import uuid
from collections import deque
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import onnxruntime as ort
import sounddevice as sd

YAMNET_PATH = os.path.join("models", "yamnet.onnx")
CLASS_MAP_PATH = os.path.join("models", "yamnet_class_map.csv")
VOICES_DIR = os.path.join("modules", "voices")
VOICE_PROFILES_FILE = os.path.join(VOICES_DIR, "profiles.json")

# Українські переклади найпоширеніших акустичних подій
UA_TRANSLATIONS = {
    "Speech": "Мова / Голос",
    "Conversation": "Розмова",
    "Child speech, kid speaking": "Дитяча мова",
    "Laughter": "Сміх",
    "Giggle": "Хіхікання",
    "Snicker": "Смішок",
    "Crying, sobbing": "Плач",
    "Sigh": "Зітхання",
    "Singing": "Спів",
    "Whispering": "Шепіт",
    "Whistling": "Свист",
    "Screaming": "Крик",
    "Shout": "Вигук",
    "Yell": "Гучний крик",
    "Cough": "Кашель",
    "Throat clearing": "Прокашлювання",
    "Sneeze": "Чхання",
    "Snort": "Фуркання",
    "Breathing": "Дихання",
    "Snoring": "Хропіння",
    "Cheering": "Оплески та вигуки",
    "Applause": "Оплески",
    "Clapping": "Плескання в долоні",
    "Finger snapping": "Клацання пальцями",
    "Hands": "Плеск / Рухи рук",
    "Footsteps": "Кроки",
    "Silence": "Тиша",
    "Music": "Музика",
    "Musical instrument": "Музичний інструмент",
    "Bell": "Дзвінок / Дзвін",
    "Door": "Двері",
    "Knock": "Стукіт у двері",
    "Tap": "Постукування",
    "Keyboard typing": "Друк на клавіатурі",
    "Computer keyboard": "Клавіатура",
    "Mouse click": "Клік мишкою",
    "Alarm": "Сигналізація / Будильник",
    "Siren": "Сирена",
    "Dog": "Гавкіт собаки",
    "Cat": "Нявкання кота",
    "Glass": "Дзвін скла",
    "Vehicle": "Транспортний звук",
    "Wind": "Шум вітру",
    "Water": "Звук води",
    "Rain": "Дощ",
}


class YamnetDetector:
    """Блок 1: Детекція будь-яких звуків навколишнього середовища через Google YAMNet."""

    def __init__(self, model_path: str = YAMNET_PATH, class_map_path: str = CLASS_MAP_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Файл моделі YAMNet відсутній: {model_path}")
        if not os.path.exists(class_map_path):
            raise FileNotFoundError(f"Файл класів YAMNet відсутній: {class_map_path}")

        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 2
        self.session = ort.InferenceSession(model_path, sess_options=opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

        self.classes: List[str] = []
        with open(class_map_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            self.classes = [row["display_name"] for row in reader]

    def infer(self, waveform: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        waveform: 1D np.float32 масив із частотою 16000 Гц.
        Повертає (probabilities [frames, 521], embeddings [frames, 1024]).
        """
        wav = waveform.astype(np.float32)
        if wav.ndim > 1:
            wav = wav.flatten()
        outputs = self.session.run(None, {self.input_name: wav})
        probs = outputs[0]        # [frames, 521]
        embeddings = outputs[1]   # [frames, 1024]
        return probs, embeddings

    def get_top_events(self, probs: np.ndarray, top_k: int = 4, min_score: float = 0.08) -> List[Dict[str, Any]]:
        """Усереднює ймовірності по кадрах і повертає top_k подій з перекладом."""
        if probs.shape[0] == 0:
            return []
        mean_scores = np.mean(probs, axis=0)
        top_indices = np.argsort(mean_scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(mean_scores[idx])
            if score < min_score:
                continue
            eng_name = self.classes[idx] if idx < len(self.classes) else f"Class_{idx}"
            uk_name = UA_TRANSLATIONS.get(eng_name, eng_name)
            results.append({
                "class_id": int(idx),
                "name": eng_name,
                "label": uk_name,
                "score": score,
                "confidence_pct": int(min(100, score * 100)),
            })
        return results


class VoiceIdEngine:
    """Блок 2: Розпізнавання власника голосу (Speaker Identification / Verification)."""

    def __init__(self, yamnet: YamnetDetector, threshold: float = 0.70):
        self.yamnet = yamnet
        self.threshold = threshold
        self.profiles: List[Dict[str, Any]] = []

        os.makedirs(VOICES_DIR, exist_ok=True)
        self.load_profiles()

    def load_profiles(self) -> List[Dict[str, Any]]:
        if os.path.exists(VOICE_PROFILES_FILE):
            try:
                with open(VOICE_PROFILES_FILE, "r", encoding="utf-8") as f:
                    self.profiles = json.load(f)
            except Exception:
                self.profiles = []
        else:
            self.profiles = []
        return self.profiles

    def save_profiles(self):
        with open(VOICE_PROFILES_FILE, "w", encoding="utf-8") as f:
            json.dump(self.profiles, f, indent=2, ensure_ascii=False)

    def delete_profile(self, profile_id: str) -> bool:
        init_len = len(self.profiles)
        self.profiles = [p for p in self.profiles if p.get("id") != profile_id]
        if len(self.profiles) != init_len:
            self.save_profiles()
            return True
        return False

    def extract_voiceprint(self, waveform: np.ndarray) -> Optional[np.ndarray]:
        """
        Вилучає нормалізований 1024-d вектор голосу з фрагментів мови (Speech).
        """
        probs, embeddings = self.yamnet.infer(waveform)
        if probs.shape[0] == 0:
            return None

        # Клас 0 у YAMNet — це "Speech"
        speech_scores = probs[:, 0]
        # Обираємо кадри, де ймовірність мови помітна, або всі якщо рівень високий
        speech_mask = speech_scores >= 0.25
        if not np.any(speech_mask):
            speech_mask = speech_scores >= 0.10

        if not np.any(speech_mask):
            return None

        selected_embeddings = embeddings[speech_mask]
        voiceprint = np.mean(selected_embeddings, axis=0)
        norm = np.linalg.norm(voiceprint)
        if norm > 1e-6:
            voiceprint = voiceprint / norm
            return voiceprint
        return None

    def register_speaker(self, name: str, waveform: np.ndarray) -> Dict[str, Any]:
        """Реєструє нового диктора за записаним аудіофрагментом."""
        voiceprint = self.extract_voiceprint(waveform)
        if voiceprint is None:
            raise ValueError("У записі не виявлено чіткого людського голосу. Спробуйте ще раз, говорячи ближче до мікрофона.")

        prof_id = str(uuid.uuid4())[:8]
        profile = {
            "id": prof_id,
            "name": name.strip(),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "embedding": voiceprint.tolist(),
        }
        self.profiles.append(profile)
        self.save_profiles()
        return profile

    def identify_speaker(self, waveform: np.ndarray) -> Dict[str, Any]:
        """
        Зіставляє поточний голосовий фрагмент із збереженими дикторами.
        """
        voiceprint = self.extract_voiceprint(waveform)
        if voiceprint is None:
            return {
                "name": "Немає мови",
                "id": None,
                "is_verified": False,
                "similarity": 0.0,
                "confidence_pct": 0,
                "speech_detected": False,
            }

        best_name = "Невідомий голос"
        best_id = None
        best_sim = -1.0
        is_verified = False

        for prof in self.profiles:
            stored_v = np.array(prof["embedding"], dtype=np.float32)
            sim = float(np.dot(voiceprint, stored_v))
            if sim > best_sim:
                best_sim = sim
                if best_sim >= self.threshold:
                    best_name = prof["name"]
                    best_id = prof["id"]
                    is_verified = True

        # Масштабуємо у % для UI: 0.70 -> ~70%, 0.85+ -> 95%
        pct = 0
        if best_sim > 0:
            pct = int(min(100, max(0, (best_sim / 0.88) * 100)))

        return {
            "name": best_name,
            "id": best_id,
            "is_verified": is_verified,
            "similarity": best_sim,
            "confidence_pct": pct,
            "speech_detected": True,
        }


class AudioStreamManager:
    """
    Потоковий фоновий менеджер:
    Захоплює аудіо з мікрофона (16kHz), рахує рівень гучності (VU Meter),
    та запускає YAMNet детекцію звуків і розпізнавання диктора.
    """

    def __init__(self, sample_rate: int = 16000, buffer_seconds: float = 1.0):
        self.sample_rate = sample_rate
        self.buffer_size = int(sample_rate * buffer_seconds)
        self.yamnet = YamnetDetector()
        self.voice_id = VoiceIdEngine(self.yamnet)

        self.stream: Optional[sd.InputStream] = None
        self.running = False
        self.thread: Optional[threading.Thread] = None

        self.lock = threading.Lock()
        self.audio_buffer = deque(maxlen=self.buffer_size)
        # RMS для шкали гучності та 16-смуговий спектр для осцилограми
        self.current_rms = 0.0
        self.current_spectrum: List[float] = [0.05] * 16
        self.top_events: List[Dict[str, Any]] = []
        self.speaker_info: Dict[str, Any] = {
            "name": "Очікування мови", "is_verified": False, "confidence_pct": 0, "speech_detected": False
        }

        # Стан запису нового диктора
        self.is_recording_speaker = False
        self.rec_speaker_frames: List[np.ndarray] = []
        self.rec_speaker_done = False

    @staticmethod
    def compute_16band_spectrum(samples: np.ndarray) -> List[float]:
        """Обчислює 16 смуг спектра (0.04 .. 1.0) для візуалізації живої осцилограми в GUI."""
        if samples is None or len(samples) < 32:
            return [0.04] * 16
        win = samples * np.hanning(len(samples))
        fft_mag = np.abs(np.fft.rfft(win))
        # Фокусуємось на голосовому та акустичному діапазоні (перші 64 біни при blocksize=1600 -> до 6.4 кГц)
        usable = fft_mag[1:min(len(fft_mag), 65)]
        if len(usable) < 16:
            usable = np.pad(usable, (0, 16 - len(usable)))
        bands = np.array_split(usable, 16)
        vals = []
        for b in bands:
            energy = float(np.mean(b)) if len(b) > 0 else 0.0
            norm_v = min(1.0, max(0.04, np.sqrt(energy) * 0.45))
            vals.append(round(norm_v, 3))
        return vals

    def start(self):
        if self.running:
            return
        self.running = True
        self.audio_buffer.clear()

        def audio_cb(indata, frames, time_info, status):
            if not self.running:
                return
            samples = indata[:, 0]
            spec = self.compute_16band_spectrum(samples)
            with self.lock:
                self.audio_buffer.extend(samples)
                if self.is_recording_speaker:
                    self.rec_speaker_frames.append(np.copy(samples))
                rms = float(np.sqrt(np.mean(samples ** 2)))
                self.current_rms = min(1.0, rms * 8.0)
                self.current_spectrum = spec

        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            callback=audio_cb,
            blocksize=int(self.sample_rate * 0.1),
        )
        self.stream.start()
        self.thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None

    def _worker_loop(self):
        """Аналізує буфер кожні 150 мс."""
        while self.running:
            time.sleep(0.15)
            with self.lock:
                if len(self.audio_buffer) < int(self.sample_rate * 0.5):
                    continue
                chunk = np.array(self.audio_buffer, dtype=np.float32)

            try:
                # 1. YAMNet події
                probs, _ = self.yamnet.infer(chunk)
                events = self.yamnet.get_top_events(probs, top_k=3, min_score=0.08)

                # 2. Розпізнавання диктора
                spk = self.voice_id.identify_speaker(chunk)

                with self.lock:
                    self.top_events = events
                    self.speaker_info = spk
            except Exception:
                pass

    def record_speaker_sample(
        self,
        duration_sec: float = 3.5,
        progress_cb: Optional[Callable[[float, float], None]] = None,
        spectrum_cb: Optional[Callable[[List[float]], None]] = None,
    ) -> np.ndarray:
        """
        Захоплює чисте аудіо для реєстрації диктора або разової перевірки.
        Працює як при увімкненому фоновому потоці, так і автономно (on-demand).
        """
        if self.running:
            with self.lock:
                self.rec_speaker_frames = []
                self.is_recording_speaker = True

            start = time.time()
            while time.time() - start < duration_sec:
                time.sleep(0.06)
                elapsed = time.time() - start
                if progress_cb:
                    progress_cb(min(1.0, elapsed / duration_sec), self.current_rms)
                if spectrum_cb:
                    spectrum_cb(list(self.current_spectrum))

            with self.lock:
                self.is_recording_speaker = False
                if self.rec_speaker_frames:
                    full_wav = np.concatenate(self.rec_speaker_frames, axis=0)
                else:
                    full_wav = np.zeros(int(self.sample_rate * duration_sec), dtype=np.float32)
            return full_wav
        else:
            collected = []
            start = time.time()

            def cb(indata, frames, time_info, status):
                samples = indata[:, 0].copy()
                collected.append(samples)
                elapsed = time.time() - start
                rms = float(np.sqrt(np.mean(samples ** 2)))
                spec = self.compute_16band_spectrum(samples)
                self.current_spectrum = spec
                if progress_cb:
                    progress_cb(min(1.0, elapsed / duration_sec), min(1.0, rms * 8.0))
                if spectrum_cb:
                    spectrum_cb(spec)

            stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                callback=cb,
                blocksize=int(self.sample_rate * 0.08),
            )
            with stream:
                time.sleep(duration_sec)

            if collected:
                return np.concatenate(collected, axis=0)
            return np.zeros(int(self.sample_rate * duration_sec), dtype=np.float32)

    def run_spot_check(
        self,
        duration_sec: float = 3.0,
        progress_cb: Optional[Callable[[float, float], None]] = None,
        spectrum_cb: Optional[Callable[[List[float]], None]] = None,
    ) -> Dict[str, Any]:
        """
        Цільова разова перевірка звуку та голосу (на вимогу користувача, без постійного фонового режиму):
        1) Записує фрагмент аудіо тривалістю duration_sec (із оновленням прогресу, VU та 16-смугової осцилограми);
        2) Класифікує тип звуку за допомогою Google YAMNet;
        3) Визначає приналежність голосу за допомогою біометричного Voice ID.
        """
        waveform = self.record_speaker_sample(
            duration_sec=duration_sec, progress_cb=progress_cb, spectrum_cb=spectrum_cb
        )

        # 1. YAMNet аналіз звуків оточення
        probs, _ = self.yamnet.infer(waveform)
        events = self.yamnet.get_top_events(probs, top_k=4, min_score=0.06)

        # 2. Voice ID аналіз диктора
        spk = self.voice_id.identify_speaker(waveform)

        # Максимальна амплітуда/гучність
        rms = float(np.sqrt(np.mean(waveform ** 2))) if len(waveform) > 0 else 0.0

        return {
            "top_events": events,
            "primary_event": events[0] if events else None,
            "speaker": spk,
            "rms": min(1.0, rms * 8.0),
            "duration": duration_sec,
        }

    def get_snapshot(self) -> Dict[str, Any]:
        """Повертає поточний знімок стану для UI."""
        with self.lock:
            return {
                "rms": self.current_rms,
                "spectrum": list(self.current_spectrum),
                "top_events": list(self.top_events),
                "speaker": dict(self.speaker_info),
                "running": self.running,
            }
