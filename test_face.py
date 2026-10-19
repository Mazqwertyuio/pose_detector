"""
Живий тест розпізнавання облич (Face ID) через веб-камеру.
Використовує YuNet для детекції та SFace для 128-d верифікації.
Вихід: ESC або Q у вікні камери.
"""
import argparse
import sys
import time

import cv2
import numpy as np

from core.face_engine import FaceEngine


def run_face_test(camera_id: int = 0, score_thresh: float = 0.65, cosine_thresh: float = 0.363):
    engine = FaceEngine(score_thresh=score_thresh, cosine_thresh=cosine_thresh)
    profiles_count = len(engine.profiles)
    print(f"[FaceTest] Завантажено профілів облич: {profiles_count}", flush=True)

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"[FaceTest] ПОМИЛКА: Не вдалося відкрити камеру ID={camera_id}.", flush=True)
        return

    print("[FaceTest] Камеру запущено. Подивіться в об'єктив. ESC / Q — вихід.", flush=True)

    prev_time = time.time()
    fps = 30.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            now = time.time()
            dt = now - prev_time
            prev_time = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)

            results = engine.recognize(frame)

            # Верхня панель статусу
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, 55), (15, 20, 25), -1)
            cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

            cv2.putText(frame, f"FPS: {int(fps)}", (15, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
            cv2.putText(frame, f"Profiles: {profiles_count}", (120, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2)

            if not results:
                cv2.putText(frame, "No Face Detected", (280, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (120, 120, 120), 2)
            else:
                for r in results:
                    bbox = r["bbox"]
                    x, y, bw, bh = bbox
                    name = r["name"]
                    is_known = r["is_known"]
                    pct = r["confidence_pct"]

                    color = (0, 220, 0) if is_known else (0, 140, 255)
                    label = f"{name} ({pct}%)" if is_known else f"{name} ({pct}%)"

                    # Рамка навколо обличчя
                    cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 2)

                    # Підпис над рамкою
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(frame, (x, max(0, y - 28)), (x + label_size[0] + 12, max(28, y)), color, -1)
                    cv2.putText(frame, label, (x + 6, max(20, y - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                    # Орієнтири (очі, ніс, кути рота)
                    for pt in r["landmarks"]:
                        cv2.circle(frame, (pt[0], pt[1]), 3, (0, 255, 255), -1)

                    # Вивід у stdout для спостереження GUI
                    print(f"[FaceID] {name} (Схожість: {pct}%, Відомо: {is_known})", flush=True)

            cv2.imshow("Face ID Monitor (YuNet + SFace)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q'), ord('Q')):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description="Живий монітор розпізнавання облич Face ID")
    ap.add_argument("--camera", type=int, default=0, help="ID веб-камери")
    ap.add_argument("--threshold", type=float, default=0.363, help="Поріг схожості SFace")
    args = ap.parse_args()

    run_face_test(camera_id=args.camera, cosine_thresh=args.threshold)


if __name__ == "__main__":
    main()
