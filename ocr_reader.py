#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
lp_reader.py
------------
Low-latency LPR over TCP stream:
- Uses fixed ROI for stability (set USE_FIXED_ROI + coords).
- Flushes buffered frames to avoid long lag.
- Light preprocessing (unsharp + CLAHE) + two thresholds.
- Temporal debounce so the label is stable on screen.

Hotkeys:
- 'q' -> quit (stops the stream)
- 's' -> save annotated frame
"""

import time
from collections import deque, Counter
from datetime import datetime

import cv2
import numpy as np
import pytesseract

from stream_control import start_stream, stop_stream

# ---- configuration -----------------------------------------------------

# Consumer (OpenCV) URL for TCP (no '?listen' here).
TCP_URL = "tcp://127.0.0.1:5800"

# Throttling and stability
OCR_EVERY = 5           # run OCR every N frames
HISTORY = 15            # keep last N recognized strings
REQUIRED = 4            # accept when a string seen >= REQUIRED
PLATE_MIN = 5           # min accepted length
PLATE_MAX = 8           # max accepted length

# Fixed ROI (x1, y1, x2, y2). Toggle ON for stability.
USE_FIXED_ROI = False
FIXED_ROI = (520, 360, 1400, 760)  # <-- tune to your plate area

# Tesseract config: single line, A-Z0-9 only.
TESS_CFG = (
    "--oem 1 --psm 7 -c "
    "tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
)


# ---- helpers -----------------------------------------------------------

def open_capture(url: str) -> cv2.VideoCapture:
    """
    Open the TCP stream via FFmpeg backend and try to reduce latency.
    Methods used:
      - cv2.VideoCapture
      - CAP_PROP_BUFFERSIZE set to 1 (may be ignored by backend)
    Variables created:
      - cap: capture object
    """
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    # Try to shrink internal buffer (best effort).
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def flush_frames(cap: cv2.VideoCapture, grabs: int = 5) -> None:
    """
    Drop a few buffered frames so we stay close to 'live'.
    Methods used:
      - cap.grab()
    Variables created:
      - none
    """
    for _ in range(grabs):
        cap.grab()


def preprocess_roi(roi_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Produce two binarized variants for OCR.
    Methods used:
      - gray, unsharp (via addWeighted), CLAHE, OTSU, invert
    Variables created:
      - th_a: normal binary
      - th_b: inverted binary
    """
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (0, 0), 1.0)
    sharp = cv2.addWeighted(gray, 1.6, blur, -0.6, 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq = clahe.apply(sharp)
    _, th_a = cv2.threshold(eq, 0, 255,
                            cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    th_b = cv2.bitwise_not(th_a)
    return th_a, th_b


def normalize_plate(s: str) -> str:
    """
    Normalize OCR text lightly (no S->5, no B->8).
    Methods used:
      - upper, strip, allowlist filter, basic O→0 and I→1
    Variables created:
      - cleaned: normalized string
    """
    s = s.upper().strip().replace(" ", "")
    s = s.replace("O", "0").replace("|", "1")
    allowed = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    return "".join(ch for ch in s if ch in allowed)


def ocr_best(roi_bgr: np.ndarray) -> str:
    """
    Upscale ROI, build two binaries, run OCR on both and pick longer.
    Methods used:
      - cv2.resize, preprocess_roi, pytesseract.image_to_string
    Variables created:
      - txt_a, txt_b, norm_a, norm_b
    """
    roi_up = cv2.resize(
        roi_bgr, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC
    )
    th_a, th_b = preprocess_roi(roi_up)
    txt_a = pytesseract.image_to_string(th_a, config=TESS_CFG)
    txt_b = pytesseract.image_to_string(th_b, config=TESS_CFG)
    norm_a = normalize_plate(txt_a)
    norm_b = normalize_plate(txt_b)
    return norm_a if len(norm_a) >= len(norm_b) else norm_b


def draw_label(img, text, org, color=(0, 255, 0)):
    """
    Draw a visible text label with filled background.
    Methods used:
      - getTextSize, rectangle, putText
    Variables created:
      - none (in-place)
    """
    if not text:
        return
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thick = 0.8, 2
    (w, h), _ = cv2.getTextSize(text, font, scale, thick)
    x, y = org
    cv2.rectangle(img, (x - 2, y - h - 8), (x + w + 2, y + 4),
                  (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), font, scale, color, thick, cv2.LINE_AA)


def main():
    """
    Start the stream, open capture, keep latency low by flushing,
    run throttled OCR on fixed ROI, and show the stable result.
    Methods used:
      - start_stream/stop_stream, flush_frames, deque vote
    """
    try:
        print("Tesseract:", pytesseract.get_tesseract_version(),
              flush=True)
    except Exception:
        print("Tesseract check failed (continuing)...", flush=True)

    start_stream()
    time.sleep(0.7)

    cap = open_capture(TCP_URL)
    if not cap.isOpened():
        print("ERROR: cannot open TCP stream.", flush=True)
        stop_stream()
        return

    hist = deque(maxlen=HISTORY)
    frame_id = 0
    print("Press 'q' to quit, 's' to save.", flush=True)

    try:
        while True:
            # Grab+flush to stay live and avoid buffered lag
            flush_frames(cap, grabs=4)

            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.01)
                continue

            frame_id += 1
            view = frame.copy()

            if USE_FIXED_ROI:
                x1, y1, x2, y2 = FIXED_ROI
                roi = frame[y1:y2, x1:x2]
                cv2.rectangle(view, (x1, y1), (x2, y2), (255, 0, 0), 2)
            else:
                roi = frame

            # Run OCR only every N-th frame
            plate = ""
            if frame_id % OCR_EVERY == 0:
                plate = ocr_best(roi)
                if PLATE_MIN <= len(plate) <= PLATE_MAX:
                    hist.append(plate)
                    print(f"OCR: {plate}", flush=True)

            # Temporal debounce: accept only if repeated
            stable = ""
            if hist:
                best, cnt = Counter(hist).most_common(1)[0]
                if cnt >= REQUIRED:
                    stable = best

            draw_label(view, stable or "(reading...)", (10, 34))
            cv2.imshow("LPR (low latency)", view)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('s'):
                name = datetime.now().strftime("lpr_%Y%m%d_%H%M%S.jpg")
                cv2.imwrite(name, view)
                print(f"Saved {name}", flush=True)

    finally:
        cap.release()
        cv2.destroyAllWindows()
        stop_stream()


if __name__ == "__main__":
    main()
