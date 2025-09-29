#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
detector.py
-----------
Minimal motion detection over MJPEG UDP stream.
It imports start/stop from stream_control.py to keep code clean.

Usage:
  1) python detector.py
  2) Press 'q' to quit.

Notes:
  - This script starts the stream on begin and stops it on exit.
  - No PID validation: if someone else started the stream and used a
    different pid file, stop_stream() here may not stop theirs.
"""

import time
from datetime import datetime

import cv2

from stream_control import start_stream, stop_stream

# Stream URL split for line length limits.
UDP_BASE = "udp://127.0.0.1:5800"
UDP_OPTS = "?overrun_nonfatal=1&fifo_size=50000000"
UDP_URL = UDP_BASE + UDP_OPTS


def open_capture(url: str) -> cv2.VideoCapture:
    """
    Open the UDP stream with FFmpeg backend.
    Methods used:
      - cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    Variables created:
      - cap: the capture object bound to the UDP url
    """
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    return cap


def make_bgs() -> cv2.BackgroundSubtractor:
    """
    Create a background subtractor (MOG2) for quick motion masks.
    Methods used:
      - cv2.createBackgroundSubtractorMOG2
    Variables created:
      - bgs: the subtractor instance
    """
    bgs = cv2.createBackgroundSubtractorMOG2(
        history=300, varThreshold=25, detectShadows=True
    )
    return bgs


def process_frame(frame, bgs):
    """
    Turn frame into a cleaned binary mask and simple bounding boxes.
    Methods used:
      - bgs.apply, threshold, morphology, findContours, boundingRect
    Variables created:
      - mask: binary foreground mask
      - boxes: list of (x, y, w, h) for moving blobs
    """
    fg = bgs.apply(frame, learningRate=0.01)
    _, mask = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in cnts:
        if cv2.contourArea(c) < 1200:
            continue
        x, y, w, h = cv2.boundingRect(c)
        boxes.append((x, y, w, h))
    return mask, boxes


def draw_boxes(img, boxes):
    """
    Draw boxes on a copy of the frame.
    Methods used:
      - cv2.rectangle, cv2.putText
    Variables created:
      - vis: annotated frame copy
    """
    vis = img.copy()
    for (x, y, w, h) in boxes:
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.putText(vis, "moving", (x, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 255), 1, cv2.LINE_AA)
    return vis


def main():
    """
    Start stream, open capture, run a simple detection loop, then stop.
    Methods used:
      - start_stream, stop_stream
      - open_capture, process_frame, draw_boxes
    Variables created:
      - cap: VideoCapture
      - bgs: subtractor
    """
    # Start the camera stream (idempotent via pid file).
    start_stream()

    # Give rpicam-vid a short moment to warm up.
    time.sleep(1.0)

    cap = open_capture(UDP_URL)
    if not cap.isOpened():
        print("ERROR: cannot open UDP stream.")
        stop_stream()
        return

    bgs = make_bgs()
    print("Press 'q' to quit.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.02)
                continue

            mask, boxes = process_frame(frame, bgs)
            vis = draw_boxes(frame, boxes)

            cv2.imshow("Detector", vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        # Stop the camera stream we started.
        stop_stream()


if __name__ == "__main__":
    main()
