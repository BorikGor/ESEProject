#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
stream_control.py
-----------------
Start/stop rpicam-vid MJPEG stream with a simple PID file.
TCP 'listen' is used to avoid UDP buffering and lag.
"""

import os
import signal
import subprocess

# Producer (camera) URL: listen on localhost TCP.
DEFAULT_URL = "tcp://127.0.0.1:5800?listen"

# Output quality/resolution; tune as needed.
DEFAULT_W = "1920"
DEFAULT_H = "1080"
DEFAULT_FPS = "15"
DEFAULT_QUALITY = "90"

DEFAULT_PID = "stream.pid"


def start_stream(url: str = DEFAULT_URL,
                 width: str = DEFAULT_W,
                 height: str = DEFAULT_H,
                 fps: str = DEFAULT_FPS,
                 quality: str = DEFAULT_QUALITY,
                 pid_file: str = DEFAULT_PID) -> None:
    """
    Start rpicam-vid in background and write its PID to pid_file.
    Methods used:
      - subprocess.Popen to launch 'rpicam-vid'
      - open(..., 'w') to write the PID
    Variables created:
      - proc: the Popen handle (local only)
    """
    if os.path.exists(pid_file):
        print("Stream already running (pid file exists).")
        return

    cmd = [
        "rpicam-vid",
        "-t", "0",
        "-n",
        "--codec", "mjpeg",
        "--quality", quality,
        "--width", width,
        "--height", height,
        "--framerate", fps,
        "-o", url
    ]
    proc = subprocess.Popen(cmd, start_new_session=True)
    with open(pid_file, "w") as f:
        f.write(str(proc.pid))
    print(f"Started stream (PID {proc.pid}) -> {url}")


def stop_stream(pid_file: str = DEFAULT_PID) -> None:
    """
    Stop rpicam-vid by reading PID from pid_file and sending SIGTERM.
    Methods used:
      - open(..., 'r'), os.kill with SIGTERM, os.remove
    Variables created:
      - pid: integer process id
    """
    if not os.path.exists(pid_file):
        print("No stream running (pid file missing).")
        return

    with open(pid_file, "r") as f:
        pid = int(f.read().strip())

    try:
        os.kill(pid, signal.SIGTERM)
        print("Stopped stream.")
    except ProcessLookupError:
        print("Process not found; cleaning pid file.")

    try:
        os.remove(pid_file)
    except FileNotFoundError:
        pass
