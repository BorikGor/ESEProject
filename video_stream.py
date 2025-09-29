"""
video_stream.py
---------------
Start rpicam-vid to stream MJPEG over UDP (localhost:5800) and then
consume the stream with OpenCV/FFmpeg. The script ensures proper
startup delay, checks capture readiness, and cleans up the child
process on exit so the camera is not left busy.

Requires:
- Raspberry Pi OS (Bookworm/Bullseye with libcamera stack)
- rpicam-vid (bookworm replacement for libcamera-vid)
- OpenCV built with FFmpeg support

Notes:
- Use '-n' (no preview) for headless setups.
- If the camera is "in use", kill leftover rpicam-vid processes or
  reboot the Pi.
"""

import os
import signal
import subprocess
import sys
import time
import cv2
import numpy as np


def start_stream():
    """
    Start rpicam-vid as a child process that streams MJPEG frames to
    UDP at 127.0.0.1:5800. The function returns the Popen handle.
    Methods used:
      - subprocess.Popen to launch the streamer
    Creates:
      - A child process 'rpicam-vid'
    """
    # Build rpicam-vid command with no preview and reasonable defaults.
    cmd = [
        "rpicam-vid",
        "-t", "0",                 # endless stream
        "-n",                      # no preview (headless)
        "--codec", "mjpeg",        # MJPEG for easy OpenCV decoding
        "--width", "1280",         # adjust to your needs
        "--height", "720",
        "--framerate", "20",
        "-o", "udp://127.0.0.1:5800"
    ]

    # Start the process in its own group; we will kill the group later.
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid
    )
    return proc


def stop_stream(proc):
    """
    Stop the rpicam-vid child process and its group safely.
    Methods used:
      - os.killpg to terminate the process group
    Variables:
      - proc: Popen handle returned by start_stream()
    """
    if proc and proc.poll() is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            # Give it a moment to exit gracefully
            time.sleep(0.3)
            if proc.poll() is None:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception as exc:
            print(f"Warning: failed to stop rpicam-vid cleanly: {exc}")


def open_capture():
    """
    Open the UDP capture via OpenCV + FFmpeg. Returns the VideoCapture
    object. Methods used:
      - cv2.VideoCapture with CAP_FFMPEG backend
    Creates:
      - cap: the capture object bound to udp:// localhost:5800
    """
    url = "udp://127.0.0.1:5800?overrun_nonfatal=1&fifo_size=50000000"
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    return cap


def main():
    """
    Main entry:
      1) Start rpicam-vid.
      2) Wait a bit for camera to come online.
      3) Open OpenCV capture; verify it is opened.
      4) Try to read a frame; print diagnostics.
      5) Clean up on exit (Ctrl+C safe).
    Methods used:
      - time.sleep, cap.isOpened, cap.read
    Variables created:
      - proc: Popen handle of rpicam-vid
      - cap: OpenCV VideoCapture
    """
    print("Starting rpicam-vid; waiting 2 s for camera to come online...")
    proc = start_stream()
    try:
        # Optional: read and print first few lines of rpicam-vid output
        time.sleep(0.2)
        for _ in range(6):
            line = proc.stdout.readline()
            if not line:
                break
            print(line.decode(errors="ignore").rstrip())

        time.sleep(2.0)  # give the camera/stream time to start

        cap = open_capture()
        if not cap.isOpened():
            print("ERROR: OpenCV could not open UDP stream.")
            print("Check if rpicam-vid is running and camera is free.")
            sys.exit(1)

        print("OpenCV capture is opened. Trying to grab one frame...")
        ok, frame = cap.read()
        if not ok or frame is None:
            print("ERROR: Failed to read a frame from the UDP stream.")
            print("If this persists, verify no other process owns the "
                  "camera and that rpicam-vid is not printing errors.")
            sys.exit(2)

        h, w = frame.shape[:2]
        print(f"Got one frame: {w}x{h}, dtype={frame.dtype}")

        # Minimal preview loop (press 'q' to quit)
        print("Press 'q' in the window to quit.")
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("Stream ended or frame not available.")
                break
            cv2.imshow("Over UDP (MJPEG)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    finally:
        stop_stream(proc)
        print("rpicam-vid stopped; exiting.")


if __name__ == "__main__":
    main()
