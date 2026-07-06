"""
camera_manager.py
------------------
Video acquisition with automatic fallback so the app is always usable:

    1. Basler industrial camera via pypylon   (production line)
    2. Any USB webcam via OpenCV              (dev machine, no Basler)
    3. Synthetic generated frame               (no camera at all)

A background thread continuously grabs frames so the Tkinter UI never
blocks. Call get_frame() from the UI thread for a thread-safe copy of
the latest frame.
"""

import threading
import time
import numpy as np
import cv2


class CameraManager:
    def __init__(self, synthetic_width=1280, synthetic_height=960):
        # Only used to size the synthetic fallback feed. Real Basler/webcam
        # frames keep whatever native resolution the device reports.
        self.synthetic_width = synthetic_width
        self.synthetic_height = synthetic_height

        self.mode = None  # "basler" | "webcam" | "synthetic"
        self._pylon = None
        self._camera = None
        self._converter = None
        self._cap = None

        self._lock = threading.Lock()
        self._latest_frame = None
        self._running = False
        self._thread = None

        self._init_camera()

    # ------------------------------------------------------------ setup
    def _init_camera(self):
        if self._try_basler():
            self.mode = "basler"
        elif self._try_webcam():
            self.mode = "webcam"
        else:
            self.mode = "synthetic"
        print(f"[CameraManager] mode = {self.mode}")

    def _try_basler(self):
        try:
            from pypylon import pylon
            self._pylon = pylon
            tl_factory = pylon.TlFactory.GetInstance()
            devices = tl_factory.EnumerateDevices()
            if not devices:
                return False
            self._camera = pylon.InstantCamera(tl_factory.CreateFirstDevice())
            self._camera.Open()
            self._camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
            self._converter = pylon.ImageFormatConverter()
            self._converter.OutputPixelFormat = pylon.PixelType_BGR8packed
            self._converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
            return True
        except Exception as e:
            print(f"[CameraManager] Basler not available: {e}")
            return False

    def _try_webcam(self):
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                return False
            self._cap = cap
            return True
        except Exception as e:
            print(f"[CameraManager] Webcam not available: {e}")
            return False

    def reconnect(self):
        """Re-scan for a camera without restarting the app (e.g. Basler was
        just plugged in)."""
        was_running = self._running
        if was_running:
            self.stop()
        self._camera = None
        self._cap = None
        self._init_camera()
        if was_running:
            self.start()

    # ------------------------------------------------------------- loop
    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._grab_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        if self.mode == "basler" and self._camera is not None:
            try:
                self._camera.StopGrabbing()
                self._camera.Close()
            except Exception:
                pass
        if self.mode == "webcam" and self._cap is not None:
            self._cap.release()

    def _grab_loop(self):
        t0 = time.time()
        while self._running:
            frame = None
            try:
                if self.mode == "basler":
                    frame = self._grab_basler()
                elif self.mode == "webcam":
                    ok, f = self._cap.read()
                    frame = f if ok else None
                else:
                    frame = self._grab_synthetic(t0)
            except Exception as e:
                print(f"[CameraManager] grab error: {e}")

            if frame is not None:
                with self._lock:
                    self._latest_frame = frame
            time.sleep(0.01)

    def _grab_basler(self):
        grab = self._camera.RetrieveResult(1000, self._pylon.TimeoutHandling_Return)
        try:
            if grab.GrabSucceeded():
                image = self._converter.Convert(grab)
                return image.GetArray()
            return None
        finally:
            grab.Release()

    def _grab_synthetic(self, t0):
        h, w = self.synthetic_height, self.synthetic_width
        frame = np.full((h, w, 3), 35, dtype=np.uint8)
        cx, cy = w // 2, h // 2
        cv2.circle(frame, (cx, cy), min(h, w) // 3, (95, 95, 95), -1)
        t = time.time() - t0
        for i in range(3):
            bx = int(cx + 55 * np.sin(t * 0.4 + i * 2.1))
            by = int(cy + 55 * np.cos(t * 0.3 + i * 2.1))
            cv2.circle(frame, (bx, by), 3 + i, (225, 225, 225), -1)
        cv2.putText(frame, "SYNTHETIC FEED - no camera detected", (20, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1, cv2.LINE_AA)
        return frame

    # ----------------------------------------------------------- access
    def get_frame(self):
        with self._lock:
            return None if self._latest_frame is None else self._latest_frame.copy()

    def status_text(self):
        return {
            "basler": "\U0001F7E2 Basler camera",
            "webcam": "\U0001F7E1 Webcam (fallback)",
            "synthetic": "\U0001F534 No camera - synthetic feed",
        }.get(self.mode, "Unknown")
