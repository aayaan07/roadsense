import time
import logging
import cv2
from typing import Optional, Tuple
import numpy as np

logger = logging.getLogger("CameraStream")


class CameraStream:
    """
    480p Camera Stream for Raspberry Pi 4 Model B.
    Supports USB Webcams, CSI Pi Cameras (via V4L2/OpenCV or libcamerify), and video files for testing.
    Configured for 640x480 resolution with low-latency buffer settings.
    """

    def __init__(self, source: str = "0", width: int = 640, height: int = 480, target_fps: float = 5.0):
        # Convert numeric string to integer index (e.g. "0" -> 0)
        self.source = int(source) if source.isdigit() else source
        self.width = width
        self.height = height
        self.target_fps = target_fps
        self.cap: Optional[cv2.VideoCapture] = None

    def start(self) -> bool:
        """Initializes and opens the camera capture device."""
        logger.info(f"Opening camera source: {self.source} at {self.width}x{self.height}...")

        # On Linux/Raspberry Pi, cv2.CAP_V4L2 is preferred for low latency
        if isinstance(self.source, int):
            try:
                self.cap = cv2.VideoCapture(self.source, cv2.CAP_V4L2)
            except Exception:
                self.cap = cv2.VideoCapture(self.source)
        else:
            self.cap = cv2.VideoCapture(self.source)

        if not self.cap or not self.cap.isOpened():
            # Fallback to default backend
            self.cap = cv2.VideoCapture(self.source)

        if not self.cap.isOpened():
            logger.error(f"Failed to open video source: {self.source}")
            return False

        # Configure 480p resolution and frame rate
        # Using MJPG FourCC significantly improves frame throughput on Raspberry Pi USB bus
        try:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        except Exception:
            pass

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        # Set buffer size to 1 to eliminate frame lag/latency
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(f"Camera opened successfully. Resolution: {actual_w}x{actual_h}")
        return True

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Reads the most recent frame from the camera.
        Returns: (success_boolean, frame_ndarray)
        """
        if self.cap is None or not self.cap.isOpened():
            return False, None

        ret, frame = self.cap.read()
        if not ret or frame is None:
            return False, None

        # Ensure frame matches requested 480p dimensions if hardware does not support exact 640x480
        h, w = frame.shape[:2]
        if w != self.width or h != self.height:
            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)

        return True, frame

    def release(self):
        """Releases the camera hardware."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            logger.info("Camera device released.")
