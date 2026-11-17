import time
import logging
import cv2
from typing import Optional, Tuple
import numpy as np

# Check for Picamera2 (standard on Raspberry Pi OS Bookworm for Pi 4 & Pi 5)
try:
    from picamera2 import Picamera2
    HAS_PICAMERA2 = True
except ImportError:
    HAS_PICAMERA2 = False

logger = logging.getLogger("CameraStream")


class CameraStream:
    """
    480p Camera Stream for Raspberry Pi 4 Model B and Raspberry Pi 5.
    Supports:
    - USB Webcams (via OpenCV VideoCapture)
    - Raspberry Pi CSI Cameras (via Picamera2 or V4L2/OpenCV)
    - Video files or test streams
    Configured for 640x480 resolution with low-latency buffer settings.
    """

    def __init__(self, source: str = "0", width: int = 640, height: int = 480, target_fps: float = 5.0):
        # Convert numeric string to integer index (e.g. "0" -> 0)
        self.source = int(source) if source.isdigit() else source
        self.width = width
        self.height = height
        self.target_fps = target_fps
        self.cap: Optional[cv2.VideoCapture] = None
        self.picam2 = None
        self.is_picamera2 = False

    def start(self) -> bool:
        """Initializes and opens the camera capture device."""
        logger.info(f"Opening camera source: {self.source} at {self.width}x{self.height}...")

        # 1. Attempt standard OpenCV capture (ideal for USB cameras & standard V4L2).
        #    Opening a device is not enough: on Pi OS Bookworm a CSI camera exposes
        #    /dev/video* nodes that open fine but never deliver frames, so verify a read.
        if self._open_opencv():
            if self._verify_opencv_frames():
                return True
            logger.warning(
                f"OpenCV opened source {self.source} but returned no frames "
                "(typical for CSI cameras); releasing and trying Picamera2..."
            )
            self._release_opencv()

        # 2. Fall back to Picamera2 (CSI cameras on Pi 4/5) for any camera index
        if isinstance(self.source, int):
            if not HAS_PICAMERA2:
                logger.warning(
                    "Picamera2 is not importable in this Python environment. If you use a CSI "
                    "camera, create the venv with --system-site-packages "
                    "(python3 -m venv --system-site-packages venv) and run: "
                    "sudo apt install -y python3-picamera2"
                )
            else:
                logger.info("Attempting Picamera2 for CSI camera...")
                if self._open_picamera2():
                    return True

        logger.error(f"Failed to open video source: {self.source}")
        return False

    def _verify_opencv_frames(self, attempts: int = 10) -> bool:
        """Returns True if the already-opened OpenCV capture actually yields a frame."""
        for _ in range(attempts):
            ret, frame = self.cap.read()
            if ret and frame is not None and frame.size > 0:
                return True
            time.sleep(0.1)
        return False

    def _release_opencv(self):
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

    def _open_opencv(self) -> bool:
        """Attempts to open camera via OpenCV."""
        if isinstance(self.source, int):
            try:
                self.cap = cv2.VideoCapture(self.source, cv2.CAP_V4L2)
            except Exception:
                self.cap = cv2.VideoCapture(self.source)
        else:
            self.cap = cv2.VideoCapture(self.source)

        if not self.cap or not self.cap.isOpened():
            self._release_opencv()
            self.cap = cv2.VideoCapture(self.source)

        if not self.cap.isOpened():
            self._release_opencv()
            return False

        # Configure 480p resolution and frame rate
        try:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        except Exception:
            pass

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(f"Camera opened via OpenCV. Resolution: {actual_w}x{actual_h}")
        return True

    def _open_picamera2(self) -> bool:
        """Attempts to open CSI camera via Picamera2 (official Raspberry Pi 4 / 5 library)."""
        try:
            self.picam2 = Picamera2()
            cam_config = self.picam2.create_preview_configuration(
                main={"size": (self.width, self.height), "format": "RGB888"}
            )
            self.picam2.configure(cam_config)
            self.picam2.start()
            self.is_picamera2 = True
            logger.info(f"Camera opened via Picamera2. Resolution: {self.width}x{self.height}")
            return True
        except Exception as e:
            logger.warning(f"Picamera2 initialization failed: {e}")
            if self.picam2:
                try:
                    self.picam2.close()
                except Exception:
                    pass
                self.picam2 = None
            return False

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Reads the most recent frame from the camera.
        Returns: (success_boolean, frame_ndarray in BGR format)
        """
        # Picamera2 backend
        if self.is_picamera2 and self.picam2 is not None:
            try:
                rgb_frame = self.picam2.capture_array()
                if rgb_frame is None or rgb_frame.size == 0:
                    return False, None
                # Picamera2's "RGB888" format is laid out B,G,R in memory, i.e. already
                # OpenCV's BGR order, so no colour conversion is needed.
                if rgb_frame.ndim == 3 and rgb_frame.shape[2] == 4:
                    rgb_frame = rgb_frame[:, :, :3]
                return True, np.ascontiguousarray(rgb_frame)
            except Exception as e:
                logger.error(f"Error capturing frame via Picamera2: {e}")
                return False, None

        # OpenCV backend
        if self.cap is None or not self.cap.isOpened():
            return False, None

        ret, frame = self.cap.read()
        if not ret or frame is None:
            return False, None

        h, w = frame.shape[:2]
        if w != self.width or h != self.height:
            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)

        return True, frame

    def release(self):
        """Releases the camera hardware."""
        if self.is_picamera2 and self.picam2 is not None:
            try:
                self.picam2.stop()
                self.picam2.close()
            except Exception:
                pass
            self.picam2 = None
            self.is_picamera2 = False
            logger.info("Picamera2 device released.")

        if self.cap is not None:
            self.cap.release()
            self.cap = None
            logger.info("OpenCV camera device released.")
