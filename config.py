import os
from pathlib import Path
from dotenv import load_dotenv

# Base directory
BASE_DIR = Path(__file__).resolve().parent

# Load .env.local if present, else fallback to .env
env_local = BASE_DIR / ".env.local"
if env_local.exists():
    load_dotenv(dotenv_path=env_local)
else:
    load_dotenv(dotenv_path=BASE_DIR / ".env")


class Config:
    """Application configuration loaded from environment variables."""

    # Firebase Settings
    FIREBASE_API_KEY = os.getenv("FIREBASE_API_KEY", "")
    FIREBASE_AUTH_DOMAIN = os.getenv("FIREBASE_AUTH_DOMAIN", "")
    FIREBASE_PROJECT_ID = os.getenv("FIREBASE_PROJECT_ID", "")
    FIREBASE_STORAGE_BUCKET = os.getenv("FIREBASE_STORAGE_BUCKET", "")
    FIREBASE_MESSAGING_SENDER_ID = os.getenv("FIREBASE_MESSAGING_SENDER_ID", "")
    FIREBASE_APP_ID = os.getenv("FIREBASE_APP_ID", "")
    FIREBASE_DATABASE_URL = os.getenv("FIREBASE_DATABASE_URL", "")
    FIREBASE_COLLECTION = os.getenv("FIREBASE_COLLECTION", "road_detections")
    # Optional path to service account key file (if using firebase-admin)
    FIREBASE_SERVICE_ACCOUNT = os.getenv("FIREBASE_SERVICE_ACCOUNT", "")

    # Cloudinary Settings
    CLOUDINARY_CLOUD_NAME = os.getenv("CLOUDINARY_CLOUD_NAME", "")
    CLOUDINARY_API_KEY = os.getenv("CLOUDINARY_API_KEY", "")
    CLOUDINARY_API_SECRET = os.getenv("CLOUDINARY_API_SECRET", "")
    CLOUDINARY_UPLOAD_PRESET = os.getenv("CLOUDINARY_UPLOAD_PRESET", "ml_default")
    CLOUDINARY_FOLDER = os.getenv("CLOUDINARY_FOLDER", "roadsense/detections")

    # Camera & Detection Settings
    # Camera index (0 for default USB/CSI camera, or path to video file for testing)
    CAMERA_SOURCE = os.getenv("CAMERA_SOURCE", "0")
    FRAME_WIDTH = int(os.getenv("FRAME_WIDTH", "640"))    # 480p width
    FRAME_HEIGHT = int(os.getenv("FRAME_HEIGHT", "480"))  # 480p height
    TARGET_FPS = float(os.getenv("TARGET_FPS", "5.0"))    # Strict 5 FPS target
    CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.45"))

    # Cooldown (in seconds) to avoid spamming alerts for the same detected object across consecutive frames
    DETECTION_COOLDOWN = float(os.getenv("DETECTION_COOLDOWN", "3.0"))

    # GPS Serial Settings
    # Raspberry Pi standard UART is /dev/serial0 (pins 8 & 10) or /dev/ttyUSB0 for USB GPS dongles
    GPS_PORT = os.getenv("GPS_PORT", "/dev/serial0")
    GPS_BAUDRATE = int(os.getenv("GPS_BAUDRATE", "9600"))
    GPS_TIMEOUT = float(os.getenv("GPS_TIMEOUT", "1.0"))

    # Device Metadata
    DEVICE_ID = os.getenv("DEVICE_ID", "rpi4-roadsense-01")

    # Model configuration
    # Can point to directory containing model.ncnn.bin, model.ncnn.param, metadata.yaml or .pt file
    MODEL_PATH = os.getenv("MODEL_PATH", str(BASE_DIR))

    # JPEG Compression Quality (1-100) for in-memory upload (lower saves bandwidth on cellular)
    JPEG_QUALITY = int(os.getenv("JPEG_QUALITY", "85"))

    @classmethod
    def validate(cls):
        """Validates critical credentials."""
        missing = []
        if not cls.CLOUDINARY_CLOUD_NAME:
            missing.append("CLOUDINARY_CLOUD_NAME")
        if not cls.CLOUDINARY_API_KEY:
            missing.append("CLOUDINARY_API_KEY")
        if not cls.CLOUDINARY_API_SECRET:
            missing.append("CLOUDINARY_API_SECRET")
        if not cls.FIREBASE_PROJECT_ID:
            missing.append("FIREBASE_PROJECT_ID")
        if not cls.FIREBASE_API_KEY:
            missing.append("FIREBASE_API_KEY")

        if missing:
            raise ValueError(f"Missing required environment variables in .env.local: {', '.join(missing)}")
