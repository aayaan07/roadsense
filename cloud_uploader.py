import io
import time
import queue
import logging
import threading
from datetime import datetime, timezone
from typing import Dict, Any, Optional

import requests
try:
    import cloudinary
    import cloudinary.uploader
    HAS_CLOUDINARY = True
except ImportError:
    cloudinary = None
    HAS_CLOUDINARY = False

from config import Config

logger = logging.getLogger("CloudUploader")


def dict_to_firestore_fields(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Converts a standard Python dictionary to Google Cloud Firestore REST API field format.
    Supports GeoPoint, timestamps, integers, floats, booleans, and nested structures.
    """
    fields = {}
    for key, value in data.items():
        if value is None:
            fields[key] = {"nullValue": None}
        elif isinstance(value, bool):
            fields[key] = {"booleanValue": value}
        elif isinstance(value, int):
            fields[key] = {"integerValue": str(value)}
        elif isinstance(value, float):
            fields[key] = {"doubleValue": value}
        elif isinstance(value, str):
            # Check for ISO8601 UTC timestamp format
            if value.endswith("Z") and ("T" in value):
                fields[key] = {"timestampValue": value}
            else:
                fields[key] = {"stringValue": value}
        elif isinstance(value, dict):
            # Check if dict represents a GeoPoint with latitude & longitude
            if "latitude" in value and "longitude" in value and len(value) == 2:
                lat = value["latitude"]
                lon = value["longitude"]
                if lat is not None and lon is not None:
                    fields[key] = {
                        "geoPointValue": {
                            "latitude": float(lat),
                            "longitude": float(lon)
                        }
                    }
                    continue
            fields[key] = {
                "mapValue": {
                    "fields": dict_to_firestore_fields(value)
                }
            }
        elif isinstance(value, (list, tuple)):
            fields[key] = {
                "arrayValue": {
                    "values": [
                        dict_to_firestore_fields({"_": item})["_"]
                        for item in value
                    ]
                }
            }
        else:
            fields[key] = {"stringValue": str(value)}
    return fields


class AsyncUploader:
    """
    Asynchronous in-memory uploader for Raspberry Pi.
    - Uploads images directly from RAM to Cloudinary (zero disk writes).
    - Writes alert records (Cloudinary URL, GPS coords, confidence, label) to Firebase Firestore.
    - Decoupled from detection loop via an in-memory Queue to maintain strict 5 FPS.
    """

    def __init__(self, config: Config, max_queue_size: int = 50):
        self.config = config
        self.queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self.running = False
        self.worker_thread: Optional[threading.Thread] = None

        # Configure Cloudinary
        if HAS_CLOUDINARY:
            cloudinary.config(
                cloud_name=config.CLOUDINARY_CLOUD_NAME,
                api_key=config.CLOUDINARY_API_KEY,
                api_secret=config.CLOUDINARY_API_SECRET,
                secure=True
            )
        else:
            logger.warning("'cloudinary' package not installed. Run: pip install -r requirements.txt")

        # Pre-build Firestore REST API endpoint
        self.firestore_url = (
            f"https://firestore.googleapis.com/v1/projects/{config.FIREBASE_PROJECT_ID}/"
            f"databases/(default)/documents/{config.FIREBASE_COLLECTION}?key={config.FIREBASE_API_KEY}"
        )

        logger.info("CloudUploader initialized with Firestore REST API and Cloudinary SDK.")

    def start(self):
        """Starts the background worker thread."""
        self.running = True
        self.worker_thread = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name="UploadWorkerThread"
        )
        self.worker_thread.start()
        logger.info("Asynchronous Upload Worker thread started.")

    def stop(self, timeout: float = 5.0):
        """Signals the worker to stop and waits for queue completion."""
        self.running = False
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=timeout)
        logger.info("Asynchronous Upload Worker stopped.")

    def enqueue(self, jpeg_bytes: bytes, detections: list, gps_data: dict):
        """
        Enqueues an alert for asynchronous upload.
        Non-blocking: drops oldest item if queue is full during network outages to protect RAM.
        """
        payload = {
            "image_bytes": jpeg_bytes,
            "detections": detections,
            "gps": gps_data,
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "enqueued_at": time.time()
        }

        try:
            self.queue.put_nowait(payload)
            logger.debug(f"Alert enqueued. Current queue size: {self.queue.qsize()}")
        except queue.Full:
            try:
                # Discard oldest to prioritize fresh real-time alerts
                _ = self.queue.get_nowait()
                self.queue.put_nowait(payload)
                logger.warning("Upload queue full (slow/lost network). Discarded oldest alert to save RAM.")
            except queue.Empty:
                pass

    def _worker_loop(self):
        """Background worker pulling from queue and uploading."""
        while self.running:
            try:
                item = self.queue.get(timeout=1.0)
            except queue.Empty:
                continue

            try:
                self._process_upload(item)
            except Exception as e:
                logger.error(f"Failed to process upload for alert: {e}", exc_info=True)
            finally:
                self.queue.task_done()

    def _upload_to_cloudinary(self, image_bytes: bytes) -> Optional[Dict[str, Any]]:
        """
        Streams image directly from RAM buffer into Cloudinary.
        NO DISK STORAGE IS USED.
        """
        if not HAS_CLOUDINARY:
            logger.error("Cloudinary SDK is missing. Cannot upload image from RAM.")
            return None

        try:
            # Wrap bytes in BytesIO buffer in memory
            buffer = io.BytesIO(image_bytes)
            buffer.seek(0)

            # Upload options
            upload_kwargs = {
                "folder": self.config.CLOUDINARY_FOLDER,
                "resource_type": "image"
            }
            if self.config.CLOUDINARY_UPLOAD_PRESET:
                upload_kwargs["upload_preset"] = self.config.CLOUDINARY_UPLOAD_PRESET

            response = cloudinary.uploader.upload(buffer, **upload_kwargs)
            logger.info(f"Cloudinary upload success: {response.get('secure_url')}")
            return response
        except Exception as e:
            logger.error(f"Cloudinary in-memory upload error: {e}")
            return None

    def _write_to_firestore(self, doc_data: Dict[str, Any]) -> bool:
        """
        Writes alert document directly to Firebase Firestore using REST API.
        Does not require Google Cloud service account JSON files on the Raspberry Pi.
        """
        try:
            fields = dict_to_firestore_fields(doc_data)
            payload = {"fields": fields}

            response = requests.post(
                self.firestore_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=10.0
            )

            if response.status_code in [200, 201]:
                doc_name = response.json().get("name", "unknown")
                logger.info(f"Firestore document successfully created: {doc_name}")
                return True
            else:
                logger.error(f"Firestore REST API failed with status {response.status_code}: {response.text}")
                return False
        except Exception as e:
            logger.error(f"Firestore document write exception: {e}")
            return False

    def _process_upload(self, item: Dict[str, Any]):
        """Uploads image to Cloudinary then commits document to Firestore."""
        start_time = time.time()
        image_bytes = item["image_bytes"]
        detections = item["detections"]
        gps = item["gps"]
        timestamp = item["timestamp"]

        # 1. Stream image from memory to Cloudinary
        cloudinary_resp = self._upload_to_cloudinary(image_bytes)
        if not cloudinary_resp:
            logger.warning("Skipping Firestore write due to Cloudinary upload failure.")
            return

        image_url = cloudinary_resp.get("secure_url")
        public_id = cloudinary_resp.get("public_id")

        # 2. Extract detection summary
        top_confidence = max([d["confidence"] for d in detections]) if detections else 0.0
        primary_label = detections[0]["label"] if detections else "unknown"

        # 3. Construct Firestore document
        doc_data = {
            "device_id": self.config.DEVICE_ID,
            "detected_class": primary_label,
            "confidence": round(top_confidence, 4),
            "detection_count": len(detections),
            "detections": detections,
            "image_url": image_url,
            "cloudinary_public_id": public_id,
            "timestamp": timestamp,
            "status": "reported",
            "gps_fixed": bool(gps.get("has_fix", False)),
            "speed_kmh": float(gps.get("speed_kmh") or 0.0),
            "altitude_m": float(gps.get("altitude_m") or 0.0),
            "satellites": int(gps.get("satellites") or 0)
        }

        # Add GeoPoint and coordinate values if GPS fix is available
        lat = gps.get("latitude")
        lon = gps.get("longitude")
        if lat is not None and lon is not None:
            doc_data["location"] = {"latitude": lat, "longitude": lon}
            doc_data["latitude"] = lat
            doc_data["longitude"] = lon
        else:
            doc_data["latitude"] = 0.0
            doc_data["longitude"] = 0.0

        # 4. Commit to Firestore
        success = self._write_to_firestore(doc_data)
        elapsed = time.time() - start_time
        if success:
            logger.info(f"Complete alert workflow finished in {elapsed:.2f}s for '{primary_label}'")
