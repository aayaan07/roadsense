import os
import cv2
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np

logger = logging.getLogger("Detector")


class ObjectDetector:
    """
    Object Detector tailored for Raspberry Pi 4 Model B.
    Primary backend: Ultralytics YOLO with NCNN acceleration for YOLOv8 (pothole detection).
    Also supports standard .pt, .onnx, or direct NCNN directories.
    """

    def __init__(self, model_path: str, conf_threshold: float = 0.45, imgsz: int = 320):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.imgsz = imgsz
        self.model = None
        self.class_names = {0: "pothole"}
        self._load_model()

    def _load_model(self):
        """Loads the detection model via Ultralytics."""
        try:
            from ultralytics import YOLO

            resolved_path = Path(self.model_path).resolve()
            
            # Check if directory contains NCNN files or if path is a file
            if resolved_path.is_dir():
                # If pointing to directory with model.ncnn.bin
                if (resolved_path / "model.ncnn.bin").exists():
                    logger.info(f"Loading YOLO NCNN model from directory: {resolved_path}")
                    self.model = YOLO(str(resolved_path), task="detect")
                else:
                    # Look for first .pt, .onnx, or .bin in dir
                    files = list(resolved_path.glob("*.pt")) + list(resolved_path.glob("*.onnx"))
                    if files:
                        logger.info(f"Loading model: {files[0]}")
                        self.model = YOLO(str(files[0]), task="detect")
                    else:
                        logger.info(f"Attempting generic YOLO load for {resolved_path}")
                        self.model = YOLO(str(resolved_path), task="detect")
            else:
                logger.info(f"Loading YOLO model from file: {resolved_path}")
                self.model = YOLO(str(resolved_path), task="detect")

            if hasattr(self.model, "names") and self.model.names:
                self.class_names = self.model.names
            logger.info(f"Model loaded successfully. Classes: {self.class_names}")

        except ImportError:
            logger.error(
                "Ultralytics is not installed. Install via: pip install ultralytics. "
                "Detector will operate in mock mode."
            )
            self.model = None
        except Exception as e:
            logger.error(f"Error loading model from {self.model_path}: {e}")
            self.model = None

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Runs object detection on a single frame.
        Returns a list of detected objects:
        [
            {
                "label": "pothole",
                "confidence": 0.82,
                "bbox": [x1, y1, x2, y2]
            },
            ...
        ]
        """
        if self.model is None or frame is None:
            return []

        detections = []
        try:
            # Predict with Ultralytics
            # imgsz=320 is optimal for Pi 4 NCNN inference speed
            results = self.model.predict(
                source=frame,
                imgsz=self.imgsz,
                conf=self.conf_threshold,
                verbose=False
            )

            if not results or len(results) == 0:
                return []

            r = results[0]
            boxes = r.boxes
            if boxes is None or len(boxes) == 0:
                return []

            for box in boxes:
                conf = float(box.conf[0].item())
                cls_id = int(box.cls[0].item())
                xyxy = [int(v) for v in box.xyxy[0].tolist()]
                label = self.class_names.get(cls_id, f"class_{cls_id}")

                detections.append({
                    "label": label,
                    "confidence": round(conf, 4),
                    "bbox": xyxy
                })

        except Exception as e:
            logger.error(f"Inference error during detect(): {e}")

        return detections

    def annotate_frame(
        self,
        frame: np.ndarray,
        detections: List[Dict[str, Any]],
        gps_data: Dict[str, Any]
    ) -> np.ndarray:
        """
        Draws bounding boxes, confidence badges, and telemetry watermarks onto the frame in memory.
        Does NOT write to disk.
        """
        annotated = frame.copy()
        h, w = annotated.shape[:2]

        # Draw detections
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            label = det["label"]
            conf = det["confidence"]

            # Box styling
            color = (0, 0, 255)  # Bright Red in BGR
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            caption = f"{label.upper()} {int(conf * 100)}%"
            (text_w, text_h), baseline = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            # Label banner
            banner_y1 = max(0, y1 - text_h - 6)
            banner_y2 = y1
            cv2.rectangle(annotated, (x1, banner_y1), (x1 + text_w + 6, banner_y2), color, -1)
            cv2.putText(
                annotated,
                caption,
                (x1 + 3, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

        # Draw Telemetry Banner at bottom of frame
        overlay_h = 24
        cv2.rectangle(annotated, (0, h - overlay_h), (w, h), (20, 20, 20), -1)

        lat = gps_data.get("latitude")
        lon = gps_data.get("longitude")
        speed = gps_data.get("speed_kmh", 0.0)
        has_fix = gps_data.get("has_fix", False)

        if has_fix and lat is not None and lon is not None:
            telemetry_str = f"GPS: {lat:.5f}, {lon:.5f} | Spd: {speed:.1f} km/h"
        else:
            telemetry_str = "GPS: Searching Fix... | Spd: 0.0 km/h"

        cv2.putText(
            annotated,
            telemetry_str,
            (8, h - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 255, 0) if has_fix else (0, 165, 255),
            1,
            cv2.LINE_AA
        )

        return annotated

    @staticmethod
    def encode_to_jpeg_bytes(frame: np.ndarray, quality: int = 85) -> bytes:
        """
        Encodes an OpenCV image to JPEG bytes in RAM.
        NEVER saves to disk.
        """
        success, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not success:
            raise RuntimeError("Failed to encode frame to in-memory JPEG bytes.")
        return encoded.tobytes()
