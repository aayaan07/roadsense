import os
import cv2
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np

logger = logging.getLogger("Detector")


class ObjectDetector:
    """
    ONNX Runtime object detector for Raspberry Pi 4 Model B (YOLOv8-style pothole model).
    Does not depend on ultralytics/torch.
    """

    def __init__(self, model_path: str, conf_threshold: float = 0.45, imgsz: int = 320,
                 iou_threshold: float = 0.45):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz
        self.session = None
        self.input_name = None
        self.class_names = {0: "pothole"}
        self._load_model()

    @staticmethod
    def _resolve_onnx(path: str) -> Path:
        p = Path(path).resolve()
        if p.is_dir():
            files = sorted(p.glob("*.onnx"))
            if not files:
                raise FileNotFoundError(f"No .onnx model found in {p}")
            return files[0]
        if not p.exists():
            raise FileNotFoundError(f"Model file not found: {p}")
        return p

    def _load_model(self):
        """Loads the ONNX model via onnxruntime."""
        try:
            import onnxruntime as ort

            onnx_path = self._resolve_onnx(self.model_path)
            logger.info(f"Loading ONNX model: {onnx_path}")

            opts = ort.SessionOptions()
            opts.intra_op_num_threads = os.cpu_count() or 4
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.session = ort.InferenceSession(
                str(onnx_path), sess_options=opts, providers=["CPUExecutionProvider"]
            )

            inp = self.session.get_inputs()[0]
            self.input_name = inp.name
            # Use the model's fixed input size if it has one
            h, w = inp.shape[2], inp.shape[3]
            if isinstance(h, int) and isinstance(w, int):
                self.input_hw = (h, w)
            else:
                self.input_hw = (self.imgsz, self.imgsz)

            # Class names from model metadata (Ultralytics exports embed "names")
            meta = self.session.get_modelmeta().custom_metadata_map or {}
            if "names" in meta:
                try:
                    import ast
                    names = ast.literal_eval(meta["names"])
                    if isinstance(names, dict):
                        self.class_names = {int(k): v for k, v in names.items()}
                except Exception:
                    pass
            logger.info(f"Model loaded. Input: {self.input_hw}, Classes: {self.class_names}")

        except ImportError:
            logger.error(
                "onnxruntime is not installed. Install via: pip install onnxruntime. "
                "Detector will operate in mock mode."
            )
            self.session = None
        except Exception as e:
            logger.error(f"Error loading model from {self.model_path}: {e}")
            self.session = None

    def _preprocess(self, frame: np.ndarray):
        """Letterbox resize to model input, BGR->RGB, NCHW float32 in [0,1]."""
        in_h, in_w = self.input_hw
        h, w = frame.shape[:2]
        scale = min(in_w / w, in_h / h)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        pad_x, pad_y = (in_w - new_w) / 2, (in_h - new_h) / 2
        top, left = int(round(pad_y - 0.1)), int(round(pad_x - 0.1))
        canvas = cv2.copyMakeBorder(
            resized, top, in_h - new_h - top, left, in_w - new_w - left,
            cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )
        blob = canvas[:, :, ::-1].transpose(2, 0, 1)
        blob = np.ascontiguousarray(blob[None], dtype=np.float32) / 255.0
        return blob, scale, left, top

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Runs object detection on a single frame.
        Returns: [{"label": "pothole", "confidence": 0.82, "bbox": [x1, y1, x2, y2]}, ...]
        """
        if self.session is None or frame is None:
            return []

        try:
            h, w = frame.shape[:2]
            blob, scale, pad_x, pad_y = self._preprocess(frame)
            out = self.session.run(None, {self.input_name: blob})[0]
            out = np.squeeze(out, axis=0)  # drop batch dim

            boxes, scores, cls_ids = [], [], []

            if out.ndim == 2 and out.shape[1] == 6 and out.shape[0] != 6:
                # End-to-end export (NMS baked in): [x1, y1, x2, y2, conf, cls]
                for x1, y1, x2, y2, conf, cls in out:
                    if conf < self.conf_threshold:
                        continue
                    boxes.append([x1, y1, x2, y2])
                    scores.append(float(conf))
                    cls_ids.append(int(cls))
                keep = list(range(len(boxes)))
            else:
                # Standard YOLOv8: (4+nc, N) -> (N, 4+nc)
                if out.shape[0] < out.shape[1]:
                    out = out.T
                cls_scores = out[:, 4:]
                cls_idx = np.argmax(cls_scores, axis=1)
                conf = cls_scores[np.arange(len(cls_idx)), cls_idx]
                mask = conf >= self.conf_threshold
                out, conf, cls_idx = out[mask], conf[mask], cls_idx[mask]
                if len(out) == 0:
                    return []
                cx, cy, bw, bh = out[:, 0], out[:, 1], out[:, 2], out[:, 3]
                xyxy = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)
                boxes, scores, cls_ids = xyxy.tolist(), conf.tolist(), cls_idx.tolist()
                nms_boxes = [[b[0], b[1], b[2] - b[0], b[3] - b[1]] for b in boxes]
                keep = np.array(cv2.dnn.NMSBoxes(
                    nms_boxes, scores, self.conf_threshold, self.iou_threshold
                )).flatten().tolist()

            detections = []
            for i in keep:
                x1, y1, x2, y2 = boxes[i]
                # Map back to original frame coordinates
                x1 = int(np.clip((x1 - pad_x) / scale, 0, w - 1))
                x2 = int(np.clip((x2 - pad_x) / scale, 0, w - 1))
                y1 = int(np.clip((y1 - pad_y) / scale, 0, h - 1))
                y2 = int(np.clip((y2 - pad_y) / scale, 0, h - 1))
                cls_id = int(cls_ids[i])
                detections.append({
                    "label": self.class_names.get(cls_id, f"class_{cls_id}"),
                    "confidence": round(float(scores[i]), 4),
                    "bbox": [x1, y1, x2, y2],
                })
            return detections

        except Exception as e:
            logger.error(f"Inference error during detect(): {e}")
            return []

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

        if gps_data.get("raw_status") == "disabled":
            telemetry_str = "GPS: Disabled"
        elif has_fix and lat is not None and lon is not None:
            telemetry_str = f"GPS: {lat:.5f}, {lon:.5f} | Spd: {speed:.1f} km/h"
        else:
            telemetry_str = "GPS: Searching Fix... | Spd: 0.0 km/h"

        cv2.putText(
            annotated,
            telemetry_str,
            (8, h - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 255, 0) if has_fix else ((180, 180, 180) if gps_data.get("raw_status") == "disabled" else (0, 165, 255)),
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
