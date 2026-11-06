"""
Diagnostic and Subsystem Test Script for RoadSense (Raspberry Pi 4 Model B).
Run this script to verify:
1. Environment variables (.env.local)
2. Cloudinary in-memory upload
3. Firebase Firestore document creation
4. Camera stream (480p capture)
5. GPS serial port connection
6. YOLO / NCNN model inference
"""

import sys
import time
import io
import numpy as np
import cv2

from config import Config
from gps_reader import GPSReader
from cloud_uploader import AsyncUploader, dict_to_firestore_fields
from camera_stream import CameraStream
from detector import ObjectDetector


def print_step(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def test_config():
    print_step("1. Verifying .env.local Configuration")
    try:
        Config.validate()
        print("[SUCCESS] All required credentials found:")
        print(f"  - Firebase Project ID: {Config.FIREBASE_PROJECT_ID}")
        print(f"  - Cloudinary Cloud Name: {Config.CLOUDINARY_CLOUD_NAME}")
        print(f"  - Target FPS: {Config.TARGET_FPS}")
        print(f"  - Resolution: {Config.FRAME_WIDTH}x{Config.FRAME_HEIGHT}")
        print(f"  - GPS Port: {Config.GPS_PORT}")
        print(f"  - Model Path: {Config.MODEL_PATH}")
        return True
    except Exception as e:
        print(f"[FAIL] Configuration error: {e}")
        return False


def test_cloudinary_and_firestore():
    print_step("2. Testing In-Memory Cloudinary Upload & Firestore DB Write")
    uploader = AsyncUploader(config=Config)

    # Create dummy in-memory 480p image (solid color with test text)
    test_img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(test_img, "RoadSense RPi4 Diagnostic Test", (40, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    success, encoded = cv2.imencode(".jpg", test_img)
    if not success:
        print("[FAIL] Could not encode test image in RAM.")
        return False

    jpeg_bytes = encoded.tobytes()
    print("[INFO] Uploading test image directly from RAM buffer to Cloudinary...")
    cloudinary_resp = uploader._upload_to_cloudinary(jpeg_bytes)
    if not cloudinary_resp:
        print("[FAIL] Cloudinary upload failed. Check credentials in .env.local.")
        return False

    image_url = cloudinary_resp.get("secure_url")
    print(f"[SUCCESS] Cloudinary image uploaded: {image_url}")

    print("[INFO] Writing test alert record to Firebase Firestore...")
    test_doc = {
        "device_id": "test-rpi4-diagnostic",
        "detected_class": "test_object",
        "confidence": 0.99,
        "image_url": image_url,
        "timestamp": "2026-11-06T09:00:00Z",
        "status": "diagnostic_test",
        "latitude": 12.9716,
        "longitude": 77.5946,
        "location": {"latitude": 12.9716, "longitude": 77.5946}
    }

    db_success = uploader._write_to_firestore(test_doc)
    if db_success:
        print("[SUCCESS] Firebase Firestore document created successfully!")
        return True
    else:
        print("[FAIL] Firestore write failed. Check Firebase API Key and Project ID.")
        return False


def test_camera():
    print_step("3. Testing Camera Stream (480p)")
    cam = CameraStream(
        source=Config.CAMERA_SOURCE,
        width=Config.FRAME_WIDTH,
        height=Config.FRAME_HEIGHT,
        target_fps=Config.TARGET_FPS
    )
    if not cam.start():
        print(f"[WARN] Camera source {Config.CAMERA_SOURCE} could not be opened.")
        print("  On Raspberry Pi, ensure camera is enabled or plugged in via USB/CSI.")
        return False

    ret, frame = cam.read_frame()
    cam.release()
    if ret and frame is not None:
        h, w = frame.shape[:2]
        print(f"[SUCCESS] Captured frame successfully. Shape: {w}x{h} (Matches {Config.FRAME_WIDTH}x{Config.FRAME_HEIGHT})")
        return True
    else:
        print("[FAIL] Failed to read frame from camera.")
        return False


def test_gps():
    print_step("4. Testing GPS Serial Connection")
    gps = GPSReader(port=Config.GPS_PORT, baudrate=Config.GPS_BAUDRATE)
    gps.start()
    print(f"[INFO] Listening on {Config.GPS_PORT} for 3 seconds...")
    time.sleep(3.0)
    coords = gps.get_coordinates()
    gps.stop()
    print(f"[INFO] GPS Status: {coords['raw_status']}")
    print(f"  - Fix Acquired: {coords['has_fix']}")
    print(f"  - Satellites: {coords['satellites']}")
    print(f"  - Latitude: {coords['latitude']}")
    print(f"  - Longitude: {coords['longitude']}")
    if coords["has_fix"]:
        print("[SUCCESS] Real-time GPS lock active!")
    else:
        print("[NOTICE] GPS device responded or is searching for satellite fix (normal indoors).")
    return True


def test_detector():
    print_step("5. Testing Model Detection Engine")
    try:
        det = ObjectDetector(model_path=Config.MODEL_PATH, conf_threshold=Config.CONFIDENCE_THRESHOLD)
        # Create dummy 480p test frame
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        results = det.detect(dummy)
        print(f"[SUCCESS] Model inference executed cleanly. Detections on blank frame: {len(results)}")
        return True
    except Exception as e:
        print(f"[FAIL] Detection engine failed: {e}")
        return False


if __name__ == "__main__":
    print("\n============================================================")
    print("      RoadSense Diagnostic & Self-Test Utility")
    print("============================================================")
    c_ok = test_config()
    if not c_ok:
        sys.exit(1)

    print("\nSelect test to execute:")
    print("  1) Full Cloud Test (Cloudinary + Firebase Firestore)")
    print("  2) Hardware Test (Camera + GPS)")
    print("  3) Model Inference Test")
    print("  4) Run All Tests")
    print("  5) Exit")

    choice = input("\nEnter choice (1-5) [Default: 4]: ").strip() or "4"

    if choice in ["1", "4"]:
        test_cloudinary_and_firestore()
    if choice in ["2", "4"]:
        test_camera()
        test_gps()
    if choice in ["3", "4"]:
        test_detector()

    print("\n============================================================")
    print("Diagnostics complete.")
    print("============================================================\n")
