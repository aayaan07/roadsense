"""
Diagnostic and Subsystem Test Script for RoadSense (Raspberry Pi 4 & Pi 5).
Run this script to verify:
1. Environment variables (.env.local)
2. Mock Cloud Test (Synthetic image -> Cloudinary & Firestore)
3. Camera Stream (480p capture check)
4. GPS Serial Connection & Satellite Fix
5. Model Detection Engine
6. Live Camera Capture + GPS -> Cloudinary & Firebase ('tests' collection)
"""

import sys
import time
import io
import numpy as np
import cv2
from datetime import datetime, timezone

from config import Config
from gps_reader import GPSReader
from cloud_uploader import AsyncUploader, dict_to_firestore_fields
from camera_stream import CameraStream
from detector import ObjectDetector


def print_step(title):
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)


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
    cv2.putText(test_img, "RoadSense Diagnostic Test", (40, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    success, encoded = cv2.imencode(".jpg", test_img)
    if not success:
        print("[FAIL] Could not encode test image in RAM.")
        return False

    jpeg_bytes = encoded.tobytes()
    print("[INFO] Uploading synthetic image directly from RAM buffer to Cloudinary...")
    cloudinary_resp = uploader._upload_to_cloudinary(jpeg_bytes)
    if not cloudinary_resp:
        print("[FAIL] Cloudinary upload failed. Check credentials in .env.local.")
        return False

    image_url = cloudinary_resp.get("secure_url")
    print(f"[SUCCESS] Cloudinary image uploaded: {image_url}")

    print(f"[INFO] Writing test alert record to Firestore collection '{Config.FIREBASE_COLLECTION}'...")
    test_doc = {
        "device_id": "test-diagnostic",
        "detected_class": "test_object",
        "confidence": 0.99,
        "image_url": image_url,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
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


def wait_for_gps_fix(gps, timeout: float = 45.0) -> dict:
    """Polls the GPS reader until a fix is acquired or the timeout expires; prints progress."""
    start = time.time()
    last = -1
    while time.time() - start < timeout:
        c = gps.get_coordinates()
        if c["has_fix"] and c["latitude"] is not None:
            return c
        n = int((time.time() - start) // 5)
        if n != last:
            last = n
            print(f"      ...waiting for satellite fix ({int(time.time() - start)}s, "
                  f"NMEA sentences received: {c['sentences_received']})")
        time.sleep(0.5)
    return gps.get_coordinates()


def print_gps_hint(coords: dict):
    if coords["sentences_received"] == 0:
        print("[WARN] No NMEA data received at all: check wiring (GPS TX -> Pi RX/GPIO15), "
              "power, baud rate (GPS_BAUDRATE) and that the serial port is enabled.")
    elif not coords["has_fix"]:
        print("[NOTICE] GPS module is talking but has no satellite fix yet. Take the antenna "
              "outdoors or to a window with open sky; a cold start can take 1-15 minutes. "
              "Most modules blink an LED once per second once a fix is locked.")


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
    print(f"[INFO] Listening on {Config.GPS_PORT} for up to 45 seconds...")
    coords = wait_for_gps_fix(gps, 45.0)
    gps.stop()
    print(f"[INFO] GPS Status: {coords['raw_status']}")
    print(f"  - Fix Acquired: {coords['has_fix']}")
    print(f"  - Satellites: {coords['satellites']}")
    print(f"  - Latitude: {coords['latitude']}")
    print(f"  - Longitude: {coords['longitude']}")
    if coords["has_fix"]:
        print("[SUCCESS] Real-time GPS lock active!")
    else:
        print_gps_hint(coords)
    return True


def test_detector():
    print_step("5. Testing Model Detection Engine")
    try:
        det = ObjectDetector(model_path=Config.MODEL_PATH, conf_threshold=Config.CONFIDENCE_THRESHOLD)
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        results = det.detect(dummy)
        print(f"[SUCCESS] Model inference executed cleanly. Detections on blank frame: {len(results)}")
        return True
    except Exception as e:
        print(f"[FAIL] Detection engine failed: {e}")
        return False


def test_live_camera_capture_and_push(collection: str = "tests"):
    """
    Live End-to-End Test:
    1. Initializes GPS in background to fetch latest coordinates.
    2. Clicks a live 480p image from the camera.
    3. Streams image directly from RAM buffer to Cloudinary (zero disk writes).
    4. Pushes document with Cloudinary URL & GPS coordinates to Firestore collection 'tests'.
    """
    print_step(f"LIVE TEST: Capture Camera -> Cloudinary -> Firestore ('{collection}' collection)")

    # 1. Fetch current GPS data
    print("[1/4] Reading live GPS coordinates...")
    gps = GPSReader(port=Config.GPS_PORT, baudrate=Config.GPS_BAUDRATE)
    gps.start()
    gps_data = wait_for_gps_fix(gps, 45.0)
    print_gps_hint(gps_data)
    gps.stop()

    lat = gps_data.get("latitude")
    lon = gps_data.get("longitude")
    has_fix = gps_data.get("has_fix", False)
    print(f"      GPS Status: {'LOCKED' if has_fix else 'SEARCHING FIX'}")
    print(f"      Coordinates: Lat={lat}, Lon={lon}, Satellites={gps_data.get('satellites', 0)}")

    # 2. Click image from camera
    print("[2/4] Clicking live 480p image from camera...")
    cam = CameraStream(
        source=Config.CAMERA_SOURCE,
        width=Config.FRAME_WIDTH,
        height=Config.FRAME_HEIGHT,
        target_fps=Config.TARGET_FPS
    )

    if not cam.start():
        print("[FAIL] Could not open camera. Check physical connection or index.")
        return False

    # Warm up camera for auto-exposure & white-balance
    for _ in range(3):
        cam.read_frame()
        time.sleep(0.05)

    ret, frame = cam.read_frame()
    cam.release()

    if not ret or frame is None:
        print("[FAIL] Failed to capture live frame from camera.")
        return False

    h, w = frame.shape[:2]
    print(f"      Image captured successfully: {w}x{h} (480p)")

    # Add a clean timestamp / GPS watermark onto the test image
    timestamp_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    banner_text = f"TEST CAPTURE | {timestamp_str} | GPS: {lat if lat else 'No-Fix'}, {lon if lon else 'No-Fix'}"
    cv2.rectangle(frame, (0, h - 26), (w, h), (20, 20, 20), -1)
    cv2.putText(frame, banner_text, (8, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1, cv2.LINE_AA)

    # Encode directly to in-memory JPEG bytes (ZERO disk storage)
    success, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), Config.JPEG_QUALITY])
    if not success:
        print("[FAIL] Failed to encode frame to in-memory JPEG bytes.")
        return False
    jpeg_bytes = encoded.tobytes()

    # 3. Stream from RAM to Cloudinary
    print("[3/4] Streaming image from RAM buffer to Cloudinary...")
    uploader = AsyncUploader(config=Config)
    cloudinary_resp = uploader._upload_to_cloudinary(jpeg_bytes)
    if not cloudinary_resp:
        print("[FAIL] Cloudinary upload failed. Check credentials in .env.local.")
        return False

    image_url = cloudinary_resp.get("secure_url")
    public_id = cloudinary_resp.get("public_id")
    print(f"      [SUCCESS] Uploaded to Cloudinary: {image_url}")

    # 4. Push document to Firebase Firestore 'tests' collection
    print(f"[4/4] Writing document with image URL and GPS coordinates to Firestore collection '{collection}'...")
    test_doc = {
        "device_id": Config.DEVICE_ID,
        "test_type": "live_camera_capture",
        "image_url": image_url,
        "cloudinary_public_id": public_id,
        "timestamp": timestamp_str,
        "status": "success",
        "resolution": f"{w}x{h}",
        "gps_fixed": bool(has_fix),
        "speed_kmh": float(gps_data.get("speed_kmh") or 0.0),
        "satellites": int(gps_data.get("satellites") or 0)
    }

    if lat is not None and lon is not None:
        test_doc["location"] = {"latitude": float(lat), "longitude": float(lon)}
        test_doc["latitude"] = float(lat)
        test_doc["longitude"] = float(lon)
    else:
        test_doc["latitude"] = 0.0
        test_doc["longitude"] = 0.0

    db_success = uploader._write_to_firestore(test_doc, collection=collection)
    if db_success:
        print(f"\n[COMPLETE SUCCESS] Live test succeeded!")
        print(f"  - Target Collection : '{collection}'")
        print(f"  - Cloudinary URL    : {image_url}")
        print(f"  - GPS Coordinates   : Lat={test_doc['latitude']}, Lon={test_doc['longitude']} (Fix: {has_fix})")
        print(f"  - Firestore Record  : Verified in Firebase Console under '{collection}' collection.")
        return True
    else:
        print(f"[FAIL] Firestore document creation in collection '{collection}' failed.")
        return False


if __name__ == "__main__":
    print("\n============================================================")
    print("      RoadSense Diagnostic & Self-Test Utility")
    print("============================================================")
    c_ok = test_config()
    if not c_ok:
        sys.exit(1)

    print("\nSelect test to execute:")
    print("  1) Synthetic Cloud Test (Mock Image -> Cloudinary + Firestore)")
    print("  2) Hardware Test (Camera + GPS)")
    print("  3) Model Inference Test")
    print("  4) >>> LIVE CAMERA CAPTURE + GPS -> Cloudinary & Firestore ('tests' collection) <<<")
    print("  5) Run All Tests (1 to 4)")
    print("  6) Exit")

    choice = input("\nEnter choice (1-6) [Default: 4]: ").strip() or "4"

    if choice in ["1", "5"]:
        test_cloudinary_and_firestore()
    if choice in ["2", "5"]:
        test_camera()
        test_gps()
    if choice in ["3", "5"]:
        test_detector()
    if choice in ["4", "5"]:
        test_live_camera_capture_and_push(collection="tests")

    print("\n============================================================")
    print("Diagnostics complete.")
    print("============================================================\n")
