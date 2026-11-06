import time
import signal
import logging
import sys
from datetime import datetime

from config import Config
from gps_reader import GPSReader
from cloud_uploader import AsyncUploader
from camera_stream import CameraStream
from detector import ObjectDetector

# Setup formatted logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("RoadSenseMain")


class RoadSenseApp:
    """
    Main Application for Raspberry Pi 4 Model B:
    - 480p Camera Feed regulated at strictly 5 FPS
    - Non-blocking Background GPS Telemetry
    - NCNN/YOLO Object Detection
    - Asynchronous In-Memory Cloudinary Upload + Firebase Firestore Alerting (Zero Disk Writes)
    """

    def __init__(self):
        # 1. Validate environment configuration
        Config.validate()

        logger.info("Initializing RoadSense Pi 4 Edge Detection System...")
        logger.info(f"Device ID: {Config.DEVICE_ID}")
        logger.info(f"Target FPS: {Config.TARGET_FPS} (Interval: {1.0 / Config.TARGET_FPS:.3f}s)")
        logger.info(f"Camera: {Config.CAMERA_SOURCE} @ {Config.FRAME_WIDTH}x{Config.FRAME_HEIGHT}")

        self.running = False
        self.last_alert_time = 0.0

        # 2. Initialize subsystems
        self.gps_reader = GPSReader(
            port=Config.GPS_PORT,
            baudrate=Config.GPS_BAUDRATE,
            timeout=Config.GPS_TIMEOUT
        )

        self.uploader = AsyncUploader(
            config=Config,
            max_queue_size=50
        )

        self.camera = CameraStream(
            source=Config.CAMERA_SOURCE,
            width=Config.FRAME_WIDTH,
            height=Config.FRAME_HEIGHT,
            target_fps=Config.TARGET_FPS
        )

        self.detector = ObjectDetector(
            model_path=Config.MODEL_PATH,
            conf_threshold=Config.CONFIDENCE_THRESHOLD,
            imgsz=320
        )

        # 3. Setup signal handlers for graceful exit
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

    def _handle_shutdown(self, signum, frame):
        """Gracefully shuts down all threads and releases hardware."""
        logger.info("\nShutdown signal received. Cleaning up resources...")
        self.running = False

    def start(self):
        """Starts the subsystems and executes the 5 FPS detection loop."""
        # Start background GPS reader
        self.gps_reader.start()

        # Start asynchronous upload worker
        self.uploader.start()

        # Open camera
        if not self.camera.start():
            logger.error("Could not start camera. Exiting.")
            self.stop()
            sys.exit(1)

        self.running = True
        logger.info(">>> RoadSense System is ACTIVE and running at 5 FPS <<<\n")

        # Strict 5 FPS timing control: 1 / 5 = 0.200 seconds per frame
        target_frame_time = 1.0 / Config.TARGET_FPS
        frame_count = 0
        fps_timer = time.perf_counter()

        try:
            while self.running:
                loop_start = time.perf_counter()

                # 1. Read fresh 480p frame
                success, frame = self.camera.read_frame()
                if not success:
                    logger.warning("Dropped or empty camera frame. Retrying...")
                    time.sleep(0.05)
                    continue

                frame_count += 1

                # 2. Run object detection
                detections = self.detector.detect(frame)

                # 3. Process alert if objects detected and debounce cooldown has elapsed
                current_time = time.time()
                if detections:
                    time_since_last_alert = current_time - self.last_alert_time
                    if time_since_last_alert >= Config.DETECTION_COOLDOWN:
                        self.last_alert_time = current_time

                        # Fetch thread-safe GPS coordinates
                        gps_coords = self.gps_reader.get_coordinates()

                        logger.info(
                            f"DETECTED {len(detections)} object(s)! "
                            f"Top: {detections[0]['label']} ({detections[0]['confidence']*100:.1f}%) | "
                            f"GPS Fix: {gps_coords['has_fix']} (Lat: {gps_coords['latitude']}, Lon: {gps_coords['longitude']})"
                        )

                        # Annotate frame in RAM with bounding boxes & GPS watermark
                        annotated_frame = self.detector.annotate_frame(frame, detections, gps_coords)

                        # Encode directly to in-memory JPEG bytes (ZERO disk storage)
                        jpeg_bytes = self.detector.encode_to_jpeg_bytes(
                            annotated_frame,
                            quality=Config.JPEG_QUALITY
                        )

                        # Asynchronously hand off to upload queue (non-blocking, returns in <1ms)
                        self.uploader.enqueue(jpeg_bytes, detections, gps_coords)

                    else:
                        logger.debug(
                            f"Detection ignored by cooldown "
                            f"({time_since_last_alert:.1f}s / {Config.DETECTION_COOLDOWN}s)"
                        )

                # 4. Strict 5 FPS Rate Limiting
                elapsed = time.perf_counter() - loop_start
                sleep_duration = target_frame_time - elapsed
                if sleep_duration > 0:
                    time.sleep(sleep_duration)

                # 5. Periodic FPS telemetry logging (every 25 frames = 5 seconds)
                if frame_count % 25 == 0:
                    now = time.perf_counter()
                    actual_fps = 25.0 / (now - fps_timer)
                    fps_timer = now
                    logger.info(
                        f"[Heartbeat] Scan loop running at: {actual_fps:.2f} FPS | "
                        f"Queue backlog: {self.uploader.queue.qsize()}"
                    )

        except Exception as e:
            logger.error(f"Fatal error in main loop: {e}", exc_info=True)
        finally:
            self.stop()

    def stop(self):
        """Stops all threads, releases devices, and flushes remaining uploads."""
        logger.info("Stopping RoadSense application...")
        self.camera.release()
        self.gps_reader.stop()
        self.uploader.stop(timeout=4.0)
        logger.info("RoadSense shutdown complete.")


if __name__ == "__main__":
    app = RoadSenseApp()
    app.start()
