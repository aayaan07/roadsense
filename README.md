# RoadSense - Raspberry Pi 4 & 5 Edge Detection System

High-performance real-time edge detection system for Raspberry Pi 4 Model B and Raspberry Pi 5. Detects road hazards/potholes using an NCNN/YOLO model, stamps live GPS coordinates, streams images directly from RAM to Cloudinary, and triggers alerts in Firebase Firestore.

---

## Key Features

- **Strict 5 FPS Rate Limiting**: Detection loop is precisely paced at 5 FPS (0.2s interval) to optimize CPU/thermal performance on the Pi 4.
- **480p Camera Stream**: Captures 640x480 resolution with buffer draining to eliminate video latency.
- **Zero Disk Storage**: Frames are captured, annotated, and encoded directly in memory (`io.BytesIO`). **No images or videos are ever written to the SD card**, preventing flash memory wear.
- **Asynchronous Non-blocking Upload**: A dedicated background worker queue processes Cloudinary uploads and Firestore writes. The 5 FPS detection loop never stalls or stutters during network requests.
- **Background GPS Telemetry**: Continuously parses `$GPRMC` and `$GPGGA` NMEA sentences in a daemon thread. Thread-safe coordinate retrieval with no serial port lock contention.
- **Alert Debounce Cooldown**: Prevents duplicate alert spam when a pothole remains in view across consecutive frames.
- **Direct Firestore REST API**: Directly writes structured documents (including `GeoPoint` coordinates) using your Firebase Web API key and project ID from `.env.local`.

---

## Hardware Connection Guide (Raspberry Pi 4 Model B)

### 1. Camera
- **USB Webcam**: Plug into any USB 3.0 / 2.0 port (identical on Pi 4 and Pi 5).
- **Raspberry Pi Camera Module (CSI)**:
  - **Raspberry Pi 4**: Uses standard 15-pin 1.0mm pitch ribbon cable.
  - **Raspberry Pi 5**: Uses a 22-pin 0.5mm mini-CSI ribbon cable plugged into `CAM/DISP 0` or `CAM/DISP 1`. The codebase automatically detects and uses `Picamera2` for Pi 5 CSI cameras.

### 2. GPS Module (NEO-6M / NEO-7M / NEO-8M / USB GPS)
If using GPIO UART:
| GPS Module Pin | Raspberry Pi 4 Physical Header Pin | Pin Function |
| :--- | :--- | :--- |
| **VCC** | **Pin 2 or Pin 4** | 5V Power |
| **GND** | **Pin 6** | Ground |
| **TX** | **Pin 10 (GPIO 15)** | RXD (Pi Receive) |
| **RX** | **Pin 8 (GPIO 14)** | TXD (Pi Transmit) |

> If using a **USB GPS Dongle**, simply plug it into a USB port. It will register as `/dev/ttyUSB0` or `/dev/ttyACM0` (the code automatically detects this).

---

## Raspberry Pi OS Configuration

### 1. Enable Hardware Serial UART
Run the configuration tool:
```bash
sudo raspi-config
```
1. Navigate to: `Interface Options` -> `Serial Port`.
2. *"Would you like a login shell to be accessible over serial?"* -> Select **No**.
3. *"Would you like the serial port hardware to be enabled?"* -> Select **Yes**.
4. Finish and reboot the Raspberry Pi:
```bash
sudo reboot
```

---

## Installation & Setup

1. **Clone/Navigate to project directory**:
```bash
cd "d:\Second Year\Iot-Project\code" # Or your Pi path: ~/Iot-Project/code
```

2. **Create Python Virtual Environment**:
```bash
python3 -m venv --system-site-packages venv   # lets the venv see Picamera2 (CSI camera)
source venv/bin/activate
```

3. **Install Dependencies**:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note on NCNN Acceleration**: If running the model in NCNN mode, install the ncnn build for ARM:
> ```bash
> pip install ncnn
> ```

4. **Verify Configuration**:
Ensure `.env.local` contains your credentials:
```env
FIREBASE_API_KEY=...
FIREBASE_PROJECT_ID=roadsense-1174d
CLOUDINARY_CLOUD_NAME=...
CLOUDINARY_API_KEY=...
CLOUDINARY_API_SECRET=...
CLOUDINARY_UPLOAD_PRESET=ml_default
```

---

## Running Diagnostics & Tests

Before launching the full service, run the built-in diagnostic test to verify your camera, GPS, Cloudinary, and Firestore connections:

```bash
python test.py
```
Menu options available:
1. **Option 1**: Synthetic Cloud Test (Mock image -> Cloudinary & Firestore)
2. **Option 2**: Hardware Test (Camera + GPS)
3. **Option 3**: Model Inference Test
4. **Option 4** *(Default)*: **Live Camera Capture + GPS -> Cloudinary & Firestore (`tests` collection)**
   - Clicks a live 480p frame from your camera.
   - Reads current GPS coordinates.
   - Uploads directly from RAM buffer to Cloudinary (zero disk writes).
   - Writes document directly to Firebase Firestore under the **`tests`** collection.
5. **Option 5**: Run all tests.

---

## Running the Detection System

Start the detection system:
```bash
python main.py
```

### Console Output Example:
```
14:30:00 [INFO] RoadSenseMain: Initializing RoadSense Pi 4 Edge Detection System...
14:30:00 [INFO] RoadSenseMain: Device ID: rpi4-roadsense-01
14:30:00 [INFO] RoadSenseMain: Target FPS: 5.0 (Interval: 0.200s)
14:30:00 [INFO] RoadSenseMain: Camera: 0 @ 640x480
14:30:01 [INFO] GPSReader: Connected to GPS device on /dev/serial0
14:30:01 [INFO] CloudUploader: Asynchronous Upload Worker thread started.
14:30:01 [INFO] CameraStream: Camera opened successfully. Resolution: 640x480
14:30:01 [INFO] RoadSenseMain: >>> RoadSense System is ACTIVE and running at 5 FPS <<<

14:30:06 [INFO] RoadSenseMain: [Heartbeat] Scan loop running at: 5.01 FPS | Queue backlog: 0
14:30:11 [INFO] RoadSenseMain: [Heartbeat] Scan loop running at: 4.99 FPS | Queue backlog: 0
14:30:14 [INFO] RoadSenseMain: DETECTED 1 object(s)! Top: pothole (88.4%) | GPS Fix: True (Lat: 12.97159, Lon: 77.59456)
14:30:15 [INFO] CloudUploader: Cloudinary upload success: https://res.cloudinary.com/.../roadsense/detections/xyz.jpg
14:30:16 [INFO] CloudUploader: Firestore document successfully created: projects/roadsense-1174d/databases/(default)/documents/road_detections/abc123
```

---

## Auto-Start on Boot (Systemd Service)

To run RoadSense automatically whenever the vehicle starts up:

1. Create a service file:
```bash
sudo nano /etc/systemd/system/roadsense.service
```

2. Paste the following configuration (adjust paths to match your Pi username):
```ini
[Unit]
Description=RoadSense Pi 4 Real-Time Pothole Detection Service
After=network.target

[Service]
Type=simple
User=pi
SupplementaryGroups=dialout video
WorkingDirectory=/home/pi/Iot-Project/code
ExecStart=/home/pi/Iot-Project/code/venv/bin/python main.py
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

3. Enable and start the service:
```bash
sudo systemctl daemon-reload
sudo systemctl enable roadsense.service
sudo systemctl start roadsense.service
```

4. Check live service logs:
```bash
sudo journalctl -u roadsense.service -f
```
