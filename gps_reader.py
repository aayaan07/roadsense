import sys
import time
import threading
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional

logger = logging.getLogger("GPSReader")


class GPSReader:
    """
    Background GPS Reader for Raspberry Pi.
    Continuously parses NMEA sentences ($GPRMC, $GNRMC, $GPGGA, $GNGGA)
    from hardware UART (/dev/serial0) or USB GPS (/dev/ttyUSB0) without blocking.
    """

    def __init__(self, port: str = "/dev/serial0", baudrate: int = 9600, timeout: float = 1.0):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout

        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._serial_conn = None
        self._last_data_time = time.time()
        self._bad_ports = set()
        self.NO_DATA_TIMEOUT = 10.0

        # Latest GPS telemetry
        self.data: Dict[str, Any] = {
            "latitude": None,
            "longitude": None,
            "altitude_m": 0.0,
            "speed_kmh": 0.0,
            "satellites": 0,
            "has_fix": False,
            "timestamp": None,
            "raw_status": "initializing",
            "sentences_received": 0
        }

    def start(self):
        """Starts the GPS reading background thread."""
        self._running = True
        self._thread = threading.Thread(target=self._worker, daemon=True, name="GPSReaderThread")
        self._thread.start()
        logger.info(f"GPS Reader started on port: {self.port} @ {self.baudrate} baud")

    def stop(self):
        """Stops the GPS reading background thread."""
        self._running = False
        # Let the reader thread exit its blocking readline() (timeout ~1s) before closing the
        # port; closing underneath it makes pyserial raise a spurious TypeError.
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=self.timeout + 2.0)
        if self._serial_conn:
            try:
                self._serial_conn.close()
            except Exception:
                pass
            self._serial_conn = None
        logger.info("GPS Reader stopped.")

    def get_coordinates(self) -> Dict[str, Any]:
        """Thread-safe getter for latest GPS coordinates."""
        with self._lock:
            # Return a copy to avoid race conditions
            return dict(self.data)

    def _open_serial(self):
        """Attempts to open the configured serial port with fallback candidate ports."""
        import serial

        candidate_ports = [self.port, "/dev/serial0", "/dev/ttyAMA0", "/dev/ttyUSB0", "/dev/ttyACM0", "COM3", "COM4"]
        # Remove duplicates while preserving order
        candidate_ports = list(dict.fromkeys(candidate_ports))

        if all(p in self._bad_ports for p in candidate_ports):
            self._bad_ports.clear()
        for p in candidate_ports:
            if p in self._bad_ports:
                continue
            try:
                ser = serial.Serial(p, self.baudrate, timeout=self.timeout)
                logger.info(f"Successfully connected to GPS device on {p}")
                self.port = p
                self._last_data_time = time.time()
                return ser
            except Exception as e:
                logger.warning(f"Could not open GPS port {p}: {e}")

        logger.warning(
            "GPS hardware port could not be opened. "
            "Running in simulation/fallback mode. Check GPIO wiring or USB connection."
        )
        return None

    def _worker(self):
        """Continuous serial read loop."""
        try:
            import serial
        except ImportError:
            logger.error(
                "'pyserial' is not installed in this Python environment (%s). "
                "Run: pip install pyserial", sys.executable
            )
            return

        while self._running:
            if self._serial_conn is None:
                self._serial_conn = self._open_serial()
                if self._serial_conn is None:
                    # Retry after a delay if no serial device is present
                    time.sleep(5.0)
                    continue

            # Port opened but silent (e.g. /dev/serial0 mapped to the wrong UART): try next port
            if time.time() - self._last_data_time > self.NO_DATA_TIMEOUT:
                logger.warning(
                    f"No NMEA data from {self.port} for {self.NO_DATA_TIMEOUT:.0f}s; "
                    "trying other serial ports..."
                )
                self._bad_ports.add(self.port)
                try:
                    self._serial_conn.close()
                except Exception:
                    pass
                self._serial_conn = None
                continue

            try:
                line = self._serial_conn.readline().decode("ascii", errors="ignore").strip()
                if not line:
                    continue

                self._last_data_time = time.time()
                with self._lock:
                    self.data["sentences_received"] += 1
                # Match any talker ID (GP, GN, GL, GA, BD, ...)
                if line.startswith("$") and len(line) > 6:
                    kind = line[3:6]
                    if kind == "RMC":
                        self._parse_rmc(line)
                    elif kind == "GGA":
                        self._parse_gga(line)

            except Exception as e:
                if not self._running:
                    break
                logger.warning(f"Error reading GPS serial stream: {e}")
                try:
                    self._serial_conn.close()
                except Exception:
                    pass
                self._serial_conn = None
                time.sleep(2.0)

    @staticmethod
    def _convert_nmea_to_decimal(raw_val: str, direction: str) -> Optional[float]:
        """
        Converts NMEA format (DDMM.MMMM or DDDMM.MMMM) to Decimal Degrees.
        e.g., 1258.1234, N -> 12.968723
        """
        if not raw_val or not direction:
            return None
        try:
            # Find decimal point
            dot_idx = raw_val.find(".")
            if dot_idx == -1:
                return None
            
            # The 2 digits before the dot are minutes start
            degrees_len = dot_idx - 2
            if degrees_len <= 0:
                return None

            degrees = float(raw_val[:degrees_len])
            minutes = float(raw_val[degrees_len:])
            decimal = degrees + (minutes / 60.0)

            if direction.upper() in ["S", "W"]:
                decimal = -decimal
            return round(decimal, 7)
        except Exception:
            return None

    def _parse_rmc(self, sentence: str):
        """
        Parses $GPRMC / $GNRMC (Recommended Minimum Navigation Information).
        Example: $GPRMC,083559.00,A,1258.1234,N,07735.1234,E,0.026,,130526,,,A*7C
        """
        parts = sentence.split(",")
        if len(parts) < 8:
            return

        status = parts[2]  # 'A' = Valid fix, 'V' = Receiver warning / no fix
        lat_raw = parts[3]
        lat_dir = parts[4]
        lon_raw = parts[5]
        lon_dir = parts[6]
        speed_raw = parts[7]  # speed in knots

        has_fix = (status == "A")
        lat = self._convert_nmea_to_decimal(lat_raw, lat_dir) if has_fix else None
        lon = self._convert_nmea_to_decimal(lon_raw, lon_dir) if has_fix else None

        speed_kmh = 0.0
        if has_fix and speed_raw:
            try:
                speed_kmh = round(float(speed_raw) * 1.852, 2)  # 1 knot = 1.852 km/h
            except ValueError:
                pass

        with self._lock:
            self.data["has_fix"] = has_fix
            self.data["raw_status"] = "fix_acquired" if has_fix else "searching_satellites"
            if has_fix and lat is not None and lon is not None:
                self.data["latitude"] = lat
                self.data["longitude"] = lon
                self.data["speed_kmh"] = speed_kmh
                self.data["timestamp"] = datetime.now(timezone.utc).isoformat()

    def _parse_gga(self, sentence: str):
        """
        Parses $GPGGA / $GNGGA (Fix data: satellites count, altitude).
        Example: $GPGGA,083559.00,1258.1234,N,07735.1234,E,1,08,1.01,912.4,M,45.2,M,,*52
        """
        parts = sentence.split(",")
        if len(parts) < 10:
            return

        try:
            fix_quality = int(parts[6]) if parts[6] else 0
            satellites = int(parts[7]) if parts[7] else 0
            alt_raw = parts[9] if parts[9] else "0.0"
            altitude = float(alt_raw)

            with self._lock:
                self.data["satellites"] = satellites
                self.data["altitude_m"] = altitude
                if fix_quality > 0:
                    self.data["has_fix"] = True
        except Exception:
            pass
