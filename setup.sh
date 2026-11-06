#!/usr/bin/env bash
# ==============================================================================
# RoadSense Setup Script for Raspberry Pi 4 Model B
# ==============================================================================

set -e

echo "=== RoadSense RPi4 Setup Started ==="

# 1. Update package lists and install system dependencies
echo "[1/4] Installing system dependencies (OpenCV runtime, v4l2, Python dev)..."
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv libgl1-mesa-glx libglib2.0-0 v4l-utils

# 2. Setup Python virtual environment
echo "[2/4] Setting up Python virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate

# 3. Upgrade pip and install requirements
echo "[3/4] Installing Python libraries..."
pip install --upgrade pip
pip install -r requirements.txt

# 4. Prompt to configure serial permissions
echo "[4/4] Adding current user ($USER) to dialout group for GPS serial access..."
sudo usermod -a -G dialout "$USER"

echo ""
echo "=== Setup Complete! ==="
echo "To test the system, run:"
echo "    source venv/bin/activate"
echo "    python test.py"
echo ""
echo "To start RoadSense:"
echo "    python main.py"
