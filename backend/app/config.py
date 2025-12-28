"""
Konfigurace aplikace - všechny parametry na jednom místě.
"""
import os
from pathlib import Path

# Root adresář projektu
BASE_DIR = Path(__file__).parent.parent.parent

# HLS Stream URL
STREAM_URL = "https://stream.teal.cz/hls/cam273.m3u8"

# ONNX Model
MODEL_PATH = BASE_DIR / "models" / "yolo.onnx"
CONF_THRESHOLD = 0.15  # Confidence threshold pro detekce (sníženo pro testování)
IOU_THRESHOLD = 0.45  # NMS threshold

# FFmpeg
FFMPEG_FPS = 8  # Kolik FPS zpracovávat ze streamu (nižší = menší zátěž CPU)
FRAME_WIDTH = 640  # Resize frame width (menší = rychlejší inference)
FRAME_HEIGHT = 480  # Resize frame height

# Tracking
TRACKER_MAX_AGE = 30  # Max frames bez detekce než track zanikne
TRACKER_MIN_HITS = 1  # Min počet detekcí pro "confirmed" track (sníženo na 1 pro okamžité tracky)
TRACKER_IOU_THRESHOLD = 0.3  # IoU threshold pro matching tracků

# Region of Interest (ROI) - souřadnice obdélníku [x1, y1, x2, y2]
# None = celý frame, jinak tuple (x1, y1, x2, y2) v pixelech
ROI_RECT = None  # TODO: Nastavit podle konkrétní kamery
# Příklad: ROI_RECT = (100, 150, 540, 450)

# Line Crossing - čára definovaná dvěma body [(x1, y1), (x2, y2)]
# None = vypnuto
LINE_CROSSING = None  # TODO: Nastavit podle pozice vleku/brány
# Příklad: LINE_CROSSING = [(200, 300), (440, 300)]  # horizontální čára

# Database
DB_PATH = BASE_DIR / "data" / "metrics.db"

# API
API_HOST = "0.0.0.0"
API_PORT = 8000

# WebSocket
WS_UPDATE_INTERVAL = 1.0  # Interval pro push metrik přes WebSocket (sekundy)

# Logging
LOG_LEVEL = "DEBUG"
