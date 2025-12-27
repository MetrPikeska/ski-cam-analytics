"""
Datové modely pro API.
"""
from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class PipelineStatus(BaseModel):
    """Status pipeline."""
    running: bool
    stream_online: bool
    fps: float
    start_time: Optional[datetime]
    uptime_seconds: Optional[float]


class MetricsLatest(BaseModel):
    """Aktuální metriky."""
    timestamp: datetime
    occupancy: int  # Počet lidí ve scéně
    crossings_this_run: int  # Celkem crossingů od startu
    crossings_last_1m: int  # Crossingy za poslední 1 minutu
    crossings_last_10m: int  # Crossingy za posledních 10 minut


class ConfigUpdate(BaseModel):
    """Aktualizace konfigurace."""
    roi_rect: Optional[list] = None  # [x1, y1, x2, y2]
    line_crossing: Optional[list] = None  # [[x1, y1], [x2, y2]]
    conf_threshold: Optional[float] = None
    ffmpeg_fps: Optional[int] = None


class TimeSeriesPoint(BaseModel):
    """Bod v časové řadě metrik."""
    timestamp: datetime
    occupancy_avg: float
    crossings: int
