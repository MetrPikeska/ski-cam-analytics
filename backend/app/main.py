"""
FastAPI server - hlavní API pro ovládání a monitoring.

DŮLEŽITÉ: Server NESPOUŠTÍ pipeline automaticky!
Pipeline se spouští pouze přes POST /api/pipeline/start
"""
import logging
import asyncio
from datetime import datetime
from typing import List
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from . import config
from .pipeline import Pipeline
from .models import PipelineStatus, MetricsLatest, ConfigUpdate, TimeSeriesPoint
from .storage import MetricsStorage

# Logging setup
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Globální instance pipeline
pipeline = Pipeline()

# Storage instance (pro čtení dat i když pipeline neběží)
storage = MetricsStorage(config.DB_PATH)

# WebSocket connections
active_websockets = set()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager.
    
    DŮLEŽITÉ: Pipeline se NESPOUŠTÍ automaticky při startu serveru!
    """
    logger.info("=== SERVER STARTING ===")
    logger.info("Pipeline is NOT started automatically")
    logger.info("Use POST /api/pipeline/start to begin analysis")
    
    # Spustit WebSocket broadcast task
    broadcast_task = asyncio.create_task(broadcast_metrics())
    
    yield
    
    # Shutdown
    logger.info("=== SERVER SHUTTING DOWN ===")
    
    # Zastavit pipeline pokud běží
    if pipeline.is_running:
        logger.info("Stopping pipeline...")
        pipeline.stop()
    
    # Zrušit broadcast task
    broadcast_task.cancel()
    try:
        await broadcast_task
    except asyncio.CancelledError:
        pass
    
    logger.info("=== SERVER STOPPED ===")


# FastAPI app
app = FastAPI(
    title="Ski Cam Analytics API",
    description="API pro počítání lidí v lyžařském areálu",
    version="1.0.0",
    lifespan=lifespan
)

# CORS (pro lokální vývoj)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# === API ENDPOINTS ===

@app.get("/api/status", response_model=PipelineStatus)
async def get_status():
    """Vrátí aktuální status pipeline."""
    status = pipeline.get_status()
    return PipelineStatus(**status)


@app.post("/api/pipeline/start")
async def start_pipeline():
    """
    Spustí pipeline processing.
    
    KLÍČOVÝ ENDPOINT: Toto je jediný způsob jak spustit analýzu!
    """
    logger.info("API: Starting pipeline...")
    
    success = pipeline.start()
    
    if success:
        return {"status": "started", "message": "Pipeline started successfully"}
    else:
        raise HTTPException(status_code=500, detail="Failed to start pipeline")


@app.post("/api/pipeline/stop")
async def stop_pipeline():
    """Zastaví pipeline processing."""
    logger.info("API: Stopping pipeline...")
    
    pipeline.stop()
    
    return {"status": "stopped", "message": "Pipeline stopped successfully"}


@app.get("/api/metrics/latest", response_model=MetricsLatest)
async def get_latest_metrics():
    """Vrátí nejnovější metriky."""
    metrics = pipeline.get_current_metrics()
    return MetricsLatest(**metrics)


@app.get("/api/metrics/timeseries", response_model=List[TimeSeriesPoint])
async def get_timeseries(minutes: int = 60):
    """
    Vrátí časovou řadu metrik za posledních N minut.
    
    Args:
        minutes: Počet minut zpět (default 60)
    """
    data = storage.get_timeseries(minutes=minutes)
    
    result = []
    for item in data:
        result.append(TimeSeriesPoint(
            timestamp=datetime.fromisoformat(item['timestamp']),
            occupancy_avg=item['occupancy_avg'],
            crossings=item['crossings']
        ))
    
    return result


@app.get("/api/config")
async def get_config():
    """Vrátí aktuální konfiguraci."""
    return {
        "stream_url": config.STREAM_URL,
        "ffmpeg_fps": config.FFMPEG_FPS,
        "frame_width": config.FRAME_WIDTH,
        "frame_height": config.FRAME_HEIGHT,
        "conf_threshold": config.CONF_THRESHOLD,
        "roi_rect": config.ROI_RECT,
        "line_crossing": config.LINE_CROSSING,
        "tracker_max_age": config.TRACKER_MAX_AGE,
        "tracker_min_hits": config.TRACKER_MIN_HITS
    }


@app.get("/api/frame/latest")
async def get_latest_frame():
    """
    Vrátí poslední frame s vizualizací jako JPEG.
    """
    import cv2
    import numpy as np
    
    frame = pipeline.get_latest_frame()
    
    if frame is None:
        # Vrátit prázdný černý obrázek pokud není frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(frame, "No frame available", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Enkódovat do JPEG
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    
    return StreamingResponse(
        iter([buffer.tobytes()]),
        media_type="image/jpeg",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"}
    )


@app.post("/api/config")
async def update_config(update: ConfigUpdate):
    """
    Aktualizuje konfiguraci.
    
    POZOR: Změny se projeví až po restartu pipeline!
    """
    updated_fields = []
    
    if update.roi_rect is not None:
        config.ROI_RECT = tuple(update.roi_rect) if update.roi_rect else None
        updated_fields.append("roi_rect")
    
    if update.line_crossing is not None:
        if update.line_crossing:
            config.LINE_CROSSING = [tuple(p) for p in update.line_crossing]
        else:
            config.LINE_CROSSING = None
        updated_fields.append("line_crossing")
    
    if update.conf_threshold is not None:
        config.CONF_THRESHOLD = update.conf_threshold
        updated_fields.append("conf_threshold")
    
    if update.ffmpeg_fps is not None:
        config.FFMPEG_FPS = update.ffmpeg_fps
        updated_fields.append("ffmpeg_fps")
    
    return {
        "status": "updated",
        "updated_fields": updated_fields,
        "note": "Restart pipeline to apply changes"
    }


# === WEBSOCKET ===

@app.websocket("/ws/live")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint pro real-time push metrik.
    
    Posílá metriky každou sekundu všem připojeným klientům.
    """
    await websocket.accept()
    active_websockets.add(websocket)
    
    logger.info("WebSocket client connected")
    
    try:
        # Čekat na zprávy (keep-alive)
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        active_websockets.remove(websocket)
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        if websocket in active_websockets:
            active_websockets.remove(websocket)


async def broadcast_metrics():
    """
    Background task pro broadcast metrik přes WebSocket.
    
    Posílá metriky každou sekundu.
    """
    while True:
        try:
            await asyncio.sleep(config.WS_UPDATE_INTERVAL)
            
            if not active_websockets:
                continue
            
            # Získat aktuální metriky
            status = pipeline.get_status()
            metrics = pipeline.get_current_metrics()
            
            data = {
                "status": status,
                "metrics": {
                    "timestamp": metrics['timestamp'].isoformat(),
                    "occupancy": metrics['occupancy'],
                    "crossings_this_run": metrics['crossings_this_run'],
                    "crossings_last_1m": metrics['crossings_last_1m'],
                    "crossings_last_10m": metrics['crossings_last_10m']
                }
            }
            
            # Broadcast všem klientům
            disconnected = set()
            for ws in active_websockets:
                try:
                    await ws.send_json(data)
                except Exception:
                    disconnected.add(ws)
            
            # Odstranit disconnected
            for ws in disconnected:
                active_websockets.remove(ws)
                
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Broadcast error: {e}")


# === STATIC FILES (Frontend) ===

# Mount frontend static files
app.mount("/static", StaticFiles(directory=str(config.BASE_DIR / "frontend")), name="static")


@app.get("/")
async def serve_frontend():
    """Serve frontend index.html"""
    return FileResponse(config.BASE_DIR / "frontend" / "index.html")


# === HEALTH CHECK ===

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


if __name__ == "__main__":
    import uvicorn
    
    logger.info(f"Starting server on {config.API_HOST}:{config.API_PORT}")
    
    uvicorn.run(
        "app.main:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=False,
        log_level=config.LOG_LEVEL.lower()
    )
