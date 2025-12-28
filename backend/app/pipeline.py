"""
Pipeline - hlavní smyčka pro video processing.

KLÍČOVÉ: Pipeline běží pouze po ručním spuštění přes API!
"""
import threading
import time
import logging
import numpy as np
from datetime import datetime, timedelta
from collections import deque
from typing import Optional

from . import config
from .ffmpeg_source import FFmpegSource
from .detector_onnx import YOLODetector
from .tracker import SORTTracker
from .analytics import Analytics
from .storage import MetricsStorage

logger = logging.getLogger(__name__)


class Pipeline:
    """
    Hlavní pipeline pro video analytics.
    
    Pipeline se spouští POUZE ručně přes start() metodu.
    Nesmí se spustit automaticky při inicializaci!
    """
    
    def __init__(self):
        """Inicializace pipeline (NESPOUŠTÍ processing!)."""
        
        # Komponenty (vytvoří se až při startu)
        self.ffmpeg: Optional[FFmpegSource] = None
        self.detector: Optional[YOLODetector] = None
        self.tracker: Optional[SORTTracker] = None
        self.analytics: Optional[Analytics] = None
        self.storage: Optional[MetricsStorage] = None
        
        # State
        self.is_running = False
        self.processing_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        
        # Metriky
        self.start_time: Optional[datetime] = None
        self.fps_counter = deque(maxlen=30)
        self.current_fps = 0.0
        
        # Vizualizace
        self.latest_frame: Optional[np.ndarray] = None
        self.frame_lock = threading.Lock()
        
        # Agregace pro ukládání do DB (per-minute)
        self.minute_occupancy_values = []
        self.minute_crossings = 0
        self.last_minute_save = None
        
        logger.info("Pipeline initialized (NOT RUNNING)")
    
    def start(self):
        """
        Spustí pipeline processing.
        
        KLÍČOVÁ METODA: Toto je jediný způsob jak spustit analýzu!
        """
        if self.is_running:
            logger.warning("Pipeline už běží!")
            return False
        
        logger.info("=== STARTING PIPELINE ===")
        
        try:
            # Inicializovat storage
            self.storage = MetricsStorage(config.DB_PATH)
            
            # Načíst ONNX model
            logger.info("Loading YOLO model...")
            self.detector = YOLODetector(
                model_path=config.MODEL_PATH,
                conf_threshold=config.CONF_THRESHOLD,
                iou_threshold=config.IOU_THRESHOLD,
                input_size=(640, 640)
            )
            
            # Vytvořit tracker
            self.tracker = SORTTracker(
                max_age=config.TRACKER_MAX_AGE,
                min_hits=config.TRACKER_MIN_HITS,
                iou_threshold=config.TRACKER_IOU_THRESHOLD
            )
            
            # Vytvořit analytics
            self.analytics = Analytics(
                roi_rect=config.ROI_RECT,
                line=config.LINE_CROSSING
            )
            
            # Spustit FFmpeg
            logger.info("Starting FFmpeg...")
            self.ffmpeg = FFmpegSource(
                stream_url=config.STREAM_URL,
                fps=config.FFMPEG_FPS,
                width=config.FRAME_WIDTH,
                height=config.FRAME_HEIGHT
            )
            self.ffmpeg.start()
            
            # Počkat na první frame (ověření že stream funguje)
            test_frame = self.ffmpeg.get_frame(timeout=10.0)
            if test_frame is None:
                raise RuntimeError("Nepodařilo se získat frame ze streamu!")
            
            logger.info(f"Stream OK, frame shape: {test_frame.shape}")
            
            # Reset state
            self.stop_event.clear()
            self.start_time = datetime.now()
            self.last_minute_save = datetime.now()
            self.is_running = True
            
            # Spustit processing thread
            self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
            self.processing_thread.start()
            
            logger.info("=== PIPELINE STARTED ===")
            return True
            
        except Exception as e:
            logger.error(f"Chyba při spouštění pipeline: {e}", exc_info=True)
            self._cleanup()
            return False
    
    def stop(self):
        """
        Zastaví pipeline processing.
        """
        if not self.is_running:
            logger.warning("Pipeline neběží")
            return
        
        logger.info("=== STOPPING PIPELINE ===")
        
        self.stop_event.set()
        self.is_running = False
        
        # Počkat na processing thread
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=5)
        
        # Uložit poslední agregaci
        self._save_minute_aggregate()
        
        # Cleanup
        self._cleanup()
        
        logger.info("=== PIPELINE STOPPED ===")
    
    def _cleanup(self):
        """Uvolní všechny resources."""
        if self.ffmpeg:
            self.ffmpeg.stop()
            self.ffmpeg = None
        
        self.detector = None
        self.tracker = None
        self.analytics = None
        # storage ponechat (potřebujeme pro čtení dat)
    
    def _processing_loop(self):
        """
        Hlavní processing smyčka (běží ve vlastním threadu).
        """
        logger.info("Processing loop started")
        
        frame_count = 0
        last_fps_time = time.time()
        
        while not self.stop_event.is_set():
            try:
                # Získat frame
                frame = self.ffmpeg.get_frame(timeout=1.0)
                if frame is None:
                    logger.warning("Timeout při čtení frame")
                    continue
                
                frame_start = time.time()
                
                # 1. Detekce
                detections = self.detector.detect(frame)
                logger.debug(f"DEBUG Pipeline: {len(detections)} detections")
                
                # 2. Filtrování podle ROI
                det_bboxes = [d.bbox for d in detections]
                filtered_bboxes = self.analytics.roi_filter.filter_detections(det_bboxes)
                logger.debug(f"DEBUG Pipeline: {len(filtered_bboxes)} after ROI filter")
                
                # 3. Tracking
                tracks = self.tracker.update(filtered_bboxes)
                logger.debug(f"DEBUG Pipeline: {len(tracks)} tracks")
                
                # 4. Analytics (occupancy, line crossing)
                self.analytics.update(tracks)
                metrics = self.analytics.get_metrics()
                logger.debug(f"DEBUG Pipeline: Occupancy={metrics['occupancy']}")
                
                # 5. Vizualizace
                vis_frame = self._draw_visualizations(frame.copy(), filtered_bboxes, tracks)
                with self.frame_lock:
                    self.latest_frame = vis_frame
                
                # 6. Ukládání agregací po minutách
                self._check_minute_aggregate()
                
                # FPS counter
                frame_time = time.time() - frame_start
                self.fps_counter.append(1.0 / frame_time if frame_time > 0 else 0)
                
                frame_count += 1
                
                # Každých 30 frames spočítat průměrné FPS
                if frame_count % 30 == 0:
                    elapsed = time.time() - last_fps_time
                    self.current_fps = 30 / elapsed if elapsed > 0 else 0
                    last_fps_time = time.time()
                    
                    metrics = self.analytics.get_metrics()
                    logger.info(
                        f"FPS: {self.current_fps:.1f} | "
                        f"Occupancy: {metrics['occupancy']} | "
                        f"Crossings: {metrics['crossings_this_run']}"
                    )
                
            except Exception as e:
                logger.error(f"Chyba v processing loop: {e}", exc_info=True)
                time.sleep(1)
        
        logger.info("Processing loop finished")
    
    def _check_minute_aggregate(self):
        """
        Kontroluje zda uplynula minuta a ukládá agregaci do DB.
        """
        now = datetime.now()
        
        # Uložit aktuální occupancy pro průměr
        metrics = self.analytics.get_metrics()
        self.minute_occupancy_values.append(metrics['occupancy'])
        
        # Přičíst nové crossingy
        # (crossings_last_1m je už za celou minutu, takže použijeme delta)
        # Pro jednoduchost: počítáme total crossings a bereme rozdíl
        
        # Každou minutu uložit
        if (now - self.last_minute_save) >= timedelta(minutes=1):
            self._save_minute_aggregate()
            self.last_minute_save = now
    
    def _save_minute_aggregate(self):
        """Uloží agregaci za minutu do databáze."""
        if not self.minute_occupancy_values or not self.storage:
            return
        
        avg_occupancy = sum(self.minute_occupancy_values) / len(self.minute_occupancy_values)
        max_occupancy = max(self.minute_occupancy_values)
        
        # Crossingy za tuto minutu
        metrics = self.analytics.get_metrics()
        crossings = metrics['crossings_last_1m']
        
        # Uložit do DB
        timestamp = datetime.now().replace(second=0, microsecond=0)
        self.storage.insert_minute_aggregate(
            timestamp=timestamp,
            occupancy_avg=avg_occupancy,
            occupancy_max=max_occupancy,
            crossings=crossings
        )
        
        # Reset pro další minutu
        self.minute_occupancy_values = []
    
    def get_status(self) -> dict:
        """
        Vrátí aktuální status pipeline.
        
        Returns:
            Dict se statusem
        """
        uptime = None
        if self.start_time:
            uptime = (datetime.now() - self.start_time).total_seconds()
        
        stream_online = False
        if self.ffmpeg:
            stream_online = self.ffmpeg.is_alive()
        
        return {
            'running': self.is_running,
            'stream_online': stream_online,
            'fps': round(self.current_fps, 1),
            'start_time': self.start_time,
            'uptime_seconds': uptime
        }
    
    def get_current_metrics(self) -> dict:
        """
        Vrátí aktuální metriky.
        
        Returns:
            Dict s metrikami
        """
        if not self.analytics:
            return {
                'timestamp': datetime.now(),
                'occupancy': 0,
                'crossings_this_run': 0,
                'crossings_last_1m': 0,
                'crossings_last_10m': 0
            }
        
        return self.analytics.get_metrics()
    
    def get_latest_frame(self) -> Optional[np.ndarray]:
        """
        Vrátí poslední frame s vizualizací.
        
        Returns:
            BGR frame nebo None
        """
        with self.frame_lock:
            if self.latest_frame is not None:
                return self.latest_frame.copy()
            return None
    
    def _draw_visualizations(self, frame: np.ndarray, detections: list, tracks: list) -> np.ndarray:
        """
        Vykreslí detekce, tracky, ROI a line crossing do framu.
        
        Args:
            frame: Vstupní BGR frame
            detections: Seznam bboxů detekcí
            tracks: Seznam tracků
            
        Returns:
            Frame s vizualizací
        """
        import cv2
        from datetime import datetime
        
        h, w = frame.shape[:2]
        
        # ROI obdélník (tenčí, šedá)
        if self.analytics.roi_filter.roi_rect:
            x1, y1, x2, y2 = self.analytics.roi_filter.roi_rect
            cv2.rectangle(frame, (x1, y1), (x2, y2), (128, 128, 128), 1)
        
        # Line crossing (žlutá, tlustší)
        if self.analytics.line_counter.line:
            p1, p2 = self.analytics.line_counter.line
            cv2.line(frame, p1, p2, (0, 255, 255), 3)
        
        # Detekce (surové) - zelené obdélníky SILNÉ (3px)
        logger.debug(f"VIZUALIZACE: Kreslim {len(detections)} detekcí")
        for bbox in detections:
            x1, y1, x2, y2 = map(int, bbox)
            logger.debug(f"  Detekce bbox: ({x1}, {y1}) - ({x2}, {y2})")
            # Bounding box - ZELENÝ SILNÝ
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        # Tracky (confirmed) - ČERVENÉ boxy SILNÉ (3px) pro viditelnost
        logger.debug(f"VIZUALIZACE: Kreslim {len(tracks)} tracků")
        for track in tracks:
            x1, y1, x2, y2 = map(int, track.bbox)
            logger.debug(f"  Track bbox: ({x1}, {y1}) - ({x2}, {y2})")
            # Bounding box - ČERVENÝ SILNÝ
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
        
        # Dashboard panel vlevo nahoře - jako na screenshotu
        metrics = self.analytics.get_metrics()
        panel_width = 280
        panel_height = 110
        
        # Černé pozadí panelu
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (10 + panel_width, 10 + panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Ohraničení panelu
        cv2.rectangle(frame, (10, 10), (10 + panel_width, 10 + panel_height), (255, 255, 255), 1)
        
        # Nadpis - bílý
        cv2.putText(frame, "SKI CAM ANALYTICS", (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Detected - zelený text
        detected_text = f"Detected: {metrics['occupancy']}"
        cv2.putText(frame, detected_text, (20, 62), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Čas - bílý, menší
        time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, f"Time: {time_str}", (20, 88), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
        
        # FPS a crossings - menší text
        cv2.putText(frame, f"FPS: {self.current_fps:.1f} | Crossings: {metrics['crossings_this_run']}", 
                   (20, 106), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1, cv2.LINE_AA)
        
        return frame
