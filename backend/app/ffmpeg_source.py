"""
FFmpeg source - čte HLS stream a poskytuje raw frames.
"""
import subprocess
import numpy as np
import cv2
import logging
from threading import Thread, Event
from queue import Queue, Empty
from typing import Optional

logger = logging.getLogger(__name__)


class FFmpegSource:
    """
    Třída pro čtení HLS streamu pomocí FFmpeg.
    
    Spouští FFmpeg subprocess, který dekóduje HLS stream
    a poskytuje raw BGR frames jako numpy arrays.
    """
    
    def __init__(self, stream_url: str, fps: int = 8, width: int = 640, height: int = 480):
        """
        Args:
            stream_url: URL HLS streamu (.m3u8)
            fps: Cílový FPS (kolik snímků za sekundu zpracovávat)
            width: Šířka výstupního frame
            height: Výška výstupního frame
        """
        self.stream_url = stream_url
        self.fps = fps
        self.width = width
        self.height = height
        
        self.process: Optional[subprocess.Popen] = None
        self.frame_queue: Queue = Queue(maxsize=10)
        self.reader_thread: Optional[Thread] = None
        self.stop_event = Event()
        self.is_running = False
        
    def start(self):
        """Spustí FFmpeg a začne číst frames."""
        if self.is_running:
            logger.warning("FFmpegSource už běží")
            return
            
        logger.info(f"Spouštím FFmpeg pro stream: {self.stream_url}")
        
        # FFmpeg příkaz:
        # -rtsp_transport tcp: stabilnější připojení
        # -i: vstupní stream
        # -vf fps=X: vzorkovat na X FPS
        # -vf scale: změnit velikost
        # -f rawvideo: raw output
        # -pix_fmt bgr24: BGR formát (kompatibilní s OpenCV)
        # pipe:1: output na stdout
        
        ffmpeg_cmd = [
            'ffmpeg',
            '-loglevel', 'warning',
            '-i', self.stream_url,
            '-vf', f'fps={self.fps},scale={self.width}:{self.height}',
            '-f', 'rawvideo',
            '-pix_fmt', 'bgr24',
            'pipe:1'
        ]
        
        try:
            self.process = subprocess.Popen(
                ffmpeg_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=10**8
            )
            
            self.stop_event.clear()
            self.is_running = True
            
            # Spustit thread pro čtení frames
            self.reader_thread = Thread(target=self._read_frames, daemon=True)
            self.reader_thread.start()
            
            logger.info("FFmpeg úspěšně spuštěn")
            
        except FileNotFoundError:
            logger.error("FFmpeg není nainstalován nebo není v PATH!")
            raise
        except Exception as e:
            logger.error(f"Chyba při spouštění FFmpeg: {e}")
            raise
    
    def _read_frames(self):
        """Thread funkce pro čtení raw frames z FFmpeg stdout."""
        frame_size = self.width * self.height * 3  # BGR = 3 bytes per pixel
        
        while not self.stop_event.is_set() and self.process:
            try:
                # Přečíst raw bytes
                raw_frame = self.process.stdout.read(frame_size)
                
                if len(raw_frame) != frame_size:
                    logger.warning("FFmpeg stream skončil nebo se přerušil")
                    break
                
                # Převést na numpy array
                frame = np.frombuffer(raw_frame, dtype=np.uint8)
                frame = frame.reshape((self.height, self.width, 3))
                
                # Vložit do fronty (pokud je plná, zahodit nejstarší)
                try:
                    self.frame_queue.put(frame, timeout=0.1)
                except:
                    # Queue plná, přeskočit frame
                    pass
                    
            except Exception as e:
                if not self.stop_event.is_set():
                    logger.error(f"Chyba při čtení frame: {e}")
                break
        
        logger.info("FFmpeg reader thread ukončen")
        self.is_running = False
    
    def get_frame(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """
        Získá další frame z fronty.
        
        Args:
            timeout: Max čekání na frame (sekundy)
            
        Returns:
            BGR frame jako numpy array nebo None pokud není dostupný
        """
        try:
            return self.frame_queue.get(timeout=timeout)
        except Empty:
            return None
    
    def stop(self):
        """Zastaví FFmpeg a čtení frames."""
        if not self.is_running:
            return
            
        logger.info("Zastavuji FFmpeg...")
        
        self.stop_event.set()
        
        # Ukončit FFmpeg proces
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning("FFmpeg se neukončil, použiji kill")
                self.process.kill()
            except Exception as e:
                logger.error(f"Chyba při ukončování FFmpeg: {e}")
        
        # Počkat na reader thread
        if self.reader_thread and self.reader_thread.is_alive():
            self.reader_thread.join(timeout=2)
        
        # Vyprázdnit frontu
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except Empty:
                break
        
        self.is_running = False
        logger.info("FFmpeg zastaven")
    
    def is_alive(self) -> bool:
        """Vrátí True pokud FFmpeg běží a čte frames."""
        return self.is_running and self.process and self.process.poll() is None
