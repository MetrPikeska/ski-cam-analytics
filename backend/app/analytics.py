"""
Analytics modul - počítání occupancy a line crossings.
"""
import numpy as np
from typing import List, Tuple, Optional, Set
from collections import deque
from datetime import datetime, timedelta
import logging

from .tracker import Track

logger = logging.getLogger(__name__)


def point_side_of_line(
    point: Tuple[int, int],
    line_p1: Tuple[int, int],
    line_p2: Tuple[int, int]
) -> int:
    """
    Určí na které straně čáry se bod nachází.
    
    Args:
        point: (x, y)
        line_p1: První bod čáry (x, y)
        line_p2: Druhý bod čáry (x, y)
        
    Returns:
        > 0: vpravo od čáry
        < 0: vlevo od čáry
        = 0: na čáře
    """
    x, y = point
    x1, y1 = line_p1
    x2, y2 = line_p2
    
    # Cross product
    return (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)


class LineCrossingCounter:
    """
    Počítá crossing přes definovanou čáru.
    """
    
    def __init__(self, line: Optional[List[Tuple[int, int]]] = None):
        """
        Args:
            line: Čára definovaná dvěma body [(x1, y1), (x2, y2)] nebo None
        """
        self.line = line
        self.crossed_track_ids: Set[int] = set()  # Tracky které už překročily
        
        # Historie crossingů pro časové intervaly
        self.crossing_times: deque = deque(maxlen=1000)
    
    def update(self, tracks: List[Track]) -> int:
        """
        Aktualizuje counter s aktivními tracky.
        
        Args:
            tracks: Seznam aktivních tracků
            
        Returns:
            Počet nových crossingů v tomto framu
        """
        if self.line is None:
            return 0
        
        new_crossings = 0
        line_p1, line_p2 = self.line
        
        for track in tracks:
            # Přeskočit tracky které už crossed
            if track.id in self.crossed_track_ids:
                continue
            
            # Potřebujeme alespoň 2 body v trajektorii
            trajectory = track.get_trajectory()
            if len(trajectory) < 2:
                continue
            
            # Zkontrolovat poslední dva body
            prev_point = trajectory[-2]
            curr_point = trajectory[-1]
            
            prev_side = point_side_of_line(prev_point, line_p1, line_p2)
            curr_side = point_side_of_line(curr_point, line_p1, line_p2)
            
            # Crossing = změna znaménka (přešel z jedné strany na druhou)
            if prev_side * curr_side < 0:
                self.crossed_track_ids.add(track.id)
                self.crossing_times.append(datetime.now())
                new_crossings += 1
                logger.debug(f"Track {track.id} crossed the line")
        
        return new_crossings
    
    def get_total_crossings(self) -> int:
        """Vrátí celkový počet crossingů od startu."""
        return len(self.crossing_times)
    
    def get_crossings_last_n_minutes(self, minutes: int) -> int:
        """
        Vrátí počet crossingů za posledních N minut.
        
        Args:
            minutes: Počet minut
            
        Returns:
            Počet crossingů
        """
        if not self.crossing_times:
            return 0
        
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        count = sum(1 for t in self.crossing_times if t >= cutoff_time)
        return count
    
    def reset(self):
        """Reset counteru."""
        self.crossed_track_ids.clear()
        self.crossing_times.clear()


class ROIFilter:
    """
    Filtruje detekce podle Region of Interest.
    """
    
    def __init__(self, roi_rect: Optional[Tuple[int, int, int, int]] = None):
        """
        Args:
            roi_rect: ROI obdélník (x1, y1, x2, y2) nebo None pro celý frame
        """
        self.roi_rect = roi_rect
    
    def is_inside_roi(self, bbox: Tuple[int, int, int, int]) -> bool:
        """
        Zkontroluje zda je bbox uvnitř ROI.
        
        Args:
            bbox: Bounding box (x1, y1, x2, y2)
            
        Returns:
            True pokud střed bboxu je v ROI
        """
        if self.roi_rect is None:
            return True
        
        # Střed bboxu
        x_center = (bbox[0] + bbox[2]) // 2
        y_center = (bbox[1] + bbox[3]) // 2
        
        roi_x1, roi_y1, roi_x2, roi_y2 = self.roi_rect
        
        return (roi_x1 <= x_center <= roi_x2) and (roi_y1 <= y_center <= roi_y2)
    
    def filter_detections(self, bboxes: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
        """
        Filtruje detekce podle ROI.
        
        Args:
            bboxes: Seznam bboxů
            
        Returns:
            Filtrovaný seznam bboxů
        """
        if self.roi_rect is None:
            return bboxes
        
        return [bbox for bbox in bboxes if self.is_inside_roi(bbox)]


class Analytics:
    """
    Hlavní analytics třída - kombinuje tracking a counting.
    """
    
    def __init__(
        self,
        roi_rect: Optional[Tuple[int, int, int, int]] = None,
        line: Optional[List[Tuple[int, int]]] = None
    ):
        """
        Args:
            roi_rect: ROI obdélník (x1, y1, x2, y2) nebo None
            line: Line crossing čára [(x1, y1), (x2, y2)] nebo None
        """
        self.roi_filter = ROIFilter(roi_rect)
        self.line_counter = LineCrossingCounter(line)
        
        self.current_occupancy = 0
        self.last_update_time = datetime.now()
    
    def update(self, tracks: List[Track]):
        """
        Aktualizuje analytics s novými tracky.
        
        Args:
            tracks: Seznam aktivních confirmed tracků
        """
        # Occupancy = počet confirmed tracků
        self.current_occupancy = len(tracks)
        
        # Line crossing
        self.line_counter.update(tracks)
        
        self.last_update_time = datetime.now()
    
    def get_metrics(self) -> dict:
        """
        Vrátí aktuální metriky.
        
        Returns:
            Dict s metrikami
        """
        return {
            'timestamp': self.last_update_time,
            'occupancy': self.current_occupancy,
            'crossings_this_run': self.line_counter.get_total_crossings(),
            'crossings_last_1m': self.line_counter.get_crossings_last_n_minutes(1),
            'crossings_last_10m': self.line_counter.get_crossings_last_n_minutes(10)
        }
    
    def reset(self):
        """Reset analytics."""
        self.line_counter.reset()
        self.current_occupancy = 0
