"""
Jednoduchý SORT-like tracker pro tracking osob.

Využívá IoU matching pro asociaci detekcí s existujícími tracky.
"""
import numpy as np
from typing import List, Tuple, Optional
from collections import deque


def iou(bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
    """
    Vypočítá IoU (Intersection over Union) mezi dvěma boxy.
    
    Args:
        bbox1: (x1, y1, x2, y2)
        bbox2: (x1, y1, x2, y2)
        
    Returns:
        IoU score (0-1)
    """
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    
    # Intersection
    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)
    
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    
    # Union
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - inter_area
    
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area


class Track:
    """
    Jeden track (sledovaná osoba).
    """
    
    _next_id = 1  # Globální counter pro unikátní ID
    
    def __init__(self, bbox: Tuple[int, int, int, int], max_age: int = 30):
        """
        Args:
            bbox: Počáteční bounding box (x1, y1, x2, y2)
            max_age: Max počet framů bez detekce než track zanikne
        """
        self.id = Track._next_id
        Track._next_id += 1
        
        self.bbox = bbox
        self.age = 0  # Počet framů od poslední detekce
        self.hits = 1  # Počet úspěšných matchů
        self.max_age = max_age
        
        # Historie pozic (pro line crossing)
        self.trajectory = deque(maxlen=10)
        self.trajectory.append(self._get_center(bbox))
    
    def _get_center(self, bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
        """Vrátí střed bboxu."""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)
    
    def update(self, bbox: Tuple[int, int, int, int]):
        """
        Aktualizuje track novou detekcí.
        
        Args:
            bbox: Nový bounding box
        """
        self.bbox = bbox
        self.age = 0
        self.hits += 1
        self.trajectory.append(self._get_center(bbox))
    
    def predict(self):
        """
        Predikce další pozice (pro SORT by se použil Kalman filter, 
        tady jen ponecháme bbox stejný).
        """
        self.age += 1
    
    def is_confirmed(self, min_hits: int = 3) -> bool:
        """Vrátí True pokud má track dostatek hitů aby byl považován za validní."""
        return self.hits >= min_hits
    
    def is_dead(self) -> bool:
        """Vrátí True pokud track přestal být trackován (starý)."""
        return self.age > self.max_age
    
    def get_center(self) -> Tuple[int, int]:
        """Vrátí aktuální střed tracku."""
        return self._get_center(self.bbox)
    
    def get_trajectory(self) -> List[Tuple[int, int]]:
        """Vrátí historii pozic."""
        return list(self.trajectory)


class SORTTracker:
    """
    Jednoduchý tracker inspirovaný SORT algoritmem.
    
    Využívá IoU matching pro asociaci detekcí s tracky.
    """
    
    def __init__(self, max_age: int = 30, min_hits: int = 3, iou_threshold: float = 0.3):
        """
        Args:
            max_age: Max počet framů bez detekce
            min_hits: Min počet hitů pro confirmed track
            iou_threshold: IoU threshold pro matching
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        
        self.tracks: List[Track] = []
    
    def update(self, detections: List[Tuple[int, int, int, int]]) -> List[Track]:
        """
        Aktualizuje tracker s novými detekcemi.
        
        Args:
            detections: Seznam bboxů (x1, y1, x2, y2)
            
        Returns:
            Seznam aktivních confirmed tracků
        """
        # Predikce pro všechny existující tracky
        for track in self.tracks:
            track.predict()
        
        # Matching detekcí s tracky
        matched_tracks, unmatched_detections = self._match(detections)
        
        # Aktualizovat matchované tracky
        for track_idx, det_idx in matched_tracks:
            self.tracks[track_idx].update(detections[det_idx])
        
        # Vytvořit nové tracky pro nematchované detekce
        for det_idx in unmatched_detections:
            new_track = Track(detections[det_idx], max_age=self.max_age)
            self.tracks.append(new_track)
        
        # Odstranit mrtvé tracky
        self.tracks = [t for t in self.tracks if not t.is_dead()]
        
        # Vrátit pouze confirmed tracky
        confirmed = [t for t in self.tracks if t.is_confirmed(self.min_hits)]
        return confirmed
    
    def _match(self, detections: List[Tuple[int, int, int, int]]) -> Tuple[List[Tuple[int, int]], List[int]]:
        """
        Matchování detekcí s tracky pomocí IoU.
        
        Args:
            detections: Seznam bboxů
            
        Returns:
            (matched_pairs, unmatched_detection_indices)
            matched_pairs: [(track_idx, detection_idx), ...]
            unmatched_detection_indices: [det_idx, ...]
        """
        if len(self.tracks) == 0:
            return [], list(range(len(detections)))
        
        if len(detections) == 0:
            return [], []
        
        # Vypočítat IoU matici
        iou_matrix = np.zeros((len(self.tracks), len(detections)))
        
        for t, track in enumerate(self.tracks):
            for d, det_bbox in enumerate(detections):
                iou_matrix[t, d] = iou(track.bbox, det_bbox)
        
        # Greedy matching (nejlepší IoU nejdřív)
        matched_tracks = []
        matched_detections = []
        
        # Seřadit všechny páry podle IoU (od nejvyššího)
        pairs = []
        for t in range(len(self.tracks)):
            for d in range(len(detections)):
                if iou_matrix[t, d] >= self.iou_threshold:
                    pairs.append((t, d, iou_matrix[t, d]))
        
        pairs.sort(key=lambda x: x[2], reverse=True)
        
        # Greedily přiřadit
        used_tracks = set()
        used_detections = set()
        
        for t, d, _ in pairs:
            if t not in used_tracks and d not in used_detections:
                matched_tracks.append((t, d))
                used_tracks.add(t)
                used_detections.add(d)
        
        # Nematchované detekce
        unmatched_detections = [d for d in range(len(detections)) if d not in used_detections]
        
        return matched_tracks, unmatched_detections
    
    def get_active_tracks(self) -> List[Track]:
        """Vrátí všechny aktivní confirmed tracky."""
        return [t for t in self.tracks if t.is_confirmed(self.min_hits)]
    
    def reset(self):
        """Resetuje tracker (vymaže všechny tracky)."""
        self.tracks = []
        Track._next_id = 1
