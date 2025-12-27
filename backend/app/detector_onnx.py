"""
YOLO ONNX detektor osob.
"""
import numpy as np
import cv2
import onnxruntime as ort
import logging
from typing import List, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


class Detection:
    """Jedna detekce."""
    def __init__(self, bbox: Tuple[int, int, int, int], confidence: float, class_id: int = 0):
        """
        Args:
            bbox: Bounding box (x1, y1, x2, y2)
            confidence: Confidence score (0-1)
            class_id: ID třídy (pro person = 0 v COCO)
        """
        self.bbox = bbox
        self.confidence = confidence
        self.class_id = class_id
    
    def __repr__(self):
        return f"Detection(bbox={self.bbox}, conf={self.confidence:.2f})"


class YOLODetector:
    """
    YOLO detektor využívající ONNX Runtime.
    
    Podporuje standardní YOLO ONNX export (YOLOv5, YOLOv8, YOLOv11).
    Detekuje pouze třídu "person" (class_id=0 v COCO datasetu).
    """
    
    def __init__(
        self,
        model_path: Path,
        conf_threshold: float = 0.4,
        iou_threshold: float = 0.45,
        input_size: Tuple[int, int] = (640, 640)
    ):
        """
        Args:
            model_path: Cesta k ONNX modelu
            conf_threshold: Confidence threshold pro filtrování detekcí
            iou_threshold: IoU threshold pro NMS
            input_size: Velikost vstupu modelu (width, height)
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.input_size = input_size
        
        # Načíst ONNX model
        logger.info(f"Načítám ONNX model: {model_path}")
        
        if not model_path.exists():
            raise FileNotFoundError(f"ONNX model nenalezen: {model_path}")
        
        # ONNX Runtime session (CPU)
        # TODO: Pro GPU použít: providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session = ort.InferenceSession(
            str(model_path),
            providers=['CPUExecutionProvider']
        )
        
        # Získat input/output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        
        logger.info(f"Model načten, input: {self.input_name}, outputs: {self.output_names}")
        logger.info(f"Providers: {self.session.get_providers()}")
    
    def preprocess(self, frame: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """
        Preprocessování frame pro YOLO.
        
        Args:
            frame: BGR obrázek
            
        Returns:
            (preprocessed_tensor, scale, (pad_w, pad_h))
        """
        # Resize s udržením aspect ratio (letterbox)
        img_h, img_w = frame.shape[:2]
        input_w, input_h = self.input_size
        
        scale = min(input_w / img_w, input_h / img_h)
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)
        
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Padding
        pad_w = (input_w - new_w) // 2
        pad_h = (input_h - new_h) // 2
        
        padded = np.full((input_h, input_w, 3), 114, dtype=np.uint8)
        padded[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized
        
        # BGR to RGB
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1] a transpose to CHW
        tensor = rgb.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))  # HWC -> CHW
        tensor = np.expand_dims(tensor, axis=0)  # Add batch dimension
        
        return tensor, scale, (pad_w, pad_h)
    
    def postprocess(
        self,
        outputs: List[np.ndarray],
        scale: float,
        pads: Tuple[int, int],
        orig_shape: Tuple[int, int]
    ) -> List[Detection]:
        """
        Postprocessování YOLO výstupu.
        
        Args:
            outputs: Raw výstup z ONNX modelu
            scale: Scale použitý při resize
            pads: Padding (pad_w, pad_h)
            orig_shape: Původní velikost frame (height, width)
            
        Returns:
            Seznam detekcí
        """
        # YOLO output shape: (1, 84, 8400) pro YOLOv8 nebo podobné
        # První 4 hodnoty = bbox (x_center, y_center, w, h)
        # Zbytek = class confidences
        
        output = outputs[0]
        
        # Transpose pokud je třeba: (1, 84, 8400) -> (8400, 84)
        if len(output.shape) == 3:
            output = output[0].transpose()  # (num_boxes, 84)
        
        # Extrahovat bboxes a scores
        boxes = output[:, :4]  # (num_boxes, 4)
        scores = output[:, 4:]  # (num_boxes, 80) pro 80 tříd COCO
        
        # Získat pouze class 0 (person)
        person_scores = scores[:, 0]
        
        # Filtrovat podle confidence
        mask = person_scores > self.conf_threshold
        boxes = boxes[mask]
        person_scores = person_scores[mask]
        
        if len(boxes) == 0:
            return []
        
        # Konverze z (x_center, y_center, w, h) na (x1, y1, x2, y2)
        x_center, y_center, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = x_center - w / 2
        y1 = y_center - h / 2
        x2 = x_center + w / 2
        y2 = y_center + h / 2
        
        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)
        
        # NMS (Non-Maximum Suppression)
        indices = self._nms(boxes_xyxy, person_scores, self.iou_threshold)
        
        boxes_xyxy = boxes_xyxy[indices]
        person_scores = person_scores[indices]
        
        # Převést souřadnice zpět na originální frame
        pad_w, pad_h = pads
        orig_h, orig_w = orig_shape
        
        detections = []
        for box, score in zip(boxes_xyxy, person_scores):
            # Odečíst padding a škálovat zpět
            x1 = int((box[0] - pad_w) / scale)
            y1 = int((box[1] - pad_h) / scale)
            x2 = int((box[2] - pad_w) / scale)
            y2 = int((box[3] - pad_h) / scale)
            
            # Clamp do rozměrů frame
            x1 = max(0, min(x1, orig_w))
            y1 = max(0, min(y1, orig_h))
            x2 = max(0, min(x2, orig_w))
            y2 = max(0, min(y2, orig_h))
            
            detections.append(Detection(
                bbox=(x1, y1, x2, y2),
                confidence=float(score),
                class_id=0
            ))
        
        return detections
    
    def _nms(self, boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> List[int]:
        """
        Non-Maximum Suppression.
        
        Args:
            boxes: Array bboxů (N, 4) ve formátu (x1, y1, x2, y2)
            scores: Array scores (N,)
            iou_threshold: IoU threshold
            
        Returns:
            Indexy vybraných boxů
        """
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]  # Seřadit od nejvyššího score
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            # Spočítat IoU s ostatními boxy
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            intersection = w * h
            
            iou = intersection / (areas[i] + areas[order[1:]] - intersection)
            
            # Ponechat pouze boxy s IoU pod threshold
            indices = np.where(iou <= iou_threshold)[0]
            order = order[indices + 1]
        
        return keep
    
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Detekce osob ve frame.
        
        Args:
            frame: BGR obrázek
            
        Returns:
            Seznam detekcí
        """
        orig_h, orig_w = frame.shape[:2]
        
        # Preprocessing
        input_tensor, scale, pads = self.preprocess(frame)
        
        # Inference
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
        
        # Postprocessing
        detections = self.postprocess(outputs, scale, pads, (orig_h, orig_w))
        
        return detections
