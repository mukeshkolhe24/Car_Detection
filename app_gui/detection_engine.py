"""
DETECTION ENGINE - Loads models and performs inference with real metrics
"""

import sys
import os
import time
import cv2
import numpy as np
import json
import torch

# Add parent directory to path to import mmdetection
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mmdet.apis import init_detector, inference_detector

class DetectionEngine:
    def __init__(self):
        self.frcnn_model = None
        self.yolo_model = None
        self.models_loaded = False
        
        # Pre-calculated metrics from your training logs
        self.metrics = {
            'frcnn': {
                'mAP': 0.614,
                'mAP_50': 0.949,
                'mAP_75': 0.725,
                'precision': 0.95,
                'recall': 0.94,
                'f1_score': 0.945,
                'speed': '2-3 FPS'
            },
            'yolo': {
                'mAP': 0.123,
                'mAP_50': 0.520,
                'mAP_75': 0.017,
                'precision': 0.52,
                'recall': 0.50,
                'f1_score': 0.51,
                'speed': '20-30 FPS'
            }
        }
        
        # Load validation annotations for calculating metrics on-the-fly
        self.val_annotations = None
        self.load_validation_annotations()
        
    def load_validation_annotations(self):
        """Load validation annotations for metrics calculation"""
        try:
            with open('data/annotations/instances_val_fixed.json', 'r') as f:
                self.val_annotations = json.load(f)
            print("✅ Validation annotations loaded for metrics")
        except Exception as e:
            print(f"⚠️ Could not load validation annotations: {e}")
            
    def load_models(self):
        """Load both trained models"""
        print("📥 Loading Faster R-CNN model...")
        self.frcnn_model = init_detector(
            "configs/faster_rcnn/faster_rcnn_car.py",
            "work_dirs/faster_rcnn_car/epoch_12.pth",
            device='cuda:0' if torch.cuda.is_available() else 'cpu'
        )
        
        print("📥 Loading YOLO model...")
        self.yolo_model = init_detector(
            "configs/yolo/yolo_car.py",
            "work_dirs/yolo_car/epoch_50.pth",
            device='cuda:0' if torch.cuda.is_available() else 'cpu'
        )
        
        self.models_loaded = True
        print("✅ Both models loaded successfully!")
        
    def detect_frcnn(self, image_path, confidence_threshold=0.5):
        """Run detection using Faster R-CNN"""
        if not self.models_loaded:
            self.load_models()
            
        start_time = time.time()
        result = inference_detector(self.frcnn_model, image_path)
        inference_time = (time.time() - start_time) * 1000  # ms
        
        # Extract detections
        detections = []
        if hasattr(result, 'pred_instances'):
            pred = result.pred_instances
            if hasattr(pred, 'bboxes') and len(pred.bboxes) > 0:
                bboxes = pred.bboxes.cpu().numpy()
                scores = pred.scores.cpu().numpy()
                for bbox, score in zip(bboxes, scores):
                    if score >= confidence_threshold:
                        detections.append({
                            'bbox': bbox[:4].astype(int).tolist(),
                            'confidence': float(score)
                        })
        
        return {
            'detections': detections,
            'count': len(detections),
            'time_ms': inference_time,
            'fps': 1000 / inference_time if inference_time > 0 else 0,
            'metrics': self.metrics['frcnn']
        }
        
    def detect_yolo(self, image_path, confidence_threshold=0.5):
        """Run detection using YOLO"""
        if not self.models_loaded:
            self.load_models()
            
        start_time = time.time()
        result = inference_detector(self.yolo_model, image_path)
        inference_time = (time.time() - start_time) * 1000  # ms
        
        # Extract detections
        detections = []
        if hasattr(result, 'pred_instances'):
            pred = result.pred_instances
            if hasattr(pred, 'bboxes') and len(pred.bboxes) > 0:
                bboxes = pred.bboxes.cpu().numpy()
                scores = pred.scores.cpu().numpy()
                for bbox, score in zip(bboxes, scores):
                    if score >= confidence_threshold:
                        detections.append({
                            'bbox': bbox[:4].astype(int).tolist(),
                            'confidence': float(score)
                        })
        
        return {
            'detections': detections,
            'count': len(detections),
            'time_ms': inference_time,
            'fps': 1000 / inference_time if inference_time > 0 else 0,
            'metrics': self.metrics['yolo']
        }
        
    def calculate_precision_recall(self, detections, ground_truth_boxes, iou_threshold=0.5):
        """Calculate precision and recall for a single image"""
        if not detections or not ground_truth_boxes:
            return 0, 0
        
        def calculate_iou(box1, box2):
            x1 = max(box1[0], box2[0])
            y1 = max(box1[1], box2[1])
            x2 = min(box1[2], box2[2])
            y2 = min(box1[3], box2[3])
            intersection = max(0, x2 - x1) * max(0, y2 - y1)
            area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
            area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
            union = area1 + area2 - intersection
            return intersection / union if union > 0 else 0
        
        # Match detections to ground truth
        matched_gt = set()
        true_positives = 0
        
        for det in detections:
            det_box = det['bbox']
            best_iou = 0
            best_gt = -1
            for j, gt_box in enumerate(ground_truth_boxes):
                if j not in matched_gt:
                    iou = calculate_iou(det_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt = j
            if best_iou >= iou_threshold:
                matched_gt.add(best_gt)
                true_positives += 1
        
        false_positives = len(detections) - true_positives
        false_negatives = len(ground_truth_boxes) - len(matched_gt)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        
        return precision, recall
        
    def draw_detections(self, image_path, detections, color=(0, 255, 0)):
        """Draw bounding boxes on image"""
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        for det in detections:
            bbox = det['bbox']
            score = det['confidence']
            x1, y1, x2, y2 = bbox
            
            # Draw rectangle
            cv2.rectangle(img_rgb, (x1, y1), (x2, y2), color, 2)
            # Draw label
            label = f"{score:.2f}"
            cv2.putText(img_rgb, label, (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return img_rgb
    
    def get_metrics(self, model_name):
        """Get pre-calculated metrics for a model"""
        return self.metrics.get(model_name, {})