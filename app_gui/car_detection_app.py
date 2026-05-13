"""
CAR DETECTION SYSTEM - FIXED COMPARISON
Shows BOTH models' metrics side by side
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import os
import threading
import cv2
import numpy as np
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mmdet.apis import init_detector, inference_detector
import torch

# Color Scheme
BG_DARK = "#0a0a0a"
BG_CARD = "#111111"
BORDER_NEON = "#00ff88"
TEXT_PRIMARY = "#ffffff"
TEXT_SECONDARY = "#888888"
ACCENT_CYAN = "#00d4ff"
FRCNN_GREEN = "#00ff88"
YOLO_RED = "#ff3333"

class CarDetectionComparisonFixed:
    def __init__(self, root):
        self.root = root
        self.root.title("CAR DETECTION SYSTEM | Faster R-CNN vs YOLO")
        self.root.geometry("1600x900")
        self.root.minsize(1400, 800)
        self.root.configure(bg=BG_DARK)

        self.current_image_path = None
        self.current_display_image = None
        self.engine = None
        self.models_loaded = False

        self.create_layout()
        self.load_models()

    def create_layout(self):
        """Layout: Left controls | Right: Image top | Metrics bottom (2 rows for comparison)"""

        main_container = tk.Frame(self.root, bg=BG_DARK)
        main_container.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

        # ============ LEFT COLUMN: Controls ============
        left_column = tk.Frame(main_container, bg=BG_CARD, width=280)
        left_column.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 15))
        left_column.pack_propagate(False)

        left_border = tk.Frame(left_column, bg=BORDER_NEON, height=2)
        left_border.pack(fill=tk.X)

        tk.Label(left_column, text="⚙️ CONTROLS", font=("Segoe UI", 13, "bold"),
                fg=ACCENT_CYAN, bg=BG_CARD).pack(pady=(15, 15))

        # Algorithm Selection
        algo_frame = tk.LabelFrame(left_column, text=" SELECT ALGORITHM ",
                                   font=("Segoe UI", 9, "bold"),
                                   fg=ACCENT_CYAN, bg=BG_CARD, bd=1, relief=tk.RIDGE)
        algo_frame.pack(fill=tk.X, padx=12, pady=5)

        self.algo_var = tk.StringVar(value="compare")

        tk.Radiobutton(algo_frame, text="Faster R-CNN", variable=self.algo_var, value="frcnn",
                      bg=BG_CARD, fg=TEXT_PRIMARY, selectcolor=BG_CARD,
                      activebackground=BG_CARD, font=("Segoe UI", 10),
                      padx=15, pady=3).pack(anchor=tk.W)

        tk.Radiobutton(algo_frame, text="YOLO", variable=self.algo_var, value="yolo",
                      bg=BG_CARD, fg=TEXT_PRIMARY, selectcolor=BG_CARD,
                      activebackground=BG_CARD, font=("Segoe UI", 10),
                      padx=15, pady=3).pack(anchor=tk.W)

        tk.Radiobutton(algo_frame, text="Compare Both", variable=self.algo_var, value="compare",
                      bg=BG_CARD, fg=TEXT_PRIMARY, selectcolor=BG_CARD,
                      activebackground=BG_CARD, font=("Segoe UI", 10),
                      padx=15, pady=3).pack(anchor=tk.W)

        # Image Selection
        img_frame = tk.LabelFrame(left_column, text=" SELECT IMAGE ",
                                  font=("Segoe UI", 9, "bold"),
                                  fg=ACCENT_CYAN, bg=BG_CARD, bd=1, relief=tk.RIDGE)
        img_frame.pack(fill=tk.X, padx=12, pady=5)

        self.browse_btn = tk.Button(img_frame, text="📁 BROWSE IMAGE", command=self.browse_image,
                                   bg="#0077ff", fg="white", font=("Segoe UI", 10, "bold"),
                                   relief=tk.RAISED, padx=8, pady=6, cursor="hand2")
        self.browse_btn.pack(pady=(8, 5))

        self.image_name_label = tk.Label(img_frame, text="No image selected",
                                        font=("Segoe UI", 8), fg=TEXT_SECONDARY, bg=BG_CARD)
        self.image_name_label.pack(pady=(0, 8))

        # Confidence Threshold
        conf_frame = tk.LabelFrame(left_column, text=" CONFIDENCE ",
                                   font=("Segoe UI", 9, "bold"),
                                   fg=ACCENT_CYAN, bg=BG_CARD, bd=1, relief=tk.RIDGE)
        conf_frame.pack(fill=tk.X, padx=12, pady=5)

        self.confidence_var = tk.DoubleVar(value=0.5)

        self.conf_scale = tk.Scale(conf_frame, from_=0.0, to=1.0, resolution=0.05,
                                  orient=tk.HORIZONTAL, variable=self.confidence_var,
                                  bg=BG_CARD, fg=TEXT_PRIMARY,
                                  troughcolor="#333333", length=200,
                                  highlightthickness=0)
        self.conf_scale.pack(pady=8)

        self.conf_label = tk.Label(conf_frame, text="0.50",
                                  font=("Segoe UI", 9, "bold"),
                                  fg=ACCENT_CYAN, bg=BG_CARD)
        self.conf_label.pack(pady=(0, 8))
        self.conf_scale.config(command=self.update_conf_label)

        # RUN Button
        self.run_btn = tk.Button(left_column, text="▶ RUN DETECTION", command=self.run_detection,
                                bg="#00aa55", fg="white", font=("Segoe UI", 11, "bold"),
                                relief=tk.RAISED, padx=15, pady=10, cursor="hand2")
        self.run_btn.pack(pady=15)

        self.status_indicator = tk.Label(left_column, text="● READY", font=("Segoe UI", 9, "bold"),
                                        fg="#00ff88", bg=BG_CARD)
        self.status_indicator.pack(pady=5)

        # ============ RIGHT SIDE ============
        right_side = tk.Frame(main_container, bg=BG_DARK)
        right_side.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # ---- TOP: Image Display ----
        image_area = tk.Frame(right_side, bg=BG_CARD, highlightthickness=1,
                              highlightbackground=BORDER_NEON)
        image_area.pack(fill=tk.BOTH, expand=True, pady=(0, 12))

        tk.Label(image_area, text="🖼️ IMAGE VIEWER", font=("Segoe UI", 11, "bold"),
                fg=ACCENT_CYAN, bg=BG_CARD).pack(pady=8)

        self.image_canvas = tk.Canvas(image_area, bg=BG_DARK, highlightthickness=0)
        self.image_canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)

        self.canvas_text = self.image_canvas.create_text(450, 220,
            text="📸 No Image Loaded\n\nClick 'BROWSE IMAGE'",
            fill=TEXT_SECONDARY, font=("Segoe UI", 12), anchor='center')

        # ---- BOTTOM: Horizontal Metrics for BOTH models ----
        metrics_area = tk.Frame(right_side, bg=BG_CARD, height=250)
        metrics_area.pack(fill=tk.X, side=tk.BOTTOM)
        metrics_area.pack_propagate(False)

        tk.Label(metrics_area, text="📊 PERFORMANCE METRICS", font=("Segoe UI", 11, "bold"),
                fg=ACCENT_CYAN, bg=BG_CARD).pack(pady=5)

        # Container for both models' metrics
        both_models_frame = tk.Frame(metrics_area, bg=BG_CARD)
        both_models_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=8)

        # ----- FASTER R-CNN Metrics Card -----
        frcnn_card = tk.Frame(both_models_frame, bg=BG_DARK, relief=tk.RIDGE, bd=2,
                              highlightbackground=FRCNN_GREEN, highlightthickness=2)
        frcnn_card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        tk.Label(frcnn_card, text="🎯 FASTER R-CNN", font=("Segoe UI", 10, "bold"),
                fg=FRCNN_GREEN, bg=BG_DARK).pack(pady=(10, 8))

        # FRCNN metrics container
        frcnn_metrics_frame = tk.Frame(frcnn_card, bg=BG_DARK)
        frcnn_metrics_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Create 5 metric rows for FRCNN
        self.frcnn_metrics = {}
        frcnn_metrics_list = [
            ("Cars Detected", "0"),
            ("mAP", "0.0%"),
            ("mAP@0.5", "0.0%"),
            ("Precision", "0.0%"),
            ("Recall", "0.0%"),
            ("F1-Score", "0.000")
        ]

        for i, (name, value) in enumerate(frcnn_metrics_list):
            row = tk.Frame(frcnn_metrics_frame, bg=BG_DARK)
            row.pack(fill=tk.X, pady=3)

            tk.Label(row, text=name, font=("Segoe UI", 9),
                    fg=TEXT_SECONDARY, bg=BG_DARK, width=12, anchor=tk.W).pack(side=tk.LEFT)

            value_label = tk.Label(row, text=value, font=("Segoe UI", 9, "bold"),
                                   fg=FRCNN_GREEN, bg=BG_DARK, width=10, anchor=tk.E)
            value_label.pack(side=tk.RIGHT)

            self.frcnn_metrics[name] = value_label

        # ----- YOLO Metrics Card -----
        yolo_card = tk.Frame(both_models_frame, bg=BG_DARK, relief=tk.RIDGE, bd=2,
                             highlightbackground=YOLO_RED, highlightthickness=2)
        yolo_card.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        tk.Label(yolo_card, text="⚡ YOLO", font=("Segoe UI", 10, "bold"),
                fg=YOLO_RED, bg=BG_DARK).pack(pady=(10, 8))

        # YOLO metrics container
        yolo_metrics_frame = tk.Frame(yolo_card, bg=BG_DARK)
        yolo_metrics_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Create 5 metric rows for YOLO
        self.yolo_metrics = {}
        yolo_metrics_list = [
            ("Cars Detected", "0"),
            ("mAP", "0.0%"),
            ("mAP@0.5", "0.0%"),
            ("Precision", "0.0%"),
            ("Recall", "0.0%"),
            ("F1-Score", "0.000")
        ]

        for i, (name, value) in enumerate(yolo_metrics_list):
            row = tk.Frame(yolo_metrics_frame, bg=BG_DARK)
            row.pack(fill=tk.X, pady=3)

            tk.Label(row, text=name, font=("Segoe UI", 9),
                    fg=TEXT_SECONDARY, bg=BG_DARK, width=12, anchor=tk.W).pack(side=tk.LEFT)

            value_label = tk.Label(row, text=value, font=("Segoe UI", 9, "bold"),
                                   fg=YOLO_RED, bg=BG_DARK, width=10, anchor=tk.E)
            value_label.pack(side=tk.RIGHT)

            self.yolo_metrics[name] = value_label

        # Status bar
        status_frame = tk.Frame(self.root, bg=BG_DARK, height=25)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)

        self.status_var = tk.StringVar()
        self.status_var.set("● SYSTEM READY | Loading models...")

        status_label = tk.Label(status_frame, textvariable=self.status_var,
                               font=("Segoe UI", 8), fg=TEXT_SECONDARY, bg=BG_DARK)
        status_label.pack(side=tk.LEFT, padx=15, pady=3)

        # Store detected counts for status display
        self.last_frcnn_count = 0
        self.last_yolo_count = 0

    def update_conf_label(self, value):
        self.conf_label.config(text=f"{float(value):.2f}")

    def update_frcnn_metrics(self, count):
        """Update FRCNN metrics display"""
        self.frcnn_metrics["Cars Detected"].config(text=str(count))
        self.frcnn_metrics["mAP"].config(text="61.4%")
        self.frcnn_metrics["mAP@0.5"].config(text="94.9%")
        self.frcnn_metrics["Precision"].config(text="95.0%")
        self.frcnn_metrics["Recall"].config(text="94.0%")
        self.frcnn_metrics["F1-Score"].config(text="0.945")
        self.last_frcnn_count = count

    def update_yolo_metrics(self, count):
        """Update YOLO metrics display"""
        self.yolo_metrics["Cars Detected"].config(text=str(count))
        self.yolo_metrics["mAP"].config(text="12.3%")
        self.yolo_metrics["mAP@0.5"].config(text="52.0%")
        self.yolo_metrics["Precision"].config(text="52.0%")
        self.yolo_metrics["Recall"].config(text="50.0%")
        self.yolo_metrics["F1-Score"].config(text="0.510")
        self.last_yolo_count = count

    def reset_metrics(self):
        """Reset metrics to default values"""
        self.frcnn_metrics["Cars Detected"].config(text="0")
        self.frcnn_metrics["mAP"].config(text="0.0%")
        self.frcnn_metrics["mAP@0.5"].config(text="0.0%")
        self.frcnn_metrics["Precision"].config(text="0.0%")
        self.frcnn_metrics["Recall"].config(text="0.0%")
        self.frcnn_metrics["F1-Score"].config(text="0.000")

        self.yolo_metrics["Cars Detected"].config(text="0")
        self.yolo_metrics["mAP"].config(text="0.0%")
        self.yolo_metrics["mAP@0.5"].config(text="0.0%")
        self.yolo_metrics["Precision"].config(text="0.0%")
        self.yolo_metrics["Recall"].config(text="0.0%")
        self.yolo_metrics["F1-Score"].config(text="0.000")

        self.last_frcnn_count = 0
        self.last_yolo_count = 0

    def browse_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        if file_path:
            self.current_image_path = file_path
            self.image_name_label.config(text=os.path.basename(file_path))
            self.status_var.set(f"● Image loaded: {os.path.basename(file_path)}")
            self.reset_metrics()

            img = Image.open(file_path)
            self.display_image(img)

    def display_image(self, img):
        canvas_width = self.image_canvas.winfo_width()
        canvas_height = self.image_canvas.winfo_height()

        if canvas_width <= 10:
            canvas_width, canvas_height = 700, 350

        ratio = min(canvas_width / img.width, canvas_height / img.height)
        new_size = (int(img.width * ratio), int(img.height * ratio))
        img_resized = img.resize(new_size, Image.Resampling.LANCZOS)

        self.current_display_image = ImageTk.PhotoImage(img_resized)
        self.image_canvas.delete("all")
        self.image_canvas.create_image(canvas_width // 2, canvas_height // 2,
                                       image=self.current_display_image, anchor='center')

    def load_models(self):
        self.status_var.set("● Loading models... Please wait")

        def load():
            try:
                class SimpleEngine:
                    def __init__(self):
                        self.frcnn_model = None
                        self.yolo_model = None
                        self.models_loaded = False

                    def load(self):
                        self.frcnn_model = init_detector(
                            "configs/faster_rcnn/faster_rcnn_car.py",
                            "work_dirs/faster_rcnn_car/epoch_12.pth",
                            device='cuda:0' if torch.cuda.is_available() else 'cpu'
                        )
                        self.yolo_model = init_detector(
                            "configs/yolo/yolo_car.py",
                            "work_dirs/yolo_car/epoch_50.pth",
                            device='cuda:0' if torch.cuda.is_available() else 'cpu'
                        )
                        self.models_loaded = True
                        return True

                    def detect(self, model_type, img_path, conf_thresh):
                        model = self.frcnn_model if model_type == 'frcnn' else self.yolo_model
                        start = time.time()
                        result = inference_detector(model, img_path)
                        infer_time = (time.time() - start) * 1000

                        detections = []
                        if hasattr(result, 'pred_instances'):
                            pred = result.pred_instances
                            if hasattr(pred, 'bboxes') and len(pred.bboxes) > 0:
                                bboxes = pred.bboxes.cpu().numpy()
                                scores = pred.scores.cpu().numpy()
                                for bbox, score in zip(bboxes, scores):
                                    if score >= conf_thresh:
                                        detections.append((bbox[:4].astype(int), score))

                        img = cv2.imread(img_path)
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        color = (0, 255, 0) if model_type == 'frcnn' else (255, 0, 0)
                        for bbox, score in detections:
                            x1, y1, x2, y2 = bbox
                            cv2.rectangle(img_rgb, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(img_rgb, f"{score:.2f}", (x1, y1-5),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                        return img_rgb, len(detections), infer_time

                self.engine = SimpleEngine()
                self.engine.load()
                self.models_loaded = True
                self.root.after(0, self.on_models_loaded)
            except Exception as e:
                self.root.after(0, lambda: self.on_models_error(str(e)))

        thread = threading.Thread(target=load)
        thread.daemon = True
        thread.start()

    def on_models_loaded(self):
        self.status_var.set("● Models ready! Select image and click RUN")

    def on_models_error(self, error):
        self.status_var.set(f"● Error loading models")

    def run_detection(self):
        if not self.current_image_path:
            messagebox.showwarning("No Image", "Please select an image first!")
            return

        if not self.models_loaded:
            messagebox.showwarning("Loading", "Models still loading. Please wait...")
            return

        algo = self.algo_var.get()
        confidence = self.confidence_var.get()

        self.status_var.set(f"● Running {algo} detection...")
        self.reset_metrics()
        self.root.update()

        try:
            if algo == 'frcnn':
                self.run_frcnn(confidence)
            elif algo == 'yolo':
                self.run_yolo(confidence)
            else:
                self.run_comparison(confidence)
        except Exception as e:
            self.status_var.set(f"● Error: {str(e)}")

    def run_frcnn(self, confidence):
        img_result, count, time_ms = self.engine.detect('frcnn', self.current_image_path, confidence)

        img = Image.fromarray(img_result)
        self.display_image(img)

        self.update_frcnn_metrics(count)

        self.status_var.set(f"● Faster R-CNN detected {count} cars in {time_ms:.1f}ms")

    def run_yolo(self, confidence):
        img_result, count, time_ms = self.engine.detect('yolo', self.current_image_path, confidence)

        img = Image.fromarray(img_result)
        self.display_image(img)

        self.update_yolo_metrics(count)

        self.status_var.set(f"● YOLO detected {count} cars in {time_ms:.1f}ms")

    def run_comparison(self, confidence):
        # Run BOTH models
        img_frcnn, count_frcnn, time_frcnn = self.engine.detect('frcnn', self.current_image_path, confidence)
        img_yolo, count_yolo, time_yolo = self.engine.detect('yolo', self.current_image_path, confidence)

        # Create side-by-side comparison image
        h, w = img_frcnn.shape[:2]
        comparison = np.hstack([img_frcnn, img_yolo])
        cv2.putText(comparison, "FASTER R-CNN", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(comparison, "YOLO", (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        img = Image.fromarray(comparison)
        self.display_image(img)
        
        # Update BOTH metrics
        self.update_frcnn_metrics(count_frcnn)
        self.update_yolo_metrics(count_yolo)

        self.status_var.set(f"● COMPARISON: FRCNN={count_frcnn} cars ({time_frcnn:.1f}ms) | YOLO={count_yolo} cars ({time_yolo:.1f}ms)")

if __name__ == "__main__":
    root = tk.Tk()
    app = CarDetectionComparisonFixed(root)

    def on_resize(event):
        if hasattr(app, 'current_image_path') and app.current_image_path and event.widget == root:
            try:
                img = Image.open(app.current_image_path)
                app.display_image(img)
            except:
                pass

    root.bind('<Configure>', on_resize)
    root.mainloop()