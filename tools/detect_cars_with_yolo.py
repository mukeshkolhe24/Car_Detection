import cv2
import matplotlib.pyplot as plt
from mmdet.apis import init_detector, inference_detector
import os
import torch

print("=" * 60)
print("YOLO CAR DETECTION")
print("=" * 60)

# Load YOLO model
print("\n📥 Loading YOLO model...")
model = init_detector(
    "configs/yolo/yolo_car.py",
    "work_dirs/yolo_car/epoch_50.pth",  # YOLO trained for 50 epochs
    device='cuda:0'
)
print("✅ YOLO model loaded successfully!")

while True:
    print("\n" + "-" * 40)
    image_name = input("Enter image filename (or 'quit' to exit): ").strip()
    
    if image_name.lower() == 'quit':
        break
    
    # Try different possible locations
    possible_paths = [
        f"data/images/val/{image_name}",
        f"data/images/train/{image_name}",
        f"data/training_images/{image_name}",
        f"data/testing_images/{image_name}",
        f"data/{image_name}",
        image_name
    ]
    
    img_path = None
    for path in possible_paths:
        if os.path.exists(path):
            img_path = path
            break
    
    if img_path is None:
        print(f"❌ Image not found: {image_name}")
        print("   Checked locations:")
        for path in possible_paths[:4]:
            print(f"   - {path}")
        continue
    
    print(f"\n📸 Testing on: {img_path}")
    
    # Run detection
    result = inference_detector(model, img_path)
    
    # Read image
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_detected = img_rgb.copy()
    
    # Get detections (YOLO format)
    detections = []
    if hasattr(result, 'pred_instances'):
        pred = result.pred_instances
        if hasattr(pred, 'bboxes') and len(pred.bboxes) > 0:
            bboxes = pred.bboxes.cpu().numpy()
            scores = pred.scores.cpu().numpy()
            for bbox, score in zip(bboxes, scores):
                detections.append((bbox, score))
    elif isinstance(result, list) and len(result) > 0 and len(result[0]) > 0:
        for det in result[0]:
            if len(det) >= 5:
                bbox = det[:4]
                score = det[4]
                detections.append((bbox, score))
    
    # Draw boxes (YOLO in RED)
    if len(detections) > 0:
        print(f"\n✅ YOLO detected {len(detections)} cars:")
        for i, (bbox, score) in enumerate(detections):
            x1, y1, x2, y2 = map(int, bbox[:4])
            print(f"   Car {i+1}: confidence={score:.3f} at [{x1},{y1},{x2},{y2}]")
            
            # RED boxes for YOLO
            cv2.rectangle(img_detected, (x1, y1), (x2, y2), (255, 0, 0), 3)
            cv2.putText(img_detected, f"YOLO: {score:.2f}", (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    else:
        print("\n❌ YOLO detected no cars")
    
    # Show result
    plt.figure(figsize=(12, 8))
    plt.imshow(img_detected)
    plt.title(f"YOLO Detection - {image_name} ({len(detections)} cars)", fontsize=14)
    plt.axis('off')
    
    # Save
    output_name = f"results/yolo_{image_name.replace('.jpg','')}.png"
    plt.savefig(output_name, dpi=150, bbox_inches='tight')
    print(f"\n✅ Saved to: {output_name}")
    plt.show()

print("\n" + "=" * 60)
print("👋 YOLO Detection Complete!")
print("=" * 60)