import cv2
import matplotlib.pyplot as plt
from mmdet.apis import init_detector, inference_detector
import os
import time

print("=" * 60)
print("FASTER R-CNN FINAL MODEL TESTER")
print("=" * 60)

# Load your newly trained model
print("\n📥 Loading Faster R-CNN model (fully trained)...")
model = init_detector(
    "configs/faster_rcnn/faster_rcnn_car.py",
    "work_dirs/faster_rcnn_car/epoch_12.pth",  # Your fully trained model!
    device='cuda:0'
)
print("✅ Model loaded successfully!")

while True:
    print("\n" + "-" * 40)
    image_name = input("Enter image filename (or 'quit' to exit): ").strip()
    
    if image_name.lower() == 'quit':
        break
    
    # Check multiple possible locations
    possible_paths = [
        f"data/images/val/{image_name}",
        f"data/images/train/{image_name}",
        f"data/images/test/{image_name}",
        f"data/{image_name}"
    ]
    
    img_path = None
    for path in possible_paths:
        if os.path.exists(path):
            img_path = path
            break
    
    if img_path is None:
        print(f"❌ Image not found: {image_name}")
        print("   Checked locations:")
        for path in possible_paths:
            print(f"   - {path}")
        continue
    
    print(f"\n📸 Testing on: {img_path}")
    
    # Measure inference time
    start_time = time.time()
    result = inference_detector(model, img_path)
    inference_time = time.time() - start_time
    
    # Read image
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_detected = img_rgb.copy()
    
    # Get detections
    detections = []
    if hasattr(result, 'pred_instances'):
        pred = result.pred_instances
        if hasattr(pred, 'bboxes') and len(pred.bboxes) > 0:
            bboxes = pred.bboxes.cpu().numpy()
            scores = pred.scores.cpu().numpy()
            for bbox, score in zip(bboxes, scores):
                detections.append((bbox, score))
    
    # Draw boxes
    if len(detections) > 0:
        print(f"\n✅ Found {len(detections)} cars:")
        for i, (bbox, score) in enumerate(detections):
            x1, y1, x2, y2 = map(int, bbox[:4])
            print(f"   Car {i+1}: confidence={score:.3f}")
            
            # Green box with confidence
            cv2.rectangle(img_detected, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_detected, f"{score:.2f}", (x1, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    else:
        print("\n❌ No cars detected")
    
    # Show inference time
    print(f"\n⏱️ Inference time: {inference_time*1000:.1f} ms")
    
    # Show result
    plt.figure(figsize=(12, 8))
    plt.imshow(img_detected)
    plt.title(f"Faster R-CNN (Fully Trained) - {len(detections)} cars detected", fontsize=14)
    plt.axis('off')
    
    # Save
    os.makedirs("results", exist_ok=True)
    output_name = f"results/frcnn_final_{image_name.replace('.jpg','')}.png"
    plt.savefig(output_name, dpi=150, bbox_inches='tight')
    print(f"\n✅ Saved to: {output_name}")
    plt.show()

print("\n" + "=" * 60)
print("👋 Testing complete!")
print("=" * 60)