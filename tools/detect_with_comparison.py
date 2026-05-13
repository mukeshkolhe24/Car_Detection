import cv2
import matplotlib.pyplot as plt
from mmdet.apis import init_detector, inference_detector
import os

print("=" * 60)
print("COMPARING FASTER R-CNN vs YOLO")
print("=" * 60)

# Load both models
print("\n📥 Loading models...")
frcnn_model = init_detector(
    "configs/faster_rcnn/faster_rcnn_car.py",
    "work_dirs/faster_rcnn_car/epoch_12.pth",
    device='cuda:0'
)

yolo_model = init_detector(
    "configs/yolo/yolo_car.py",
    "work_dirs/yolo_car/epoch_50.pth",
    device='cuda:0'
)
print("✅ Both models loaded!")

while True:
    print("\n" + "-" * 40)
    image_name = input("Enter image filename (or 'quit' to exit): ").strip()
    
    if image_name.lower() == 'quit':
        break
    
    # Find image
    possible_paths = [
        f"data/images/val/{image_name}",
        f"data/images/train/{image_name}",
        f"data/training_images/{image_name}",
        f"data/testing_images/{image_name}"
    ]
    
    img_path = None
    for path in possible_paths:
        if os.path.exists(path):
            img_path = path
            break
    
    if img_path is None:
        print(f"❌ Image not found: {image_name}")
        continue
    
    print(f"\n📸 Testing on: {img_path}")
    
    # Read image
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original image
    axes[0].imshow(img_rgb)
    axes[0].set_title("Original Image", fontsize=14)
    axes[0].axis('off')
    
    # Faster R-CNN detection
    frcnn_result = inference_detector(frcnn_model, img_path)
    img_frcnn = img_rgb.copy()
    frcnn_dets = []
    
    if hasattr(frcnn_result, 'pred_instances'):
        pred = frcnn_result.pred_instances
        if hasattr(pred, 'bboxes') and len(pred.bboxes) > 0:
            bboxes = pred.bboxes.cpu().numpy()
            scores = pred.scores.cpu().numpy()
            for bbox, score in zip(bboxes, scores):
                frcnn_dets.append((bbox, score))
    
    for bbox, score in frcnn_dets:
        x1, y1, x2, y2 = map(int, bbox[:4])
        cv2.rectangle(img_frcnn, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_frcnn, f"{score:.2f}", (x1, y1-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    axes[1].imshow(img_frcnn)
    axes[1].set_title(f"Faster R-CNN: {len(frcnn_dets)} cars", fontsize=14)
    axes[1].axis('off')
    
    # YOLO detection
    yolo_result = inference_detector(yolo_model, img_path)
    img_yolo = img_rgb.copy()
    yolo_dets = []
    
    if hasattr(yolo_result, 'pred_instances'):
        pred = yolo_result.pred_instances
        if hasattr(pred, 'bboxes') and len(pred.bboxes) > 0:
            bboxes = pred.bboxes.cpu().numpy()
            scores = pred.scores.cpu().numpy()
            for bbox, score in zip(bboxes, scores):
                yolo_dets.append((bbox, score))
    
    for bbox, score in yolo_dets:
        x1, y1, x2, y2 = map(int, bbox[:4])
        cv2.rectangle(img_yolo, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(img_yolo, f"{score:.2f}", (x1, y1-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    axes[2].imshow(img_yolo)
    axes[2].set_title(f"YOLO: {len(yolo_dets)} cars", fontsize=14)
    axes[2].axis('off')
    
    plt.suptitle(f"Model Comparison - {image_name}", fontsize=16)
    plt.tight_layout()
    
    # Save
    output_name = f"results/comparison_{image_name.replace('.jpg','')}.png"
    plt.savefig(output_name, dpi=150, bbox_inches='tight')
    print(f"\n✅ Comparison saved to: {output_name}")
    plt.show()
    
    # Print summary
    print(f"\n📊 RESULTS SUMMARY:")
    print(f"   Faster R-CNN: {len(frcnn_dets)} cars detected")
    print(f"   YOLO: {len(yolo_dets)} cars detected")

print("\n" + "=" * 60)
print("👋 Comparison Complete!")
print("=" * 60)