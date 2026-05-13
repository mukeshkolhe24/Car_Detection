# YOLO config for car detection - WITH CORRECT PATH
_base_ = 'C:/Users/LENOVO/Desktop/Car_Detection/mmdetection/configs/yolo/yolov3_d53_8xb8-ms-608-273e_coco.py'

# 1. Dataset paths
data_root = 'C:/Users/LENOVO/Desktop/Car_Detection/data/'

# 2. Class info (only cars)
metainfo = {
    'classes': ('car',),
    'palette': [(220, 20, 60)]  # Red color for boxes
}

# 3. Training data settings
train_dataloader = dict(
    batch_size=4,  # YOLO can handle larger batches
    num_workers=2,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/instances_train_fixed.json',
        data_prefix=dict(img='images/train/')
    )
)

# 4. Validation data settings
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/instances_val_fixed.json',
        data_prefix=dict(img='images/val/')
    )
)

# 5. Test data settings
test_dataloader = val_dataloader

# 6. Evaluation settings
val_evaluator = dict(ann_file=data_root + 'annotations/instances_val.json')
test_evaluator = val_evaluator

# 7. Training schedule (YOLO needs more epochs)
train_cfg = dict(
    max_epochs=50,  # YOLO needs more training
    val_interval=5   # Validate every 5 epochs
)

# 8. Model settings (only 1 class: car)
model = dict(
    bbox_head=dict(
        num_classes=1
    )
)