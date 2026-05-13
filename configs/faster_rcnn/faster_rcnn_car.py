# Faster R-CNN config for car detection - FINAL VERSION
_base_ = 'C:/Users/LENOVO/Desktop/Car_Detection/mmdetection/configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py'

# Dataset paths - USING RELATIVE PATHS (works anywhere!)
data_root = 'data/'

# Class info
metainfo = {
    'classes': ('car',),
    'palette': [(220, 20, 60)]
}

# Training data
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/instances_train_fixed.json',
        data_prefix=dict(img='images/train/')
    )
)

# Validation data
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/instances_val_fixed.json',
        data_prefix=dict(img='images/val/'),
        test_mode=True
    )
)

# Test data
test_dataloader = val_dataloader

# Evaluation
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/instances_val_fixed.json',
    metric='bbox'
)
test_evaluator = val_evaluator

# Training schedule
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=12,
    val_interval=1
)

# Model settings
model = dict(
    roi_head=dict(
        bbox_head=dict(
            num_classes=1
        )
    )
)


