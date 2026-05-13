import json
import os
import glob

print("=" * 60)
print("FINDING VALIDATION METRICS")
print("=" * 60)

# Look for all JSON files in vis_data folders
json_files = glob.glob("work_dirs/**/vis_data/*.json", recursive=True)

print(f"Found {len(json_files)} JSON files:")
for json_file in json_files:
    print(f"  - {json_file}")

print("\n" + "-" * 60)

# Check each file for validation metrics
for json_file in json_files:
    print(f"\n📁 Checking: {json_file}")
    
    with open(json_file, 'r') as f:
        lines = f.readlines()
    
    val_metrics = []
    for line in lines:
        try:
            data = json.loads(line)
            # Look for validation metrics (often have 'bbox_mAP' or 'coco/bbox_mAP')
            if 'bbox_mAP' in data or 'coco/bbox_mAP' in data:
                val_metrics.append(data)
        except:
            pass
    
    if val_metrics:
        print(f"✅ Found {len(val_metrics)} validation entries!")
        for i, metrics in enumerate(val_metrics[-5:]):  # Show last 5
            print(f"\n  Entry {i+1}:")
            for key, value in metrics.items():
                if 'mAP' in key:
                    print(f"    {key}: {value}")
    else:
        print("❌ No validation metrics found in this file")