import json
import re

print("=" * 70)
print("FASTER R-CNN VALIDATION METRICS EXTRACTOR")
print("=" * 70)

log_file = "work_dirs/faster_rcnn_car/20260318_232826/20260318_232826.log"
json_file = "work_dirs/faster_rcnn_car/20260318_232826/vis_data/scalars.json"

# Method 1: Extract from log file
print(f"\n📁 Checking log file: {log_file}")
try:
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    print(f"📄 Log has {len(lines)} lines")
    
    # Find validation lines
    val_lines = []
    for line in lines:
        if 'Epoch(val)' in line and 'bbox_mAP' in line:
            val_lines.append(line.strip())
    
    if val_lines:
        print(f"\n✅ Found {len(val_lines)} validation entries in log:")
        for line in val_lines:
            # Extract mAP values
            mAP = re.search(r'bbox_mAP: (0\.\d+)', line)
            mAP50 = re.search(r'bbox_mAP_50: (0\.\d+)', line)
            mAP75 = re.search(r'bbox_mAP_75: (0\.\d+)', line)
            
            print(f"\n  {line[:100]}...")
            if mAP:
                print(f"    → mAP: {mAP.group(1)}")
            if mAP50:
                print(f"    → mAP@0.5: {mAP50.group(1)}")
            if mAP75:
                print(f"    → mAP@0.75: {mAP75.group(1)}")
    else:
        print("❌ No validation entries found in log")
        
except FileNotFoundError:
    print("❌ Log file not found")

# Method 2: Extract from JSON
print(f"\n📁 Checking JSON file: {json_file}")
try:
    with open(json_file, 'r') as f:
        lines = f.readlines()
    
    print(f"📄 JSON has {len(lines)} lines")
    
    # Look for validation metrics in JSON
    val_metrics = []
    for i, line in enumerate(lines):
        try:
            data = json.loads(line)
            if 'bbox_mAP' in data:
                val_metrics.append((i, data))
        except:
            pass
    
    if val_metrics:
        print(f"\n✅ Found {len(val_metrics)} validation entries in JSON:")
        for i, data in val_metrics:
            print(f"\n  Entry {i}:")
            for key in ['bbox_mAP', 'bbox_mAP_50', 'bbox_mAP_75', 'bbox_mAP_s', 'bbox_mAP_m', 'bbox_mAP_l']:
                if key in data:
                    print(f"    {key}: {data[key]}")
    else:
        print("❌ No validation metrics found in JSON")
        
        # Show what IS in the JSON
        print("\n📊 Sample of what's in the JSON (first 3 lines):")
        for line in lines[:3]:
            data = json.loads(line)
            print(f"  Keys: {list(data.keys())}")

except FileNotFoundError:
    print("❌ JSON file not found")

print("\n" + "=" * 70)