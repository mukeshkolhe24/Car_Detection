import json
import os

print("=" * 60)
print("FIXING YOLO VALIDATION ANNOTATIONS")
print("=" * 60)

# Fix training annotations
train_json = "data/annotations/instances_train.json"
val_json = "data/annotations/instances_val.json"
train_img_folder = "data/images/train"
val_img_folder = "data/images/val"

def fix_annotations(json_file, img_folder, split_name):
    print(f"\n📁 Fixing {split_name} annotations...")
    
    # Check if files exist
    if not os.path.exists(json_file):
        print(f"   ❌ JSON file not found: {json_file}")
        return None
        
    if not os.path.exists(img_folder):
        print(f"   ❌ Image folder not found: {img_folder}")
        # Try alternative paths
        alt_folders = [
            "data/training_images",
            "data/train",
            "data/car_dataset/train",
            "data/images"
        ]
        for alt in alt_folders:
            if os.path.exists(alt):
                img_folder = alt
                print(f"   ✅ Found images at: {alt}")
                break
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    print(f"   Original: {len(data['images'])} images, {len(data['annotations'])} annotations")
    
    # Check which images exist
    valid_images = []
    valid_image_ids = set()
    missing_count = 0
    
    for img in data['images']:
        img_path = os.path.join(img_folder, img['file_name'])
        if os.path.exists(img_path):
            valid_images.append(img)
            valid_image_ids.add(img['id'])
        else:
            missing_count += 1
            if missing_count <= 5:  # Show first 5 missing
                print(f"   ❌ Missing: {img['file_name']}")
    
    # Keep only annotations for existing images
    valid_annotations = [ann for ann in data['annotations'] if ann['image_id'] in valid_image_ids]
    
    # Update
    data['images'] = valid_images
    data['annotations'] = valid_annotations
    
    # Save fixed version
    output_file = json_file.replace('.json', '_fixed.json')
    with open(output_file, 'w') as f:
        json.dump(data, f)
    
    print(f"   ✅ Fixed: {len(valid_images)} images, {len(valid_annotations)} annotations")
    print(f"   ✅ Removed {missing_count} missing images")
    print(f"   ✅ Saved to: {output_file}")
    
    return output_file

# Fix both train and val
print("\n📂 Checking your folder structure...")
print("\nContents of data folder:")
if os.path.exists("data"):
    for item in os.listdir("data"):
        print(f"   - {item}")
else:
    print("   ❌ data folder not found!")

train_fixed = fix_annotations(train_json, train_img_folder, "TRAIN")
val_fixed = fix_annotations(val_json, val_img_folder, "VAL")

print("\n" + "=" * 60)
if train_fixed and val_fixed:
    print("✅ FIXED FILES CREATED SUCCESSFULLY!")
    print(f"   Train: {train_fixed}")
    print(f"   Val: {val_fixed}")
else:
    print("⚠️ Some files couldn't be fixed. Check the errors above.")
print("=" * 60)