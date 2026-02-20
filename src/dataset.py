import os
import shutil
import glob
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from config import BASE_DATA_DIR, YOLO_ROOT_DIR, CLASS_NAMES

def create_yolo_dirs(yolo_root):
    """Creates the necessary target directories for YOLOv8."""
    if yolo_root.exists():
        shutil.rmtree(yolo_root) # Clean start
    
    for split in ['train', 'val']:
        for sub_dir in ['images', 'labels']:
            path = yolo_root / sub_dir / split
            path.mkdir(parents=True, exist_ok=True)
            print(f"Created: {path}")

def copy_file(args):
    """Helper function for parallel copying."""
    src, dst = args
    shutil.copy(src, dst)

def restructure_dataset_parallel(base_dir, yolo_root):
    print("\n--- Starting Data Restructuring (Parallel) ---")
    
    # Map 'Train'/'Val' folders to 'train'/'val' splits
    splits = {'Train': 'train', 'Val': 'val'}
    
    tasks = []
    
    for source_split, target_split in splits.items():
        for class_name in CLASS_NAMES:
            source_class_path = base_dir / source_split / class_name
            # Check for slight variations in case sensitivity if needed, but assuming exact match for now based on list_dir
            
            if not source_class_path.exists():
                 # Try matching case-insensitive if exact match fails
                found = False
                for p in (base_dir / source_split).iterdir():
                     if p.name.lower() == class_name.lower():
                         source_class_path = p
                         found = True
                         break
                if not found:
                    print(f"⚠️ Warning: Source path not found: {source_class_path}")
                    continue

            source_images_path = source_class_path / 'images'
            source_labels_path = source_class_path / 'labels'

            dest_images_path = yolo_root / 'images' / target_split
            dest_labels_path = yolo_root / 'labels' / target_split

            if not source_images_path.exists():
                print(f"⚠️ Missing images folder: {source_images_path}")
                continue

            # Collect all image files
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.png']:
                image_files.extend(list(source_images_path.glob(ext)))
            
            print(f"Processing {len(image_files)} images for {class_name} ({target_split})...")

            for img_path in image_files:
                filename = img_path.name
                # prefix with class name to avoid collisions
                safe_class = class_name.replace(" ", "_")
                new_name = f"{safe_class}__{filename}"
                
                # Image copy task
                tasks.append((img_path, dest_images_path / new_name))
                
                # Label copy task
                lbl_file = img_path.stem + ".txt"
                lbl_src = source_labels_path / lbl_file
                if lbl_src.exists():
                    lbl_dst = dest_labels_path / (Path(new_name).stem + ".txt")
                    tasks.append((lbl_src, lbl_dst))

    # Execute copy operations in parallel
    print(f"Copying {len(tasks)} files...")
    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(copy_file, tasks))
        
    print("✅ Data restructuring complete.")

def create_data_yaml(yolo_root, class_names):
    # Absolute path is safer for YOLO
    abs_root = yolo_root.resolve()
    
    yaml_content = f"""
# YOLOv8 Data Configuration
path: {abs_root}
train: images/train
val: images/val

nc: {len(class_names)}
names: {[c.replace(" ", "_").lower() for c in class_names]}
"""
    yaml_path = yolo_root / 'data.yaml'
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    print(f"\n✅ Created data.yaml at: {yaml_path}")
    return yaml_path

if __name__ == "__main__":
    create_yolo_dirs(YOLO_ROOT_DIR)
    restructure_dataset_parallel(BASE_DATA_DIR, YOLO_ROOT_DIR)
    create_data_yaml(YOLO_ROOT_DIR, CLASS_NAMES)
