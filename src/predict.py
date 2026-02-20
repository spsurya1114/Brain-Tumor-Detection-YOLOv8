from ultralytics import YOLO
import glob
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from config import TEST_PIC_DIR

def predict():
    # Locate the best model
    # Note: Adjust path if you have multiple runs. This finds the latest 'brain_tumor_yolov8' run.
    run_dir = Path("runs/detect")
    candidates = sorted(run_dir.glob("brain_tumor_yolov8*"), key=lambda p: p.stat().st_mtime, reverse=True)
    
    if not candidates:
        print("❌ No training runs found in runs/detect/. Please train the model first.")
        return
    
    best_weights = candidates[0] / "weights" / "best.pt"
    if not best_weights.exists():
        print(f"⚠️ 'best.pt' not found in {candidates[0]}. Using 'last.pt' if available.")
        best_weights = candidates[0] / "weights" / "last.pt"
    
    if not best_weights.exists():
        print("❌ No weights found.")
        return

    print(f"Using model: {best_weights}")
    model = YOLO(str(best_weights))

    # Process test images
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG']
    test_images = []
    for ext in image_extensions:
        test_images.extend(TEST_PIC_DIR.glob(ext))

    if not test_images:
        print(f"⚠️ No images found in {TEST_PIC_DIR}")
        return

    print(f"Found {len(test_images)} images for prediction.")

    # Create output directory for predictions
    output_dir = Path("predictions")
    output_dir.mkdir(exist_ok=True)

    for img_path in test_images:
        print(f"Predicting: {img_path.name}")
        results = model.predict(source=str(img_path), save=True, project="predictions", name="results", exist_ok=True)
        
        # Optional: Display or process results further
        # for r in results:
        #     r.show() # Attempt to show image window (may not work in all envs)

    print(f"✅ Predictions saved to {output_dir}/results")

if __name__ == "__main__":
    predict()
