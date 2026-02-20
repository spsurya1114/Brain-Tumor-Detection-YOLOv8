from pathlib import Path
from ultralytics import YOLO
from config import YOLO_ROOT_DIR, IMG_SIZE, BATCH_SIZE, EPOCHS, LEARNING_RATE, WORKERS, DEVICE


def train():
    # Ensure data.yaml exists
    yaml_path = Path("yolov8_data/data.yaml")
    if not yaml_path.exists():
        print(f"❌ Error: {yaml_path} not found. Please run dataset.py first.")
        return

    # Load a model
    model = YOLO("yolov8n.pt")  # Switch to Nano model for speed

    # Training arguments with strong augmentation
    train_args = dict(
        data=str(yaml_path),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        lr0=LEARNING_RATE,
        name="brain_tumor_yolov8",
        workers=WORKERS,
        device=DEVICE,
        
        # Optimization
        cache=True,  # Cache images for faster training
        patience=10,  # Early stopping

        # Augmentation Hyperparameters
        mosaic=0.0,      # Disable mosaic for speed
        mixup=0.0,       # Disable mixup for speed
        degrees=15.0,    # image rotation (+/- deg)
        fliplr=0.5,      # image flip left-right
        scale=0.5,       # image scale (+/- gain)
        shear=2.5,       # image shear (+/- deg)
        perspective=0.0005,  # image perspective (+/- fraction)
        hsv_h=0.015,     # image HSV-Hue augmentation
        hsv_s=0.7,       # image HSV-Saturation augmentation
        hsv_v=0.4,       # image HSV-Value augmentation
    )

    print(f"🚀 Starting training with batch={BATCH_SIZE}, workers={WORKERS}...")
    model.train(**train_args)
    print("✅ Training finished. Check runs/detect/brain_tumor_yolov8 for results.")


if __name__ == '__main__':
    # multiprocessing support for Windows
    train()
