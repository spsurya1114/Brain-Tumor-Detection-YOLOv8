try:
    print("✅ Checking imports...")
    from ultralytics import YOLO
    print("✅ YOLO class imported")
    import torch
    print(f"✅ Torch version: {torch.__version__}")
    print(f"✅ CUDA available: {torch.cuda.is_available()}")

    model = YOLO("yolov8n.pt")
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
