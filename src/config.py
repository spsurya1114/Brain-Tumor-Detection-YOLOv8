from pathlib import Path

# --- Configuration ---
# Define base paths
BASE_DATA_DIR = Path("data")  # Path to the 'data' folder
YOLO_ROOT_DIR = Path("yolov8_data")  # Path where YOLOv8 formatted data will be stored
TEST_PIC_DIR = Path("test_pic")  # Path to test images

# Define classes in the dataset
CLASS_NAMES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

# Training Hyperparameters
IMG_SIZE = 320   # Reduced from 416 for speed
BATCH_SIZE = 8  # Adjust based on GPU VRAM (16 or 32 recommended)
EPOCHS = 10      # Increased for better convergence
LEARNING_RATE = 0.001
WORKERS = 2    # Number of data loader workers
DEVICE = "cpu"

# Inference
# Default path to the best trained model. 
# In production, this might be overwritten by an env var or the file might be moved to a standard 'models/' dir.
MODEL_PATH = Path("runs/detect/brain_tumor_yolov84/weights/best.pt") 

