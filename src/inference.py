import numpy as np
from ultralytics import YOLO
from PIL import Image
import io
from config import MODEL_PATH


class BrainTumorClassifier:
    def __init__(self, model_path=None):
        self.model_path = model_path if model_path else MODEL_PATH
        print(f"Loading model from {self.model_path}...")
        try:
            self.model = YOLO(self.model_path)
            print("✅ Model loaded successfully.")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.model = None

    def predict_image(self, image_bytes: bytes, conf_threshold=0.25):
        """
        Runs inference on a single image provided as bytes.
        Returns a list of detections: {'class': str, 'confidence': float, 'box': [x1, y1, x2, y2]}
        """
        if self.model is None:
            return {"error": "Model not loaded"}

        # Convert bytes to PIL Image
        try:
            image = Image.open(io.BytesIO(image_bytes))
        except Exception as e:
            return {"error": f"Invalid image format: {e}"}

        # Run inference
        results = self.model.predict(
            source=image, 
            conf=conf_threshold, 
            save=False, 
            verbose=False
        )
        
        detections = []
        for result in results:
            for box in result.boxes:
                # Extract info
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].tolist() # Bounding box coordinates
                
                class_name = result.names[cls_id] if result.names else str(cls_id)
                
                detections.append({
                    "class": class_name,
                    "confidence": round(conf, 4),
                    "box": [round(x, 2) for x in xyxy]
                })
        
        return {"predictions": detections}


if __name__ == "__main__":
    # Simple test if run directly
    classifier = BrainTumorClassifier()
    # Mock test if a file exists (optional, mostly for dev verification)
    from config import TEST_PIC_DIR
    test_files = list(TEST_PIC_DIR.glob("*.jpg"))
    if test_files:
        with open(test_files[0], "rb") as f:
            print(classifier.predict_image(f.read()))
import cv2
from heatmap import create_heatmap

img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

boxes = []
for box in result.boxes:
    boxes.append(box.xyxy[0].tolist())

heatmap_overlay = create_heatmap(img_cv, boxes)