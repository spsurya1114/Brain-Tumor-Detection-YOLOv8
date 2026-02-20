from fastapi import FastAPI, UploadFile, File, HTTPException
import uvicorn
from contextlib import asynccontextmanager
from inference import BrainTumorClassifier

# Global model instance
classifier = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model on startup
    global classifier
    classifier = BrainTumorClassifier()
    yield
    # Clean up (if needed)


app = FastAPI(title="Brain Tumor MRI Detection API", lifespan=lifespan)


@app.get("/")
def home():
    return {
        "message": (
            "Brain Tumor MRI Detection API is running. "
            "Use /predict to analyze images."
        )
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image only.")

    try:
        content = await file.read()
        result = classifier.predict_image(content)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
