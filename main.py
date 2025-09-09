# main.py
import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from io import BytesIO
from PIL import Image
import torch
from transformers import AutoFeatureExtractor, AutoModelForImageClassification

MODEL_ID = os.getenv("HF_MODEL_ID", "march18/FacialConfidence")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI(title="FacialConfidence Inference")

# Load once at startup
try:
    feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID)
    model = AutoModelForImageClassification.from_pretrained(MODEL_ID)
    model.to(DEVICE)
    model.eval()
except Exception as e:
    raise RuntimeError(f"Failed to load model {MODEL_ID}: {e}")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        inputs = feature_extractor(images=image, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()[0]
        labels = model.config.id2label if hasattr(model.config, "id2label") else {i: str(i) for i in range(len(probs))}
        result = [{"label": labels[i], "score": float(probs[i])} for i in range(len(probs))]
        result = sorted(result, key=lambda x: x["score"], reverse=True)
        return {"predictions": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
