# main.py
import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
from PIL import Image
import torch
from transformers import AutoFeatureExtractor, AutoModelForImageClassification

MODEL_ID = os.getenv("HF_MODEL_ID", "march18/FacialConfidence")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# threshold to decide confident (class 0 probability)
THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))

app = FastAPI(title="FacialConfidence Inference (minimal)")

# --- CORS (adjust origins in production) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],            # <- change to your domain(s) in production
    allow_credentials=True,
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["*"],
)

# --- load model once at startup ---
try:
    feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID)
    model = AutoModelForImageClassification.from_pretrained(MODEL_ID)
    model.to(DEVICE)
    model.eval()
except Exception as e:
    raise RuntimeError(f"Failed to load model {MODEL_ID}: {e}")

# mapping
LABELS = {0: "confident", 1: "unconfident"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Expect multipart/form-data with key 'file' containing the image.
    Response example:
      { "predictions": [ { "class": 0, "label": "confident" } ] }
    """
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")

        inputs = feature_extractor(images=image, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()[0]

        # As you specified: class 0 = confident, class 1 = unconfident
        # Choose the predicted class as argmax of probs
        pred_class = int(probs.argmax())
        pred_label = LABELS.get(pred_class, str(pred_class))

        # If you prefer threshold-based decision for class 0:
        # confidence_of_class0 = float(probs[0])
        # pred_class = 0 if confidence_of_class0 >= THRESHOLD else 1
        # pred_label = LABELS[pred_class]

        return {"predictions": [{"class": pred_class, "label": pred_label}]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
