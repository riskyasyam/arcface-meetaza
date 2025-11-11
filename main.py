# main.py
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import io
from PIL import Image
import numpy as np
import cv2
import insightface
import pickle
import os
from sklearn.svm import SVC

app = FastAPI(
    title="Face Recognition API",
    description="API pengenalan wajah pakai InsightFace (buffalo_s) + SVM (form-data)",
    version="1.1.0",
)

# ====== CORS ======
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

face_app = insightface.app.FaceAnalysis(name="buffalo_s")
# ctx_id = 0 → GPU | -1 → CPU
face_app.prepare(ctx_id=0, det_size=(640, 640))

SVM_PATH = "svm_face.pkl"
if not os.path.exists(SVM_PATH):
    raise FileNotFoundError("❌ svm_face.pkl tidak ditemukan. Jalankan /train-svm atau siapkan model terlebih dulu.")

with open(SVM_PATH, "rb") as f:
    svm_model = pickle.load(f)


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/recognize")
async def recognize(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        faces = face_app.get(img)
        if len(faces) == 0:
            return {"status": "error", "message": "no face detected"}

        emb = faces[0].normed_embedding

        pred_label = svm_model.predict([emb])[0]
        prob = float(max(svm_model.predict_proba([emb])[0]))

        return {
            "status": "success",
            "person": pred_label,
            "confidence": round(prob, 4),
        }

    except Exception as e:
        return {"status": "error", "message": f"Processing failed: {str(e)}"}


@app.post("/register-face")
async def register_face(name: str = Form(...), file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    faces = face_app.get(img)
    if len(faces) == 0:
        return {"status": "error", "message": "no face detected"}

    emb = faces[0].normed_embedding

    os.makedirs("dataset_dynamic", exist_ok=True)
    with open(f"dataset_dynamic/{name}.npy", "wb") as f:
        np.save(f, emb)

    return {"status": "success", "message": f"face of {name} registered"}


@app.post("/train-svm")
def train_svm():
    dataset_dir = "dataset_dynamic"
    if not os.path.exists(dataset_dir):
        return {"status": "error", "message": "dataset_dynamic folder not found"}

    X, y = [], []
    for file in os.listdir(dataset_dir):
        if file.endswith(".npy"):
            path = os.path.join(dataset_dir, file)
            emb = np.load(path)
            label = os.path.splitext(file)[0]
            X.append(emb)
            y.append(label)

    if len(X) < 2:
        return {"status": "error", "message": "Need at least 2 faces to train SVM"}

    X = np.array(X)
    y = np.array(y)

    model = SVC(kernel="linear", probability=True)
    model.fit(X, y)

    with open("svm_face.pkl", "wb") as f:
        pickle.dump(model, f)

    global svm_model
    svm_model = model

    return {"status": "success", "message": f"SVM trained with {len(y)} faces"}