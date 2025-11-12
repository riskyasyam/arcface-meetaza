# main.py
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware

import io
import os
import base64

from PIL import Image
import numpy as np
import cv2
import insightface

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = []
    for error in exc.errors():
        error_msg = error.get("msg", "")
        error_loc = " -> ".join(str(loc) for loc in error.get("loc", []))
        errors.append(f"{error_loc}: {error_msg}")

    return JSONResponse(
        status_code=422,
        content={
            "status": "error",
            "message": "Validation error",
            "errors": errors,
        },
    )

class FaceCompareRequest(BaseModel):
    reference_image: str  # base64
    target_image: str     # base64
    threshold: Optional[float] = None  # kalau mau override threshold

class RegisterFaceRequest(BaseModel):
    name: str
    image_base64: str

class RecognizeRequest(BaseModel):
    image_base64: str
    threshold: Optional[float] = None


def decode_base64_to_image(base64_string: str) -> Image.Image:
    try:
        if "," in base64_string:
            base64_string = base64_string.split(",", 1)[1]
        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        return image
    except Exception as e:
        raise ValueError(f"Invalid base64 image: {str(e)}")


DATASET_DIR = "dataset_dynamic"
os.makedirs(DATASET_DIR, exist_ok=True)

print("Loading InsightFace model ...")
face_app = insightface.app.FaceAnalysis(name="buffalo_s")
# ctx_id = 0 -> GPU, -1 -> CPU
face_app.prepare(ctx_id=0, det_size=(640, 640))
print("InsightFace loaded ✅")


def preload_test_images():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dummy1 = os.path.join(current_dir, "dataset_static/billie/dummy1.jpg")
    dummy2 = os.path.join(current_dir, "dataset_static/madison/dummy2.jpg")

    if os.path.exists(dummy1) and os.path.exists(dummy2):
        print("Found dummy images, testing InsightFace...")
        img1 = cv2.cvtColor(np.array(Image.open(dummy1).convert("RGB")), cv2.COLOR_RGB2BGR)
        img2 = cv2.cvtColor(np.array(Image.open(dummy2).convert("RGB")), cv2.COLOR_RGB2BGR)
        faces1 = face_app.get(img1)
        faces2 = face_app.get(img2)
        if faces1 and faces2:
            emb1 = faces1[0].normed_embedding
            emb2 = faces2[0].normed_embedding
            sim = float(np.dot(emb1, emb2))
            print(f"Preload test similarity: {sim:.4f}")
        else:
            print("Dummy images do not contain detectable faces.")
    else:
        print("No dummy images found, skipping preload test.")


preload_test_images()


def load_all_embeddings():
    db = []
    for file in os.listdir(DATASET_DIR):
        if file.endswith(".npy"):
            path = os.path.join(DATASET_DIR, file)
            emb = np.load(path)
            label = os.path.splitext(file)[0]
            db.append((label, emb))
    return db

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/register-face")
async def register_face_json(body: RegisterFaceRequest):
    try:
        pil_img = decode_base64_to_image(body.image_base64)
    except Exception as e:
        return {
            "status": "error",
            "message": "Invalid image_base64",
            "errors": str(e),
        }

    img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    faces = face_app.get(img_bgr)
    if len(faces) == 0:
        return {"status": "error", "message": "no face detected"}

    emb = faces[0].normed_embedding  # vector 512 dim normalized

    save_path = os.path.join(DATASET_DIR, f"{body.name}.npy")
    np.save(save_path, emb)

    return {
        "status": "success",
        "message": f"face of {body.name} registered",
    }

@app.post("/compare-insight")
async def compare_insight(body: FaceCompareRequest):
    # --- decode reference ---
    try:
        ref_pil = decode_base64_to_image(body.reference_image)
    except Exception as e:
        return {
            "status": "error",
            "message": "Invalid reference image",
            "errors": str(e),
        }

    # --- decode target ---
    try:
        tgt_pil = decode_base64_to_image(body.target_image)
    except Exception as e:
        return {
            "status": "error",
            "message": "Invalid target image",
            "errors": str(e),
        }

    ref_bgr = cv2.cvtColor(np.array(ref_pil), cv2.COLOR_RGB2BGR)
    tgt_bgr = cv2.cvtColor(np.array(tgt_pil), cv2.COLOR_RGB2BGR)

    ref_faces = face_app.get(ref_bgr)
    tgt_faces = face_app.get(tgt_bgr)

    if len(ref_faces) == 0 or len(tgt_faces) == 0:
        return {
            "status": "error",
            "message": "face not detected on one/both images",
        }

    ref_emb = ref_faces[0].normed_embedding
    tgt_emb = tgt_faces[0].normed_embedding

    similarity = float(np.dot(ref_emb, tgt_emb))
    threshold = body.threshold if body.threshold is not None else 0.35
    is_match = similarity >= threshold

    return {
        "status": "success",
        "message": "Face comparison successful",
        "data": {
            "match": is_match,
            "similarity": round(similarity, 4),
            "threshold_used": threshold,
        },
    }
 
@app.post("/recognize-face")
async def recognize_face(body: RecognizeRequest):
    try:
        pil_img = decode_base64_to_image(body.image_base64)
    except Exception as e:
        return {
            "status": "error",
            "message": "Invalid image_base64",
            "errors": str(e),
        }

    img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    faces = face_app.get(img_bgr)
    if len(faces) == 0:
        return {"status": "error", "message": "no face detected"}

    probe_emb = faces[0].normed_embedding

    db = load_all_embeddings()
    if len(db) == 0:
        return {"status": "error", "message": "no registered faces in database"}

    best_label = None
    best_score = -1.0

    for label, emb in db:
        score = float(np.dot(probe_emb, emb))
        if score > best_score:
            best_score = score
            best_label = label

    threshold = body.threshold if body.threshold is not None else 0.35
    is_match = best_score >= threshold

    return {
        "status": "success",
        "message": "Face recognition finished",
        "data": {
            "predicted_label": best_label if is_match else "unknown",
            "similarity": round(best_score, 4),
            "threshold_used": threshold,
            "matched": is_match,
        },
    }