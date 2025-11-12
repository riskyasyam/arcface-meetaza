# main.py
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

import base64, io
import numpy as np
from PIL import Image
import cv2
import insightface

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = []
    for e in exc.errors():
        loc = " -> ".join(str(x) for x in e.get("loc", []))
        errors.append(f"{loc}: {e.get('msg','')}")
    return JSONResponse(status_code=422, content={
        "status": "error", "message": "Validation error", "errors": errors
    })

class FaceCompareRequest(BaseModel):
    profile_image: str        # base64
    current_image: str        # base64
    threshold: Optional[float] = None  # default 0.35 kalau None

def decode_base64_to_image(b64: str) -> Image.Image:
    if "," in b64:
        b64 = b64.split(",", 1)[1]
    img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    return img

def pick_largest_face(faces):
    if not faces:
        return None
    return sorted(
        faces,
        key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]),
        reverse=True
    )[0]

print("Loading InsightFace (buffalo_s)…")
face_app = insightface.app.FaceAnalysis(name="buffalo_s")
face_app.prepare(ctx_id=0, det_size=(640, 640))
print("InsightFace loaded ✅")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/compare")
def compare_faces(body: FaceCompareRequest):
    try:
        profile_rgb = np.array(decode_base64_to_image(body.profile_image))
        current_rgb = np.array(decode_base64_to_image(body.current_image))
    except Exception as e:
        return {"status": "error", "message": "Invalid base64 image", "errors": str(e)}

    profile_bgr = cv2.cvtColor(profile_rgb, cv2.COLOR_RGB2BGR)
    current_bgr = cv2.cvtColor(current_rgb, cv2.COLOR_RGB2BGR)

    # detect
    profile_faces = face_app.get(profile_bgr)
    current_faces = face_app.get(current_bgr)

    profile_face = pick_largest_face(profile_faces)
    current_face = pick_largest_face(current_faces)

    if profile_face is None or current_face is None:
        return {"status": "error", "message": "face not detected on one/both images"}

    # embeddings (InsightFace sudah L2-normalized)
    profile_emb = profile_face.normed_embedding
    current_emb = current_face.normed_embedding

    # cosine similarity & (opsional) cosine distance
    sim = float(np.dot(profile_emb, current_emb))
    th = 0.35 if body.threshold is None else float(body.threshold)
    cos_distance = float(1.0 - sim)  # kalau backend lama butuh "distance"

    return {
        "status": "success",
        "message": "Face comparison successful",
        "data": {
            "match": sim >= th,
            "similarity": round(sim, 2),
            "distance": round(cos_distance, 2),
            "threshold_used": th,
        },
    }
