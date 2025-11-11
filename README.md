# Face Recognition API (InsightFace + SVM)

Proyek ini berisi dua bagian utama:

1. **Notebook (`train.ipynb`)** — dipakai untuk **eksperimen awal / testing** face recognition secara _static_ (langsung di notebook).
2. **API (`main.py`)** — versi yang sudah dibungkus **FastAPI** supaya bisa diakses dari frontend / service lain (upload foto → dapat nama).

Jadi alurnya: coba dan latih di notebook ➜ pindah ke FastAPI untuk pemakaian dinamis.

---

## 1. Konsep Utama

- **InsightFace (`buffalo_s`)** dipakai untuk **deteksi wajah + ekstraksi embedding**.
- **SVM (Scikit-learn)** dipakai untuk **mengklasifikasikan** embedding tadi ke nama orang.
- Embedding disimpan dalam bentuk **`.npy`** supaya bisa nambah orang **tanpa retrain dari nol**.
- Endpoint FastAPI disiapkan untuk:
  - daftar wajah baru (`/register-face`)
  - latih ulang SVM dari dataset dinamis (`/train-svm`)
  - kenali wajah (`/recognize`)

---

## 2. Bagian Notebook: `train.ipynb`

Notebook ini sifatnya **test + static**. Biasanya isinya:

- load model InsightFace
- load beberapa gambar contoh
- deteksi wajah → ambil `normed_embedding`
- latihan SVM pertama kali dari dataset yang sudah disiapkan
- simpan model jadi `svm_face.pkl`

Jadi ini versi “laboratorium”-nya: kamu pastikan pipeline jalan dulu di notebook sebelum dipindah ke API.

Output penting dari notebook:
- `svm_face.pkl` → model hasil training pertama
- mungkin juga folder dataset awal (kalau kamu buat di notebook)

---

## 3. Bagian API: `main.py`

File ini adalah versi **production / service**. Fungsinya:

- load model InsightFace sekali di awal
- load `svm_face.pkl` sekali di awal
- sediakan endpoint buat:
  - cek: `GET /health`
  - daftar embedding baru: `POST /register-face`
  - retrain dari embedding baru: `POST /train-svm`
  - kenali wajah: `POST /recognize`

### 3.1. Jalankan API

Aktifkan env kamu dulu, lalu:

```bash
uvicorn main:app --reload